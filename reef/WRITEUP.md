# Streaming KV eviction on Reef's workload (TinyStories-28M)

*Akash · October 2026 · code, tests and raw JSONL in this folder; `python -m reef.reproduce` regenerates every number and figure*

## Question

The TIU evicts continuously during decode by H2O score (accumulated attention mass, argmin
victim). In SubCache, H2O compressed *once* after prefill collapsed on retrieval at small
budgets. Does it drop early-but-important tokens when it evicts **every step**, on Reef's actual
model, and is a cheap alternative better?

## Findings

1. **With a protected recent window, H2O holds up.** On *recall tokens* (names and content words
   reused ≥ 32 tokens after their last mention, the retrieval analog), H2O is within 4 points of
   the best policy at every budget. The budget dominates: vs full cache, every score-based policy
   loses 18–24 recall points at 32 slots per (layer, head), 6–9 at 64, 1–3 at 128.
2. **Without the recent window, H2O and the TIU collapse at every budget**: recall 5–30%,
   perplexity 6–66x full cache, even at 128 slots. A new token's single row of attention cannot
   outscore older tokens' accumulated mass, so it is evicted on arrival. **The TIU's recent-window
   wrapper is load-bearing.**
3. **EMA (score ← α·score + attention, α = 0.9) is modestly better and needs no wrapper.** With
   protection: +2.9 / +3.9 / +1.5 recall points over H2O at 16 / 32 / 64 slots (paired 95% CIs
   exclude 0) and 1.26x vs 1.43x full-cache perplexity at 32; at 128 it is 0.8 points worse (CI
   −1.4 to −0.1). Without any protected window it matches protected H2O at 64 slots (59.8% vs
   59.8%). Hardware cost: one shift-subtract per slot if α = 1 − 2⁻ᵏ (0.875 untested).
4. **TIU numerics:** mapping attention 1.0 → weight 255 saturates the 8-bit score; 18–28% of
   evictions at 16–64 slots pick among candidates all stuck at 255 (ties → lowest slot index),
   costing ~2 recall points at 16–32 slots and nothing at 64–128. Mapping 1.0 → 32 (a 3-bit
   shift) matches float H2O at every budget (|Δ| ≤ 0.6 points).
5. **One keep set per layer costs recall at tight budgets** (H2O: −10.2 points at 32 slots,
   −4.8 at 64, −1.3 at 128, vs per-head).
6. **Sliding window + sinks is worst on recall** (−19 points vs full at 64 slots; H2O −8),
   though it loses only 3.8 points of overall accuracy there. Recall tokens are where eviction
   shows: H2O at 64 slots loses 2.3 points overall but 7.9 on recall tokens.
7. **Robustness:** with simulated int8 weights every recall gap to full cache moves by ≤ 0.4
   points and the ranking is unchanged. Evicting global layers only (local layers keep their 256 window)
   cuts H2O's loss at 32 slots from 21.8 to 6.4 points, but here that is no saving: the local
   layers hold nearly the whole story (~3.6x the KV of all-layer eviction at 32 slots).

| recall-token top-1 %, per-head, 4 sinks + recent B/2 | B=16 | 32 | 64 | 128 |
|---|---|---|---|---|
| full cache | 67.6 | 67.6 | 67.6 | 68.1 |
| sliding window + sinks | 18.0 | 25.7 | 48.7 | 61.9 |
| H2O (float) | 26.9 | 45.8 | 59.8 | 66.9 |
| TIU, 8-bit, weight = 255·attn | 24.5 | 43.6 | 59.3 | 66.8 |
| TIU, 8-bit, weight = 32·attn | 26.9 | 45.8 | 60.4 | 66.9 |
| recent-window attention, W=16 | 28.4 | 47.8 | 59.5 | 66.0 |
| EMA α=0.9 | 29.9 | 49.7 | 61.3 | 66.1 |
| *EMA α=0.9, no recent window* | *20.6* | *45.4* | *59.8* | *65.5* |
| *H2O, no recent window* | *5.2* | *8.1* | *13.3* | *29.5* |

The protection convention does not waste the budget: no policy did better without the recent
window than the best policy with it. 95% CIs on recall are ±1.5–2 points; paired differences
between policies are tighter (all in `results/summary.md`).

![Per-head eviction with sinks + recent window](figures/main_protected.png)
![Per-head eviction, sinks only](figures/main_no_recent.png)

## Setup

- **Model:** `roneneldan/TinyStories-28M` (GPT-Neo: 8 layers alternating global/local with a
  256 window, 16 heads × 32, d = 512; 52.0M parameters incl. the 25.7M tied embedding/lm_head),
  fp32; int8 = per-channel weight-only fake quantization.
- **Engine and gate:** a manual GPT-Neo decode loop with a slot-addressed cache (a new token
  takes the freed slot, as in the TIU wrapper). Full-cache logits match HF to 4e-5; under
  streaming eviction every policy matches an HF forward with per-head masks to 7e-5 even though
  slots are scrambled. GPT-Neo's absolute positions live in the cached K/V, so eviction needs no
  position fix-up: verified, not assumed (`tests/test_gate.py`). The full sweep was run on CPU and
  again on GPUs (RTX 2080 Ti / A6000, `run_gpu.sh`); all 140 configs agree to ≤ 0.15 recall
  points and ≤ 0.04% perplexity (`results/device_check.md`).
- **Data:** 500 TinyStories-valid stories (seed 0, uniform among those of 160–512 tokens; median
  201), teacher-forced, token by token. 3,897 recall tokens (620 names).
- **Protocol:** after each token, any (layer, head) holding more than B tokens evicts its
  lowest-scoring unprotected token, in all layers (local layers first drop tokens outside their
  window). Protection: 4 sinks + newest round(B/2), or 4 sinks only. Policies: sliding window;
  H2O; TIU (8-bit saturating, bit-exact vs a port of `token_importance_unit.sv`); recent-window
  attention (last W = 8/16/32 queries); EMA (α = 0.9/0.99). Metrics only on positions after the
  first eviction; paired bootstrap CIs over stories.

## Hardware cost (arithmetic from the config; `hw_cost.py`)

Assumptions: int8 weights and KV, HyperRAM 90 MB/s, 32 KB DTCM, 16 MHz, weights streamed each
token, 8-bit attention probabilities into the scorer, serial argmin at 1 compare/cycle.

- **KV:** 1 KB per token per layer, 8 KB for all layers (fp16: 16 KB). **DTCM holds 4 tokens of
  KV for the whole model**, so the cache lives in HyperRAM.
- **Traffic per token:** weights 50.9 MB → 566 ms, **half of it the tied lm_head**. KV reads at
  B = 16 / 64 / 128 add 1.5 / 5.9 / 11.7 ms (0.3–2% of the total); full cache at a 2,048 context
  adds 105 ms. **At these budgets eviction buys capacity, not bandwidth.**

| policy | state/slot | extra buffer | work per (layer, head) per token, N = 128 | scorer state, whole model: per-head (shared) |
|---|---|---|---|---|
| sliding window + sinks | – | ring pointer | – | 0 |
| H2O / TIU | 8 b | – | 129 add + argmin (129 cycles) | 16 KB (1 KB) |
| EMA, α = 1 − 2⁻ᵏ | 16 b | – | 129 shift-sub + 129 add + argmin | 32 KB (2 KB) |
| recent-window, W = 16 | 8 b | last W attention rows | 258 add/sub + argmin | 274 KB (17 KB) |

The per-head argmin is 16.5k cycles (1 ms) per token, small next to weight traffic. Per-head
H2O state at 128 slots is half the DTCM; recent-window attention needs W rows per head and does
not fit; EMA gets most of its benefit with no buffer.

## Notes for the TIU team

- **Your H2O simulation is optimistic.** `analysis/h2o_analysis.py` re-selects each query's
  top-k from full-cache cumulative attention, so evicted tokens can return and scores never see
  earlier evictions. Run on this model and these stories at equal budgets (`sim.py`), it reports
  a smaller loss than true streaming: recall −4.9 vs −7.9 points at 64 slots, −14.8 vs −21.8 at
  32 (paired differences +2.9 [2.0, 3.8] and +7.0 [5.8, 8.2]); perplexity 1.08x vs 1.10x and
  1.34x vs 1.43x full. Its one-row lookahead (row i's own attention in its score) explains only
  0.7–1.6 of those points; the rest is non-permanent re-selection. Your +1% perplexity at 25% on
  WikiText also used a 4x larger absolute cache (256 of 1,024 tokens); this simulation reaches
  +1.2% only at 128 slots. The 8-bit study quantizes the final sum over [0, T], which a streaming
  8-bit adder cannot do.
- **Weight scale is a design parameter** the RTL leaves open (finding 4). Longer contexts need a
  wider score or decay.
- **Argmin edge case:** the scan seeds with slot 0 and uses strict `<`; if protection were added
  by excluding slots and every candidate sat at 255, EVICT would return slot 0 whatever it holds.
- **Ordering:** your wrapper evicts before attending (N slots incl. the new token); mine attends
  over B + 1 then evicts.

## How this compares with prior work

- *StreamingLLM* (window + sinks) says the context window "remains unchanged": consistent with its
  low recall here. *RoCo* (AlpacaEval BLEU 63.0 H2O vs 47.3 StreamingLLM at 250 tokens) also
  puts H2O well ahead; *TOVA* found them tied on PG-19 perplexity (10.22 vs 10.28 at 64 tokens),
  where here H2O leads (1.10x vs 1.24x). TinyStories' heavy name reuse may explain that (untested).
- *RoCo* attributes H2O's weakness to recent tokens being under-scored by accumulation; *TOVA*
  (newest query only, no fixed window or sinks) beats H2O. Finding 2 is the extreme form, and the
  recency-aware scores fix it, with smaller W better without protection, TOVA's direction. H2O's
  own ablation also degrades with heavy hitters alone (Winogrande 51.9 vs 68.6 full, OPT-13B).
- Per-head beats shared here and in your Qwen2 study; *TOVA* found head-averaging better for its
  last-token score (W=1, untested here).
- *SubCache* (one-shot, Llama-3.1-8B, RULER-style) had H2O *below* the window at ≤ 25% and a
  question-aware score near full. In streaming H2O beats the window everywhere and the recency
  score gains only 2–4 points: plain continuation has no question to look from.
- *SpAtten* (HPCA 2021) pruned tokens permanently by cumulative attention with an on-chip top-k;
  the TIU follows that line with a serial argmin.

## Limitations

- Short contexts (160–512 tokens): 128 slots is most of a story; the transferable part is the
  ranking and the knees, not absolute budgets for Reef's real context.
- Teacher-forced next-token metrics, not free generation; one model, one dataset.
- The recall probe is a heuristic (word-initial tokens, string match, stop-word list) and
  includes generic content words.
- The TIU weight scale and protection mechanism are my assumptions; α = 0.875 was not tested;
  the int8 run covers weights only (fp32 activations and KV), not Reef's actual numerics.
