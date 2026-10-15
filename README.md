# SubCache

When an LLM's KV cache gets too big and you have to throw tokens away, can a submodular function
pick what to keep better than H2O? I built a small inference engine with its own KV cache to find
out. The short answer: only barely, only on very repetitive text, and it costs too much time to be
worth it.

Full write-up with all the figures: https://sky1219.github.io/SubCache/

## The short version

- **Evicting is worth it on an 11 GB card, but for memory, not speed.** Keeping a quarter of the
  cache fit 3x as many sequences and gave 2.8x the throughput, for about 5% worse perplexity.
  Each decode step didn't get any faster.
- **My policy (WFL) beat H2O in one place.** On late coding-agent sessions it was 1.1% to 2.5%
  better on perplexity. Everywhere else it tied or lost.
- **It's too slow.** Choosing what to keep took 71% of decode time at a 12.5% budget, so it ended
  up slower than not evicting at all.
- **The baseline that looks at the question won the task benchmarks.** An R-KV-style policy beat
  both H2O and WFL on retrieval and on LongBench. MATH was too noisy to rank anything.

## Why I tried this

My research is in subset selection for ML: using submodular functions (functions with
diminishing returns) to pick the most useful subset out of a bigger set. Lately I've gotten into
inference, and into how the hardware decides how fast a model runs. Cache eviction is a subset
selection problem sitting inside an inference engine, so it seemed like a good place to combine
the two.

The usual heuristic is H2O, which keeps the tokens that have received the most attention. My
version adds a weighted facility location term on top of it (`wfl` in the code), so if two cached
tokens are basically duplicates, only one of them has to stay.

It didn't work out the way I hoped. H2O is very hard to beat. But I learned a lot, mostly about
where the time actually goes.

## Setup

Everything runs on one RTX 2080 Ti (11 GB) in fp16. The main model is deepseek-coder-1.3b, and I
used TinyLlama for the early experiments. For data I used real coding-agent sessions from
`nebius/SWE-rebench-openhands-trajectories`, because agents re-read the same files and logs over
and over. WikiText is the control, since it has almost no repetition.

The engine keeps its own KV cache instead of using Hugging Face's, so tokens can be masked out or
actually removed.

Text goes in 64 tokens at a time, and after each chunk the cache is cut back down to its budget.
Every policy keeps the first 4 tokens and the newest half of the budget, so they only differ in
which older tokens they keep. I measure perplexity on the part of the sequence where the cache is
over budget, and compare policies on the same sequences.

## Eviction is worth it, mostly because of memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/throughput-dark.svg">
  <img src="figures/throughput-light.svg" alt="Bar chart: decode throughput rises from 190 tokens per second with the full cache to about 1,400 at a 6.25% budget">
</picture>

With 8,192-token contexts, the full cache only fits 4 sequences on the card. Keeping a quarter of
it fits 12, which is 2.8x the throughput for about 5% worse perplexity.

What surprised me is that a smaller cache doesn't make each step faster. A decode step takes about
21 ms no matter the batch size or cache size. On this machine, decoding is limited by the CPU
launching GPU kernels, not by the GPU itself. I checked by running the same thing on an RTX A6000
in the same box and got the same 21 ms. So all of the speedup comes from fitting more sequences in
memory.

## Does the submodular part help?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/redundancy-dark.svg">
  <img src="figures/redundancy-light.svg" alt="Line chart: on agent tokens the redundancy objective is 0.2 to 0.5 percent better than H2O with weight 0.25 and 1.1 to 2.5 percent better with weight 1.0, more so at smaller budgets">
</picture>

A little. On late agent sessions it beat H2O at every budget I tried, and the gap grew as the
budget shrank. With the redundancy weight I picked in advance (0.25), that's 0.2% to 0.5% lower
perplexity on the agent's own tokens.

A smaller sweep said to turn the weight all the way up to 1.0, so I reran that on the full
8,192-token setup. It held up: 1.1% to 2.5% better than H2O. That closes about a fifth of the gap
to the full cache at a 25% budget and a tenth at 12.5%. At those budgets it ties the R-KV-style
baseline, and below them it beats it easily. But that was the only dataset where it helped.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/repetition-dark.svg">
  <img src="figures/repetition-light.svg" alt="Bar chart: the redundancy objective improves on H2O only on late agent sessions, not on session starts or WikiText">
</picture>

I measured how repetitive each dataset is (how often a run of 16 tokens has already appeared
earlier in the same sequence):

| Data | Repetition | Weight 0.25 | Weight 1.0 |
|---|---|---|---|
| WikiText | 0.2% | No | Mixed |
| Start of an agent session | 8% | No | Worse |
| First 16k tokens of an agent session | 16% | No | Worse |
| 24k tokens into an agent session | 26% | Yes, at every budget | Yes, at every budget |

It only helps on the most repetitive data, which is what you'd expect. But the 16k sessions are
pretty repetitive too and it didn't help there, so repetition alone doesn't tell you when it will
work. A higher weight makes the trade sharper: 1.0 is the best setting on late sessions and up to
4% worse than H2O everywhere else.

## Why I wouldn't actually use it

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/selection-dark.svg">
  <img src="figures/selection-light.svg" alt="Bar chart: selection takes under 1 percent of decode time for H2O, 19 percent for the R-KV-style baseline and 71 percent for the redundancy objective">
</picture>

Picking what to evict isn't free. H2O's choice is basically a sort. Mine has to compare cached
tokens against each other in every attention head, and at a 12.5% budget that took 71% of the
decode time. Throughput dropped to 78 tok/s, which is worse than not evicting at all (190). A
1–2% perplexity gain doesn't pay for that.

## Task benchmarks

Perplexity only tells you so much, so I also checked whether the policies change actual answers.
These ran on the RTX A6000 in bf16 with bigger models, and WFL is the weight-1.0 version. Every
task uses a small subset (listed below), so treat these as a first look, not a leaderboard entry.

For LongBench and the RULER-style tasks, the model reads the whole prompt, the cache is cut down
to the budget once, and then it answers with nothing else evicted. Budgets are a share of the
prompt, and scores are each task's own metric out of 100.

One caveat: like in SnapKV-style evaluations, the first answer token comes from the full cache, so
every method gets it for free. On the RULER-style tasks that token is only punctuation. On
LongBench's short-answer tasks it can be the whole answer (a fifth to over a third of the answers
there are one word), which makes the methods look closer than they are.

**RULER-style retrieval** (Llama-3.1-8B-Instruct, 8,192-token prompts, 5 tasks × 10 examples,
full cache 92.8):

| | 50% | 25% | 12.5% | 6.25% |
|---|---|---|---|---|
| Window | 51.6 | 31.7 | 18.7 | 8.7 |
| H2O | 59.7 | 26.4 | 11.7 | 4.9 |
| R-KV-style | **90.8** | **93.6** | **91.6** | **91.1** |
| WFL | 59.3 | 38.7 | 14.9 | 4.1 |

**LongBench** (Llama-3.1-8B-Instruct, prompts cut to 16k tokens; qasper, multifieldqa_en,
hotpotqa and lcc with 20 examples each, gov_report with 10; average over the 5 tasks, full cache
51.8):

| | 50% | 25% | 12.5% | 6.25% |
|---|---|---|---|---|
| Window | 43.2 | 40.1 | 37.0 | 33.4 |
| H2O | 51.0 | 46.9 | 44.8 | 41.0 |
| R-KV-style | **51.8** | **50.2** | **49.1** | **44.4** |
| WFL | 51.2 | 48.0 | 46.4 | 41.4 |

**MATH-500** (DeepSeek-R1-Distill-Qwen-1.5B, 20 problems from levels 1 and 2, one sample each,
capped at 4,096 generated tokens). Here the cache is cut back every 64 generated tokens, and the
budget is a number of cache slots. Accuracy in percent, full cache 55:

| | 2048 | 1024 | 512 | 256 |
|---|---|---|---|---|
| Window | 60 | 60 | 25 | 15 |
| H2O | 55 | 60 | 65 | 45 |
| R-KV-style | **70** | 45 | 45 | 40 |
| WFL | 60 | **65** | **70** | **50** |

What I take from these:

- **For retrieval, what matters is whose attention you count.** The R-KV-style baseline scores
  tokens by how much the last 32 tokens of the prompt (the question) attend to them, so it keeps
  the needle. H2O and WFL use attention summed over the whole prompt, and they lose the needle at
  every budget. WFL is built on H2O's scores, so being redundancy-aware can't fix that.
- **On LongBench, WFL lands between H2O and R-KV-style.** On every single task it was
  statistically tied with H2O. R-KV-style was clearly better on two of the QA tasks (qasper and
  multifieldqa_en) at some budgets, but clearly worse than H2O on summarization (gov_report) at
  25% and below, while WFL stayed level with H2O there. That makes sense: a summary needs
  coverage of the whole document.
- **MATH is too noisy to rank.** Half of the full-cache runs hit the 4,096-token cap before
  answering, so the scores mostly say whether the model finished thinking in time. WFL was 5
  points above H2O at every budget, but every interval was about ±20 points. The only clear
  results are that Window falls apart at 512 and 256 slots, and that R-KV-style was worse than H2O
  at 512.

The obvious next step is WFL with question-window weights instead of H2O's. That might keep the
retrieval wins and the summary coverage at the same time. I haven't run it yet.

## Limitations

- **I couldn't measure the real speed effect.** Decoding on my machine was CPU bound, so a step
  took the same time at every cache size.
- **The perplexity runs use small models,** and lower perplexity isn't the same as an agent
  actually finishing its task.
- **The task benchmarks are small.** They use 10 to 20 examples per task, LongBench prompts cut to
  16k tokens, only the easiest MATH levels, a 4,096-token generation cap and one sample per
  problem. Many intervals are ±10 points or wider.
- **Some pieces are my own versions.** The RULER-style tasks are my reimplementation with a
  WikiText haystack. To keep selection fast on long prompts, WFL only chose among the older tokens
  H2O ranks highest (twice as many as it had slots to fill).

So none of these numbers should be compared with published tables. I also didn't report timings
for the task benchmarks, since the GPU was shared with other jobs.

## Running it

```bash
pip install torch transformers datasets
pip install rouge fuzzywuzzy math-verify      # only for the task benchmarks
```

The tests run on the CPU, so you can check that everything works without a GPU:

```bash
python tests/test_policies.py
python tests/test_longctx.py                  # benchmark code on a tiny random model
```

Everything else needs a GPU:

```bash
python scripts/gate.py                        # check the engine matches HF generate
python scripts/run_eviction.py --policies full h2o wfl rkv --mix 1.0 --n-seqs 8   # quick WikiText sweep
python scripts/analyze.py results/batch/core8k_late.jsonl --masked                # main agent result
python scripts/make_figures.py

# task benchmarks (needs an A6000-sized GPU)
bash scripts/bench_all.sh long                # RULER-style tasks, then LongBench
bash scripts/bench_all.sh math                # MATH-500
python scripts/analyze_bench.py results/bench/longbench_llama31.jsonl
```

The agent-session runs need their data built first with `scripts/prepare_agent_data.py`.
`jobs.txt` has the exact command behind every result in `results/`.

## Where things are

- `subcache/engine.py`: the inference engine and its KV cache
- `subcache/policies.py`: the eviction policies (H2O, WFL, R-KV-style and others)
- `subcache/evals.py`: the perplexity evaluations
- `subcache/benchmarks.py` and `subcache/longctx.py`: the task benchmarks, plus the
  prompt-compression and decode-time eviction loops
- `scripts/`: the correctness gate, sweeps, speed benchmark, task benchmarks and figure code
- `results/`: the raw numbers behind everything above

## Credit

- H2O: Zhang et al., 2023
- PRISM: Kothawade et al., AAAI 2022
- R-KV: Cai et al., NeurIPS 2025
- Agent trajectories: `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0)
- LongBench: Bai et al., ACL 2024
- RULER: Hsieh et al., COLM 2024
- MATH: Hendrycks et al., NeurIPS 2021 (the MATH-500 subset from Lightman et al., 2023)

Right now I'm building my own inference engine from scratch:
[hipinfer](https://github.com/Sky1219/hipinfer).
