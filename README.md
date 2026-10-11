# SubCache

Experiments on KV-cache eviction for LLM inference: which cached tokens can be dropped, what
that costs in accuracy, and what it buys in memory and throughput. The starting question was
whether submodular selection (the PRISM family of objectives) beats the standard
attention-score heuristic, H2O.

**Status: work in progress.** The engine and harness run and are tested. The results below are
preliminary: small models, one dataset per result, and a larger batch on coding-agent sessions
is still running.

## What is here

| Path | What it does |
|---|---|
| `subcache/engine.py` | Forward pass over Hugging Face weights with its own preallocated KV cache. Supports masked eviction (for accuracy), physically compacted eviction (for speed and memory), and batched decoding. |
| `subcache/policies.py` | Eviction policies: recency, random, H2O, three PRISM objectives (GCMI, COM, FLQMI), a redundancy-aware variant of H2O (`wfl`), and an R-KV-style baseline. |
| `subcache/evals.py` | Streaming perplexity under eviction, with per-sequence and per-token losses; a needle-retrieval test. |
| `scripts/gate.py` | Correctness gate: the engine must reproduce `generate(do_sample=False)` token for token, single and batched, and fails on non-finite logits. |
| `scripts/run_eviction.py` | Accuracy sweep over policies and budgets. |
| `scripts/bench_speed.py` | Throughput, step time and memory against budget and batch size. |
| `scripts/prepare_agent_data.py` | Builds evaluation sequences from public coding-agent trajectories. |
| `scripts/analyze.py` | Paired-bootstrap comparison of each run against H2O. |
| `tests/test_policies.py` | Unit tests for selection, compaction and eviction invariants. |
| `results/` | Raw result rows (JSON lines) from the runs quoted below. |

## Preliminary findings

Hardware: RTX 2080 Ti (11 GB), fp16. Perplexity is measured on the positions where the cache
is over budget; "ratio" is relative to H2O at the same budget.

**1. Query-coverage objectives do not beat H2O.** TinyLlama-1.1B, WikiText-2, 32 sequences of
1,024 tokens, half of each budget reserved for the newest tokens:

| Policy | 50% budget | 25% | 12.5% |
|---|---|---|---|
| no eviction | 7.52 | 7.76 | 7.97 |
| H2O | 7.65 | 8.14 | 8.86 |
| H2O + redundancy term (`wfl`, mix 0.25) | 7.64 | 8.13 | 8.84 |
| FLQMI | 7.70 | 8.29 | 9.03 |
| GCMI / COM (32 recent queries) | 7.73 | 8.37 | 9.19 |
| random | 7.85 | 8.73 | 9.80 |
| recency | 7.93 | 8.85 | 9.97 |

- COM's concave term makes no measurable difference: it tracks the modular GCMI score.
- The policies that retain the most recent-query attention mass are not the most accurate.
- Giving the PRISM objectives a query set sampled from the whole history closes most of their
  gap to H2O (about 1% worse instead of up to 4%), but not all of it.
- A redundancy term on top of H2O helps slightly: 0.1% to 0.3% lower perplexity, with paired
  95% intervals that exclude "no difference" at the 25% and 12.5% budgets. Its selection is far
  more expensive.

**2. Eviction pays through memory, not step time.** deepseek-coder-1.3b, 8,192-token contexts,
256 decoded tokens, one run per configuration:

| Cache | Largest batch that fits | Throughput |
|---|---|---|
| full | 4 | 192 tok/s |
| H2O, 25% budget | 8 | 361 tok/s |
| H2O, 10% budget | 16 | 721 tok/s |

Decode step time stays near 20 ms from batch 1 to 16, because on this machine it is bound by
the CPU issuing kernel launches, not by the GPU. Batch-1 decode took the same 21 ms per token
on a 2080 Ti and on an RTX A6000 in the same host.

**3. Coding-agent sessions are far more repetitive than prose.** Share of 16-token runs that
already occurred earlier in the same sequence: 0.2% for WikiText, 8% for the first 8k tokens of
an agent session, 26% for an 8k window starting 24k tokens in. Whether redundancy-aware
eviction gains more there is what the running batch tests.

## Caveats

- Small models (1.1B and 1.3B parameters) and short contexts by current standards.
- Agent-session accuracy is perplexity of a small model on trajectories written by a much
  larger one; it is not task success.
- The R-KV-style baseline uses three settings I had to choose (the paper text I had did not
  give them) and this harness's rule for protected tokens. It is not a faithful reproduction.
- Timings come from a shared machine and vary by roughly 15% between runs.
- Qwen2.5-Coder-1.5B was dropped because it produces NaN logits in fp16.

## Running it

```bash
pip install torch transformers datasets
python tests/test_policies.py
python scripts/gate.py                                   # TinyLlama, fp16, needs a GPU
python scripts/run_eviction.py --n-seqs 8                # accuracy sweep on WikiText-2
python scripts/analyze.py results/eviction.jsonl
python scripts/bench_speed.py --context 8192             # speed and memory
```

`subcache_colab.ipynb` bundles an earlier version of the same steps for a Colab T4.

## Background and credit

- H2O: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models" (2023).
- PRISM: Kothawade et al., "PRISM: A Rich Class of Parameterized Submodular Information
  Measures for Guided Subset Selection" (AAAI 2022).
- R-KV: Cai et al., "R-KV: Redundancy-aware KV Cache Compression for Reasoning Models"
  (NeurIPS 2025).
- Agent trajectories: `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0).

Most of the code in this repository was written with Claude Code, which also proposed many of
the individual experiments. I set the research question and direction (submodular eviction,
measuring speed alongside accuracy, coding-agent sessions as the test bed), ran and reviewed the
experiments, and am re-implementing the engine by hand as a follow-up.
