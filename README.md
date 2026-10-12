# SubCache

My goal as a researcher and coder is always to push for efficiency and speed wherever possible.
My research is in subset selection for ML, where I use submodular functions, which capture
diminishing returns, to choose the most useful subset of a larger set. Recently I've become
interested in inference optimization and in how the hardware that runs models shapes their
speed, and I saw an opportunity to combine the two areas. This project applies submodular
selection to KV-cache eviction, using an importance-weighted facility-location objective
(`wfl` in the code), and compares throughput, output quality, wall time and other relevant
metrics against the standard attention-score heuristic, H2O.

**Short version of the result:** evicting cache entries is well worth doing, and plain H2O
gets nearly all of the benefit. The submodular redundancy term beats H2O only on repetitive
context, by a fraction of a percent to a few percent, and its selection cost outweighs that.

## Setup

- **Models:** deepseek-coder-1.3b-instruct for the main results, TinyLlama-1.1B for early
  ones. fp16 on an RTX 2080 Ti (11 GB).
- **Engine:** a forward pass over Hugging Face weights with its own preallocated KV cache
  (`subcache/engine.py`). It reproduces `generate(do_sample=False)` token for token on both
  models, single and batched (`scripts/gate.py`).
- **Data:** public coding-agent sessions (`nebius/SWE-rebench-openhands-trajectories`) and
  WikiText-2 as a non-repetitive control.
- **Protocol:** text is fed 64 tokens at a time and the cache is evicted down to its budget
  after each chunk. Every policy keeps the first 4 tokens and the newest half of the budget,
  and chooses the rest. Perplexity is measured on the positions where the cache is over
  budget. Policies are compared on the same sequences with paired bootstrap intervals.

## 1. Eviction buys throughput through memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/throughput-dark.svg">
  <img src="figures/throughput-light.svg" alt="Bar chart: decode throughput rises from 190 tokens per second with the full cache to about 1,400 at a 6.25% budget">
</picture>

| Cache | Largest batch tested that fits | Throughput | Perplexity vs full cache |
|---|---|---|---|
| Full | 4 | 190 tok/s | reference |
| H2O, 50% budget | 6 | 263 tok/s | not measured |
| H2O, 25% budget | 12 | 530 tok/s | +5% |
| H2O, 12.5% budget | 24 | 1,046 tok/s | +12% |
| H2O, 6.25% budget | 32 | 1,404 tok/s | +25% |

A quarter-size cache gives 2.8 times the throughput for about 5% higher perplexity. The decode
step takes 21 to 23 ms at every batch size, so the whole gain comes from fitting more
sequences in memory. The reason is that decoding on this machine is bound by the CPU issuing
GPU kernel launches: batch-1 decode took the same 21 ms per token on the 2080 Ti and on an RTX
A6000 in the same host.

## 2. A redundancy term beats H2O on repetitive context, slightly

H2O scores each token on its own, so it keeps every copy of content that appears twice. The
`wfl` objective adds a facility-location term: each cached token, weighted by its H2O score,
should have a similar token among those kept. Similarity is the product of key cosine and
value cosine.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/redundancy-dark.svg">
  <img src="figures/redundancy-light.svg" alt="Line chart: the redundancy objective is 0.2 to 1.4 percent better than H2O on agent tokens, more so at smaller budgets">
</picture>

Perplexity relative to H2O on late agent sessions (8,192 tokens; 1.0 means equal):

| Budget | Weight 0.25, all tokens | Weight 0.25, agent's tokens | Weight 0.5, agent's tokens |
|---|---|---|---|
| 25% | 0.9986 | 0.9980 | 0.9959 |
| 12.5% | 0.9979 | 0.9966 | 0.9934 |
| 6.25% | 0.9980 | 0.9959 | 0.9903 |
| 3.1% | 0.9977 | 0.9957 | 0.9861 |

Every interval excludes 1.0, so the improvement is consistent, but it is small. With weight
0.25, fixed before the run, it recovers 1% to 3% of the gap between H2O and the full cache.
A sweep at 4,096 tokens found that the redundancy term alone (weight 1.0) does better, 1.7% to
2.7% on the agent's tokens; that was found after the fact on one dataset and is being rerun at
8,192 tokens.

## 3. The gain appears only where the context repeats

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/repetition-dark.svg">
  <img src="figures/repetition-light.svg" alt="Bar chart: the redundancy objective improves on H2O only on late agent sessions, not on session starts or WikiText">
</picture>

| Dataset | Share of 16-token runs seen earlier in the sequence | Effect of the redundancy term |
|---|---|---|
| WikiText-2 | 0.2% | none (slightly worse at a 3.1% budget) |
| Agent session, first 8,192 tokens | 8% | none |
| Agent session, 8,192 tokens starting 24,000 in | 26% | 0.1% to 0.2% better at every budget |

This is the evidence that redundancy is what the objective exploits.

## 4. Selection cost decides it

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/selection-dark.svg">
  <img src="figures/selection-light.svg" alt="Bar chart: selection takes under 1 percent of decode time for H2O, 19 percent for the R-KV-style baseline and 71 percent for the redundancy objective">
</picture>

At a 12.5% budget the redundancy objective spends 71% of decode time choosing what to evict
and reaches 78 tok/s, below the 190 tok/s of not evicting at all. H2O's selection is free by
comparison. As built, the accuracy gain does not come close to paying for the cost.

## Other findings

- **Query-coverage objectives lose to H2O.** The PRISM objectives COM and FLQMI, with attention
  as the kernel, were up to 4% worse than H2O on TinyLlama. COM's concave term made no
  measurable difference, and the policies that retained the most recent-query attention were
  not the most accurate.
- **At very small budgets, scoring by recent attention beats H2O.** At a 3.1% budget, summing
  attention from only the 32 newest queries was 2% to 4% better than H2O on session starts,
  WikiText and 16,000-token sessions, and worse on late sessions. H2O's score accumulates over
  the whole history, which seems to hurt when little can be kept. Not investigated further.
- **An R-KV-style baseline** (attention importance minus key redundancy) matched or slightly
  beat H2O at a 25% budget on the agent's tokens and was much worse at small budgets.
- **Greedy removal matches exact greedy** on TinyLlama at 4 to 30 times less selection time,
  without the approximation guarantee.
- **Qwen2.5-Coder-1.5B produces NaN logits in fp16**, in Hugging Face as well as here, so it
  was dropped. The gate now fails on non-finite logits; before that, two NaN paths "matched".

## Limits

- Small models (1.1B and 1.3B parameters), and perplexity rather than task success. The agent
  sessions were written by a much larger model.
- The R-KV-style baseline uses three settings I had to choose and this harness's rule for
  protected tokens. It is not a faithful reproduction of the paper.
- Timings come from a shared machine. Repeated runs of the same configuration differed by a
  median of 13% in step time. Batch sizes were tried on a grid, so "largest batch" is the
  largest tested.
- One machine. With a faster CPU, or CUDA graphs, step time would depend on cache size and the
  speed picture would change.

## Repository

| Path | Contents |
|---|---|
| `subcache/engine.py` | Forward pass, KV cache, masked and physically compacted eviction, batched decode |
| `subcache/policies.py` | Recency, random, H2O, GCMI, COM, FLQMI, `wfl` (greedy and pruning), R-KV-style |
| `subcache/evals.py` | Streaming perplexity with per-sequence and per-token losses; needle retrieval |
| `scripts/gate.py` | Correctness gate against `generate` |
| `scripts/run_eviction.py`, `scripts/bench_speed.py` | Accuracy sweeps; speed and memory benchmark |
| `scripts/prepare_agent_data.py` | Builds evaluation sequences from agent trajectories |
| `scripts/analyze.py`, `scripts/make_figures.py` | Paired-bootstrap comparison; the figures above |
| `scripts/make_batch.py`, `scripts/batch_worker.sh`, `jobs.txt` | The unattended batch that produced `results/batch/` |
| `results/` | Raw result rows (JSON lines) for every number quoted here |
| `tests/test_policies.py` | Unit tests for selection, compaction and eviction invariants |

```bash
pip install torch transformers datasets
python tests/test_policies.py
python scripts/gate.py                              # needs a GPU
python scripts/run_eviction.py --n-seqs 8           # small accuracy sweep on WikiText-2
python scripts/analyze.py results/batch/core8k_late.jsonl --masked
python scripts/make_figures.py
```

## Credit

- H2O: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models" (2023).
- PRISM: Kothawade et al., "PRISM: A Rich Class of Parameterized Submodular Information
  Measures for Guided Subset Selection" (AAAI 2022).
- R-KV: Cai et al., "R-KV: Redundancy-aware KV Cache Compression for Reasoning Models"
  (NeurIPS 2025). The closest prior work: it also combines attention importance with key
  redundancy, as a per-token score rather than a set function.
- Agent trajectories: `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0).

Most of the code in this repository was written with Claude Code, which also proposed many of
the individual experiments. I set the research question and direction (submodular eviction,
measuring speed alongside accuracy, coding-agent sessions as the test bed), ran and reviewed the
experiments, and am writing my own inference engine from scratch in a separate repository.
