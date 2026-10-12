# SubCache

My goal as a researcher and coder is always to push for efficiency and speed wherever possible.
My research is in subset selection for ML, where I use submodular functions (functions with
diminishing returns) to pick the most useful subset out of a bigger set. Recently I've gotten
interested in inference optimization and how the hardware behind a model affects how fast it
runs, and I saw a chance to combine the two.

So this project asks one question: when an LLM's KV cache gets too big and you have to throw
tokens away, can a submodular function pick what to keep better than the usual heuristic?

The usual heuristic is H2O, which keeps the tokens that have received the most attention. My
version adds a weighted facility location term (`wfl` in the code) on top of it, so that if two
cached tokens are basically duplicates, only one of them needs to stay.

While the results weren't great, it was still a fun experiment. H2O turned out to be very hard
to beat. My method did win, but only on repetitive text and only by a little, and it was much
slower. I learned a lot more about inference from finding that out than I expected to.

## What I ran it on

Everything runs on one RTX 2080 Ti (11 GB) in fp16. The main model is deepseek-coder-1.3b, and
I used TinyLlama for the early experiments. For data I used real coding-agent sessions from
`nebius/SWE-rebench-openhands-trajectories`, because agents re-read the same files and logs
over and over, plus WikiText as a control with almost no repetition.

The engine has its own KV cache instead of using Hugging Face's, so tokens can be masked out or
physically removed. Before trusting any number I checked that it reproduces Hugging Face's
greedy `generate` token for token on both models.

Text goes in 64 tokens at a time, and after each chunk the cache is cut back down to its
budget. Every policy keeps the first 4 tokens and the newest half of the budget, so they only
differ in which older tokens they keep. I measure perplexity on the part of the sequence where
the cache is actually over budget, and compare policies on the same sequences.

## Eviction is worth it, mostly because of memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/throughput-dark.svg">
  <img src="figures/throughput-light.svg" alt="Bar chart: decode throughput rises from 190 tokens per second with the full cache to about 1,400 at a 6.25% budget">
</picture>

With 8,192-token contexts, the full cache only fits 4 sequences on the card. Keeping a quarter
of it fits 12, which is 2.8x the throughput for about 5% worse perplexity.

Something I didn't expect: a smaller cache doesn't make each step faster. A decode step takes
about 21 ms no matter the batch size or cache size. It turns out that on this machine decoding
is limited by the CPU launching GPU kernels, not by the GPU. I checked by running the same
thing on an RTX A6000 in the same box and got the same 21 ms. So all of the speedup comes from
fitting more sequences in memory.

## Does the submodular part help?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/redundancy-dark.svg">
  <img src="figures/redundancy-light.svg" alt="Line chart: the redundancy objective is 0.2 to 1.4 percent better than H2O on agent tokens, more so at smaller budgets">
</picture>

A little. On agent sessions it beat H2O at every budget I tried, and the gap grew as the budget
shrank, but we're talking 0.2% to 1.4% lower perplexity on the agent's own tokens. That closes
only a few percent of the distance between H2O and not evicting at all.

One setting looked better in a sweep (using only the redundancy term, about 2-3% better), but I
found it after the fact on one dataset, so I'm rerunning it before I believe it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/repetition-dark.svg">
  <img src="figures/repetition-light.svg" alt="Bar chart: the redundancy objective improves on H2O only on late agent sessions, not on session starts or WikiText">
</picture>

The part I find most convincing is where it helps. I measured how repetitive each dataset is
(how often a run of 16 tokens has already appeared earlier in the same sequence):

| Data | Repetition | Did my method help? |
|---|---|---|
| WikiText | 0.2% | No |
| Start of an agent session | 8% | No |
| 24k tokens into an agent session | 26% | Yes, at every budget |

It only helps where there is redundancy to remove, which is what the math says should happen.

## Why I wouldn't actually use it

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/selection-dark.svg">
  <img src="figures/selection-light.svg" alt="Bar chart: selection takes under 1 percent of decode time for H2O, 19 percent for the R-KV-style baseline and 71 percent for the redundancy objective">
</picture>

Picking what to evict isn't free. H2O's choice is basically a sort. Mine has to compare cached
tokens against each other in every attention head, and at a 12.5% budget that took 71% of the
decode time. Throughput dropped to 78 tok/s, which is worse than not evicting at all (190). A
1% accuracy gain doesn't pay for that.

## Other things I ran into

- My first idea was different: use submodular functions from the PRISM paper to cover what
  recent queries attend to. That lost to H2O by up to 4%. The policy that kept the most
  attention mass was not the most accurate one, which surprised me.
- At really small budgets (3%), just using attention from the last 32 queries beat H2O on
  three of my four datasets. I think H2O holds on to old tokens for too long, but I haven't
  dug into it.
- Qwen2.5-Coder-1.5B gives NaN for every output in fp16. My correctness check passed anyway at
  first, because Hugging Face and my engine were both producing the same garbage token. It now
  fails if any logit isn't finite.
- I tried an R-KV-style baseline, which is the closest existing method (it also mixes attention
  with key similarity). It was about even with H2O at big budgets and much worse at small ones.
  I had to guess three of its settings, so this isn't a fair test of the real paper.

## What this doesn't show

These are small models, and I measured perplexity, not whether an agent actually finishes its
task. The timings come from a shared lab machine and vary by around 13% between runs. And the
"decoding is CPU-bound" result is about my hardware. On a faster host, cache size would matter
for step time too and the speed story would be different.

## Running it

```bash
pip install torch transformers datasets
python tests/test_policies.py
python scripts/gate.py                              # needs a GPU
python scripts/run_eviction.py --n-seqs 8           # small sweep on WikiText
python scripts/analyze.py results/batch/core8k_late.jsonl --masked
python scripts/make_figures.py
```

The engine is in `subcache/engine.py`, the eviction policies in `subcache/policies.py`, and the
evaluations in `subcache/evals.py`. `scripts/` has the correctness gate, the sweeps, the speed
benchmark and the figure code. `results/` has the raw numbers behind everything above.

## Credit

- H2O: Zhang et al., 2023
- PRISM: Kothawade et al., AAAI 2022
- R-KV: Cai et al., NeurIPS 2025
- Agent trajectories: `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0)

Most of the code here was written with Claude Code, and it suggested a lot of the individual
experiments too. The question, the direction (submodular eviction, measuring speed and not just
accuracy, testing on agent sessions) and running and checking the experiments were me. I'm now
writing my own inference engine from scratch in a separate repo, so I understand every piece of
this properly.
