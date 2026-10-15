"""Summarize scripts/run_bench.py output: python scripts/analyze_bench.py results/bench/<file>.jsonl ...

For every task (and context length, for RULER) prints a policy x budget table of mean scores,
then each policy's paired difference from H2O at the same budget with a 95% bootstrap interval
over examples. MATH-500 scores are accuracy in percent; LongBench and RULER use their own
metrics (x100). A final block averages the per-task means, as LongBench's tables do.
"""

import json
import sys
from collections import defaultdict

import numpy as np

ORDER = ["full", "recency", "h2o", "rkv", "wfl"]
NAME = {"full": "Full cache", "recency": "Window", "h2o": "H2O", "rkv": "R-KV-style", "wfl": "WFL"}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    return rows


def score_of(r):
    return 100.0 * r["correct"] if "correct" in r else r["score"]


def paired_ci(diff, n_boot=5000, seed=0):
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def budget_label(b):
    return f"{b:g}" if b > 1 else f"{100 * b:g}%"


def table(title, examples):
    """examples: list of {(policy, budget): score} dicts, one per example."""
    keys = sorted({k for ex in examples for k in ex})
    budgets = sorted({b for p, b in keys if p != "full"}, reverse=True)
    policies = [p for p in ORDER if p != "full" and any(k[0] == p for k in keys)]
    full = [ex[("full", 1.0)] for ex in examples if ("full", 1.0) in ex]
    print(f"\n{title}  ({len(examples)} examples; full cache {np.mean(full):.1f})")
    print("  " + f"{'policy':12}" + "".join(f"{budget_label(b):>30}" for b in budgets))
    for p in policies:
        line = f"  {NAME.get(p, p):12}"
        for b in budgets:
            have = [ex for ex in examples if (p, b) in ex and ("h2o", b) in ex]
            if not have:
                line += f"{'':>30}"
                continue
            mean = np.mean([ex[(p, b)] for ex in have])
            if p == "h2o":
                line += f"{mean:>9.1f}{'':21}"
            else:
                d, lo, hi = paired_ci([ex[(p, b)] - ex[("h2o", b)] for ex in have])
                mark = "+" if lo > 0 else "-" if hi < 0 else " "
                line += f"{mean:>9.1f} {d:+6.1f} [{lo:+.1f},{hi:+.1f}]{mark}".rjust(30)
        print(line)


def main(paths):
    rows = load(paths)
    groups = defaultdict(list)
    for r in rows:
        ex = {(x["policy"], x["budget"]): score_of(x) for x in r["results"]}
        groups[(r["task"], r.get("length"))].append(ex)
    for (task, length), examples in sorted(groups.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
        table(task + (f" @ {length} tokens" if length else ""), examples)
        if task == "math500":
            by = defaultdict(list)
            for r in rows:
                for x in r["results"]:
                    by[(x["policy"], x["budget"])].append((x["gen_len"], x["truncated"]))
            print("  mean generated tokens / share hitting the cap:")
            for (p, b), v in sorted(by.items(), key=lambda kv: (ORDER.index(kv[0][0]), -kv[0][1])):
                print(f"    {NAME.get(p, p):12} {budget_label(b):>6}  {np.mean([g for g, _ in v]):7.0f}  "
                      f"{100 * np.mean([t for _, t in v]):5.1f}%")
    # Average over tasks, as LongBench reports it: the mean of the per-task means.
    by_len = defaultdict(dict)
    for (task, length), examples in groups.items():
        if task == "math500":
            continue
        keys = {k for ex in examples for k in ex}
        by_len[length][task] = {k: np.mean([ex[k] for ex in examples if k in ex]) for k in keys}
    for length, tasks in by_len.items():
        if len(tasks) < 2:
            continue
        keys = set.intersection(*[set(v) for v in tasks.values()])
        print(f"\nAverage over {len(tasks)} tasks" + (f" @ {length} tokens" if length else "") + ":")
        budgets = sorted({b for p, b in keys if p != "full"}, reverse=True)
        print("  " + f"{'policy':12}" + "".join(f"{budget_label(b):>9}" for b in budgets))
        print(f"  {'Full cache':12}" + f"{np.mean([v[('full', 1.0)] for v in tasks.values()]):>9.1f}")
        for p in ORDER[1:]:
            if any((p, b) in keys for b in budgets):
                print(f"  {NAME[p]:12}" + "".join(
                    f"{np.mean([v[(p, b)] for v in tasks.values()]):>9.1f}" if (p, b) in keys else f"{'':>9}"
                    for b in budgets))


if __name__ == "__main__":
    main(sys.argv[1:])
