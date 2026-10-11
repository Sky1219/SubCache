"""Compare eviction runs against H2O with paired bootstrap confidence intervals.

    python scripts/analyze.py results/edmonds/main32_*.jsonl

Runs are paired by sequence (every run evaluates the same sequences in the same order), so the
difference in per-sequence loss is resampled, not the two losses separately. A run is compared
with the h2o run at the same budget and recent_frac. `ratio` is ppl / ppl_h2o: below 1 beats
H2O. The verdict is 'better' / 'worse' only if the 95% interval excludes 1.
"""

import json
import math
import os
import random
import sys


def label(row):
    s = row["settings"]
    parts = [row["policy"]]
    if row["policy"] == "wfl":
        parts.append(f"mix={s.get('mix')} pow={s.get('sim_power')}")
        if s.get("wfl_mode") == "prune":
            parts.append(f"prune{s.get('prune_step')}")
    if row["policy"] == "rkv":
        parts.append(f"lam={s.get('lam')} win={s['window']}")
    if row["policy"] in ("gcmi", "com", "flqmi"):
        parts.append(f"win={s['window']}{'r' if s.get('window_mode') == 'reservoir' else ''}")
        if s["eta"] != 1.0 or s["psi"] != "sqrt":
            parts.append(f"eta={s['eta']} {s['psi']}")
    if s.get("eps") is not None:
        parts.append(f"eps={s['eps']}")
    if s.get("value_aware"):
        parts.append("value-aware")
    return " ".join(parts)


def paired_ci(a, b, n_boot=5000, seed=0):
    """95% bootstrap interval for exp(mean(a - b)), resampling sequences."""
    diff = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    n = len(diff)
    rng = random.Random(seed)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return math.exp(sum(diff) / n), math.exp(means[int(0.025 * n_boot)]), math.exp(means[int(0.975 * n_boot)]), n


def main(paths):
    masked = "--masked" in paths  # compare on the masked tokens (e.g. the agent's own turns)
    paths = [p for p in paths if p != "--masked"]
    rows = [json.loads(line) for p in paths for line in open(p)]
    if masked:
        for r in rows:
            if "nll_post_masked_per_seq" in r:
                r["nll_post_per_seq"] = [x for x in r["nll_post_masked_per_seq"]]
                r["ppl_post"] = r.get("ppl_post_masked", float("nan"))
            else:
                r.pop("nll_post_per_seq", None)
    rows = [r for r in rows if "nll_post_per_seq" in r]
    if not rows:
        raise SystemExit("no rows with per-sequence losses (rerun with the current run_eviction.py)")
    groups = {}
    for r in rows:
        s = r["settings"]
        key = (s["model"], os.path.basename(s.get("data") or s["text"]), s["seq_len"], s["interval"],
               r["budget"], s["recent_frac"])
        groups.setdefault(key, []).append(r)
    for key, group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], -kv[0][4], -kv[0][5])):
        model, data, seq_len, interval, budget, recent_frac = key
        ref = next((r for r in group if r["policy"] == "h2o"), None)
        print(f"\n{model.split('/')[-1]} | {data} | {seq_len} tokens | evict every {interval} | budget {budget:g} | "
              f"recent share {recent_frac:g}" + ("" if ref else "   (no h2o run to compare with)"))
        print(f"  {'run':34} {'seqs':>4} {'ppl_post':>9} {'ratio vs h2o':>12} {'95% interval':>17}  {'sel s/evict':>11}  verdict")
        for r in sorted(group, key=lambda r: r["ppl_post"]):
            per_evict = r["select_s_total"] / r["evictions"] if r.get("evictions") else float("nan")
            line = f"  {label(r):34} {len(r['nll_post_per_seq']):4d} {r['ppl_post']:9.3f}"
            if ref is None or r is ref or r["policy"] == "full":
                print(line + f" {'':12} {'':17}  {per_evict:11.4f}")
                continue
            ratio, lo, hi, n = paired_ci(r["nll_post_per_seq"], ref["nll_post_per_seq"])
            verdict = "better" if hi < 1 else "worse" if lo > 1 else "no clear difference"
            print(line + f" {ratio:12.4f} {f'[{lo:.4f}, {hi:.4f}]':>17}  {per_evict:11.4f}  {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
