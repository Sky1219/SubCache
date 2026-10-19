"""Cross-device check: the same sweep run on CPU (reef/results) and on GPU (reef/results/gpu).

    python -m reef.compare_devices      -> reef/results/device_check.md

Both runs decode the same 500 stories with the same configs; only the device (and the batch
size, which changes padding but not any story's computation) differ. fp32 rounding differs
between devices, so per-story numbers can differ slightly; what must hold is that every
aggregate metric agrees to well within its bootstrap CI.
"""

import os

import numpy as np

from reef.analyze import METRICS, RES, load, table

GPU = os.path.join(RES, "gpu")


def aggregate(d, name):
    num, den, tr, _ = METRICS[name]
    v = d[num].sum() / d[den].sum()
    return float(np.exp(v)) if tr == "exp" else float(v)


def main():
    import reef.analyze as A

    files = ["main.part*.jsonl", "int8.jsonl", "global.jsonl"]
    cpu, _ = table([r for f in files for r in load(f)])
    A.RES = GPU
    gpu, _ = table([r for f in files for r in load(f)])
    A.RES = RES
    missing = set(cpu) ^ set(gpu)
    rows, worst = [], {m: 0.0 for m in METRICS}
    for key in sorted(set(cpu) & set(gpu)):
        for m in METRICS:
            a, b = aggregate(cpu[key], m), aggregate(gpu[key], m)
            d = abs(a - b) / (a if m == "ppl" else 1.0)  # perplexity: relative difference
            worst[m] = max(worst[m], d)
        same = float(np.mean(cpu[key]["correct"] == gpu[key]["correct"]))
        rows.append((key, same))
    md = ["# CPU vs GPU reproduction\n",
          f"Configs compared: {len(rows)} (only in one run: {len(missing)}).\n",
          "| metric | largest |CPU - GPU| over all configs |", "|---|---|"]
    for m, d in worst.items():
        if m == "ppl":
            md.append(f"| {METRICS[m][3]} | {100 * d:.3f}% (relative) |")
        elif m in ("recall_nll", "kl"):
            md.append(f"| {METRICS[m][3]} | {d:.4f} nats |")
        else:
            md.append(f"| {METRICS[m][3]} | {100 * d:.3f} points |")
    frac = np.mean([s for _, s in rows])
    md.append(f"\nPer-story count of correct top-1 predictions identical on both devices: {100 * frac:.1f}% of "
              f"(config, story) pairs.")
    with open(os.path.join(RES, "device_check.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
