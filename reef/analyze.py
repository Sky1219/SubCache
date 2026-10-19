"""Bootstrap analysis and figures for the streaming-eviction sweep.

    python -m reef.analyze      reads reef/results/*.jsonl, writes reef/results/summary.{json,md}
                                and reef/figures/*.png

Every metric is a ratio of sums over stories (e.g. recall accuracy = sum of correct recall
tokens / sum of recall tokens), computed only on positions evaluated under eviction. 95% CIs are
percentile bootstraps over stories (2000 resamples). The same resamples are used for every
config, so differences between configs (vs full cache, vs H2O) are paired.
"""

import glob
import re
import json
import os

import numpy as np

RES = os.path.join("reef", "results")
FIG = os.path.join("reef", "figures")
N_BOOT = 2000

# Reference categorical palette (dataviz skill, light mode): one hue per policy family, in fixed
# order; parameter variants of a family share its hue and differ by line style.
FAMILY_COLOR = {"window": "#2a78d6", "h2o": "#eb6834", "tiu": "#1baf7a", "rw": "#eda100", "ema": "#e87ba4"}
STYLE = {"window": ("window", "-"), "h2o": ("h2o", "-"), "tiu": ("tiu", "-"), "tiu/7": ("tiu", "--"),
         "rw16": ("rw", "-"), "rw8": ("rw", ":"), "rw32": ("rw", "--"), "ema0.9": ("ema", "-"), "ema0.99": ("ema", "--")}
CORE = ["window", "h2o", "tiu", "rw16", "ema0.9"]
NAMES = {"window": "sliding window", "h2o": "H2O (float)", "tiu": "TIU (8-bit sat.)", "tiu/7": "TIU (1/8-scale weight, 1.0 -> 32)",
         "rw8": "recent-window W=8", "rw16": "recent-window W=16", "rw32": "recent-window W=32",
         "ema0.9": "EMA a=0.9", "ema0.99": "EMA a=0.99", "full": "full cache",
         "h2o_sim": "H2O, TIU-team simulation (non-permanent)",
         "h2o_sim_nolook": "H2O, TIU-team simulation, no lookahead"}

METRICS = {
    # name: (numerator, denominator, transform, label)
    "recall_acc": ("r_correct", "rn", None, "recall-token top-1 accuracy"),
    "recall_nll": ("r_nll", "rn", None, "recall-token loss (nats)"),
    "name_acc": ("r_correct_name", "rn_name", None, "name-token top-1 accuracy"),
    "ppl": ("nll", "n", "exp", "perplexity"),
    "agree": ("agree", "n", None, "top-1 agreement with full cache"),
    "kl": ("kl", "n", None, "KL(full || policy), nats/token"),
    "acc": ("correct", "n", None, "next-token top-1 accuracy"),
}


def load(pattern, device=None):
    """JSONL rows matching `pattern` in RES (or RES/gpu with device='gpu': tagged so they pair only
    with other GPU rows)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, "gpu" if device == "gpu" else "", pattern))):
        with open(path) as f:
            rows += [dict(json.loads(line), **({"device": device} if device else {})) for line in f]
    return rows


def table(rows):
    """{config key: {field: np.array over stories (sorted story ids)}}, and the story ids."""
    stories = sorted({r["story"] for r in rows})
    index = {s: i for i, s in enumerate(stories)}
    out = {}
    for r in rows:
        setting = r["quant"] if r.get("evict_layers", "all") == "all" else r["quant"] + ", global layers only"
        if r.get("kv", "none") != "none":
            setting += ", KV " + r["kv"]
        if r.get("device") == "gpu":
            setting += " (GPU)"
        key = (setting, r["policy"], r["granularity"], r["protect"], r["budget"])
        d = out.setdefault(key, {})
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("story", "batch", "seed", "gap"):
                d.setdefault(k, np.zeros(len(stories)))[index[r["story"]]] = v
        d.setdefault("_seen", np.zeros(len(stories), bool))[index[r["story"]]] = True
    for key, d in out.items():
        if not d["_seen"].all():
            raise ValueError(f"config {key} is missing {int((~d['_seen']).sum())} stories")
    return out, stories


def boot_weights(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)]).astype(float)


def metric(d, name, Wb):
    """(point, bootstrap samples) for one config."""
    num, den, tr, _ = METRICS[name]
    point = d[num].sum() / d[den].sum()
    samples = (Wb @ d[num]) / np.maximum(Wb @ d[den], 1e-12)
    if tr == "exp":
        point, samples = np.exp(point), np.exp(samples)
    return point, samples


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def full_key(key):
    quant, _, _, _, budget = key
    return (quant, "full", "head", "none", budget)


def summarize(tab, Wb):
    out = []
    for key, d in sorted(tab.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3], kv[0][1], kv[0][4])):
        quant, policy, gran, protect, budget = key
        fk = full_key(key)
        h2o_key = (quant, "h2o", gran, protect, budget)
        entry = dict(quant=quant, policy=policy, granularity=gran, protect=protect, budget=budget,
                     positions=int(d["n"].sum()), recall_tokens=int(d["rn"].sum()), name_tokens=int(d["rn_name"].sum()))
        for m in METRICS:
            p, s = metric(d, m, Wb)
            entry[m] = float(p)
            entry[m + "_ci"] = ci(s)
            if policy != "full":
                fp, fs = metric(tab[fk], m, Wb)
                if m == "ppl":
                    entry["ppl_ratio"] = float(p / fp)
                    entry["ppl_ratio_ci"] = ci(s / fs)
                else:
                    entry[m + "_minus_full"] = float(p - fp)
                    entry[m + "_minus_full_ci"] = ci(s - fs)
                if h2o_key in tab and policy != "h2o":
                    hp, hs = metric(tab[h2o_key], m, Wb)
                    entry[m + "_minus_h2o"] = float(p - hp)
                    entry[m + "_minus_h2o_ci"] = ci(s - hs)
        if "tiu_decisions" in d:
            dec = d["tiu_decisions"].sum()
            entry["tiu_saturated_frac"] = float(d["tiu_saturated"].sum() / dec)
            entry["tiu_tied_frac"] = float(d["tiu_tied"].sum() / dec)
        out.append(entry)
    return out


def check_reference(tab):
    """Each policy line carries the full-cache sums for its own positions; they must equal the
    full-cache lines for the same budget (a cross-check between the two code paths)."""
    worst = 0.0
    for key, d in tab.items():
        if key[1] == "full":
            continue
        f = tab[full_key(key)]
        assert np.array_equal(d["n"], f["n"]), key
        assert np.array_equal(d["rn"], f["rn"]), key
        worst = max(worst, float(np.abs(d["nll_full"] - f["nll"]).max()))
        assert np.array_equal(d["r_correct_full"], f["r_correct"]), key
    return worst


def fmt(x, c, pct=False, digits=3):
    if pct:
        return f"{100 * x:.1f} [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
    return f"{x:.{digits}f} [{c[0]:.{digits}f}, {c[1]:.{digits}f}]"


def markdown(summary, quant="fp32", only=None):
    md = []
    budgets = sorted({e["budget"] for e in summary})
    for gran, protect, title in [("head", "sink+recent", "per-head, 4 sinks + recent window (half the budget)"),
                                 ("head", "gold", "per-head, TIU team's gold config (no sinks, recent C/2)"),
                                 ("head", "sink", "per-head, 4 sinks, no recent window"),
                                 ("layer", "sink+recent", "one keep set per layer, 4 sinks + recent window")]:
        if not any((e["quant"], e["granularity"], e["protect"]) == (quant, gran, protect) for e in summary):
            continue
        if only and protect not in only:
            continue
        md += [f"\n### {quant}: {title}\n",
               "| budget | policy | recall acc % | Δ vs full (pts) | Δ vs H2O (pts) | ppl / full | top-1 agree % | KL |",
               "|---|---|---|---|---|---|---|---|"]
        for b in budgets:
            full = next(e for e in summary if e["quant"] == quant and e["policy"] == "full" and e["budget"] == b)
            md.append(f"| {b} | full cache | {fmt(full['recall_acc'], full['recall_acc_ci'], True)} | | | 1 "
                      f"(ppl {full['ppl']:.2f}) | 100 | 0 |")
            for e in summary:
                if (e["quant"], e["granularity"], e["protect"], e["budget"]) != (quant, gran, protect, b) or e["policy"] == "full":
                    continue
                dh = fmt(e["recall_acc_minus_h2o"], e["recall_acc_minus_h2o_ci"], True) if "recall_acc_minus_h2o" in e else ""
                md.append(f"| {b} | {NAMES.get(e['policy'], e['policy'])} | {fmt(e['recall_acc'], e['recall_acc_ci'], True)} | "
                          f"{fmt(e['recall_acc_minus_full'], e['recall_acc_minus_full_ci'], True)} | {dh} | "
                          f"{fmt(e['ppl_ratio'], e['ppl_ratio_ci'])} | {fmt(e['agree'], e['agree_ci'], True)} | "
                          f"{e['kl']:.3f} |")
    return "\n".join(md)


def figure(summary, quant, gran, protect, policies, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    panels = [("recall_acc", True), ("ppl_ratio", False), ("agree", True), ("kl", False)]
    titles = dict({m: v[3] for m, v in METRICS.items()}, ppl_ratio="perplexity / full-cache perplexity (paired)")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    budgets = sorted({e["budget"] for e in summary if e["quant"] == quant})
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    for ax, (m, pct) in zip(axes, panels):
        sc = 100 if pct else 1
        full = [next(e for e in summary if e["quant"] == quant and e["policy"] == "full" and e["budget"] == b) for b in budgets]
        if m == "recall_acc":
            ax.errorbar(budgets, [sc * e[m] for e in full],
                        yerr=[[sc * (e[m] - e[m + "_ci"][0]) for e in full], [sc * (e[m + "_ci"][1] - e[m]) for e in full]],
                        color=muted, ls="--", lw=1.5, marker="o", ms=4, capsize=0, label=NAMES["full"])
        elif m == "ppl_ratio":
            ax.axhline(1.0, color=muted, ls="--", lw=1.5)
        for i, p in enumerate(policies):
            es = [next((e for e in summary if (e["quant"], e["policy"], e["granularity"], e["protect"], e["budget"]) ==
                        (quant, p, gran, protect, b)), None) for b in budgets]
            if p == "window" and es[0] is None:  # window ignores recent / granularity: reuse its run with sinks
                es = [next(e for e in summary if (e["quant"], e["policy"], e["protect"], e["budget"]) ==
                           (quant, p, "sink+recent", b)) for b in budgets]
            if any(e is None for e in es):
                continue
            off = (i - (len(policies) - 1) / 2) * 0.035  # small horizontal dodge so CIs don't overlap
            x = [b * (1 + off) for b in budgets]
            y = [sc * e[m] for e in es]
            lo = [sc * (e[m] - e[m + "_ci"][0]) for e in es]
            hi = [sc * (e[m + "_ci"][1] - e[m]) for e in es]
            fam, ls = STYLE[p]
            ax.errorbar(x, y, yerr=[lo, hi], color=FAMILY_COLOR[fam], ls=ls, lw=2, marker="o", ms=5, capsize=0,
                        elinewidth=1.5, label=NAMES[p])
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("KV budget (slots per layer-head)", color=muted)
        ax.set_title(titles[m] + (" (%)" if pct else ""), fontsize=10, color=ink)
        if m in ("ppl_ratio", "kl"):
            ax.set_yscale("log")
            if m == "ppl_ratio":
                lo_, hi_ = ax.get_ylim()
                cands = (1, 1.1, 1.25, 1.5, 2, 3) if hi_ < 5 else (1, 2, 5, 10, 20, 50)
                ticks = [t for t in cands if lo_ <= t <= hi_]
                ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.08, fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = (load("main.part*.jsonl") + load("int8.jsonl") + load("global.jsonl") + load("sim.jsonl")
            + load("gold*.jsonl") + load("kvq_G*.jsonl")  # from reproduce.py on CPU, if present
            + load("kvq_G*.jsonl", "gpu") + load("main.part*.jsonl", "gpu") + load("gold.jsonl", "gpu")
            + load("sim.jsonl", "gpu"))
    tab, stories = table(rows)
    Wb = boot_weights(len(stories))
    worst = check_reference(tab)
    summary = summarize(tab, Wb)
    quants = sorted({e["quant"] for e in summary})
    md = [f"# Results summary\n\n{len(stories)} stories, {N_BOOT} paired bootstrap resamples over stories; "
          f"95% percentile CIs in brackets. Cross-check of reference sums between code paths: "
          f"max |diff| {worst:.2e} nats.\n"]
    for q in quants:
        # The GPU copy of the main sweep exists to pair with the GPU-only gold run (and for
        # device_check.md); its other tables would repeat the CPU ones.
        md.append(markdown(summary, q, only=("gold", "sink+recent") if q == "fp32 (GPU)" else None))
    tiu = [e for e in summary if "tiu_saturated_frac" in e]
    if tiu:
        md += ["\n### TIU eviction decisions\n", "| quant | policy | granularity | protect | budget | all candidates saturated at 255 | min score tied |",
               "|---|---|---|---|---|---|---|"]
        for e in tiu:
            md.append(f"| {e['quant']} | {e['policy']} | {e['granularity']} | {e['protect']} | {e['budget']} | "
                      f"{100 * e['tiu_saturated_frac']:.1f}% | {100 * e['tiu_tied_frac']:.1f}% |")
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    figure(summary, "fp32", "head", "sink+recent", CORE, os.path.join(FIG, "main_protected.png"),
           "Per-head eviction, 4 sinks + recent window (half the budget) protected")
    figure(summary, "fp32", "head", "sink", CORE, os.path.join(FIG, "main_no_recent.png"),
           "Per-head eviction, 4 sinks protected, no recent window")
    figure(summary, "fp32", "layer", "sink+recent", CORE[1:], os.path.join(FIG, "shared_per_layer.png"),
           "One keep set per layer (all heads share it), 4 sinks + recent window")
    figure(summary, "fp32", "head", "sink+recent", ["h2o", "tiu", "tiu/7", "rw8", "rw16", "rw32", "ema0.9", "ema0.99"],
           os.path.join(FIG, "variants_protected.png"), "Parameter variants (per-head, sinks + recent window)")
    figure(summary, "fp32", "head", "sink", ["h2o", "tiu", "tiu/7", "rw8", "rw16", "rw32", "ema0.9", "ema0.99"],
           os.path.join(FIG, "variants_no_recent.png"), "Parameter variants (per-head, sinks only, no recent window)")
    figure(summary, "fp32 (GPU)", "head", "gold", CORE, os.path.join(FIG, "gold_config.png"),
           "Per-head eviction, TIU team's gold config: no sinks, recent window C/2")
    for q in quants:
        if ", KV " in q:
            G = re.search(r"_G(\d+)", q).group(1)
            figure(summary, q, "head", "sink+recent", CORE, os.path.join(FIG, f"kvq_G{G}_protected.png"),
                   f"KV Cache Engine codecs (CQ-4+ keys, G={G}; CQ-3-rot values): sinks + recent window")
            figure(summary, q, "head", "gold", CORE[1:], os.path.join(FIG, f"kvq_G{G}_gold.png"),
                   f"KV Cache Engine codecs (G={G}), TIU team's gold config")
    if "int8" in quants:
        # The int8 grid ran EMA alpha = 0.99 (see run.grid), so that is the EMA line here.
        figure(summary, "int8", "head", "sink+recent", ["window", "h2o", "tiu", "rw16", "ema0.99"],
               os.path.join(FIG, "int8_protected.png"),
               "Simulated int8 weights: per-head eviction, sinks + recent window")
    print(f"wrote {RES}/summary.md, {RES}/summary.json and figures in {FIG}")


if __name__ == "__main__":
    main()
