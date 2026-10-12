"""Draw the README figures from the result files: python scripts/make_figures.py

Writes figures/<name>-light.svg and figures/<name>-dark.svg. No plotting library is needed;
the SVG is written directly. Every number is read from results/batch/*.jsonl, so the figures
change only if the results do.
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import paired_ci  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", ramp=["#184f95", "#256abf", "#3987e5", "#6da7ec"]),
}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
BUDGET_LABEL = {0.5: "50%", 0.25: "25%", 0.125: "12.5%", 0.0625: "6.25%", 0.03125: "3.1%"}


def rows(name):
    with open(os.path.join(ROOT, "results", "batch", f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def text(x, y, s, fill, size=12, anchor="start", weight="400"):
    return f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'


def frame(w, h, t, title, subtitle, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{title}">'
            f'<rect width="{w}" height="{h}" rx="8" fill="{t["surface"]}"/>'
            + text(24, 32, title, t["ink"], 15, weight="600") + text(24, 52, subtitle, t["ink2"], 12)
            + body + "</svg>")


def hbar(x0, y, length, thick, color):
    """Horizontal bar from x0: square at the baseline, 4px rounded at the data end."""
    r = min(4, max(length, 0) / 2)
    x1 = x0 + length
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + thick - r:.1f} '
            f'Q{x1:.1f},{y + thick:.1f} {x1 - r:.1f},{y + thick:.1f} H{x0:.1f} Z" fill="{color}"/>')


def vbar(x, y0, height, width, color):
    """Vertical bar from baseline y0; positive height goes up. Rounded at the data end."""
    r = min(4, abs(height) / 2)
    y1 = y0 - height
    s = 1 if height >= 0 else -1
    return (f'<path d="M{x:.1f},{y0:.1f} V{y1 + s * r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + width - r:.1f} '
            f'Q{x + width:.1f},{y1:.1f} {x + width:.1f},{y1 + s * r:.1f} V{y0:.1f} Z" fill="{color}"/>')


# ------------------------------------------------------------------ figure 1: throughput
def fig_throughput(t):
    speed = rows("speed")
    core = rows("core8k_late")
    full_ppl = {r["budget"]: r["ppl_post"] for r in core if r["policy"] == "full"}
    h2o_ppl = {r["budget"]: r["ppl_post"] for r in core if r["policy"] == "h2o"}

    def best(policy, budget):
        per_rep = []
        for rep in ("1", "2", "3"):
            ok = [r for r in speed if r["settings"]["tag"] == f"speed8k_rep{rep}" and r["policy"] == policy
                  and abs(r["budget"] - budget) < 1e-9 and not r.get("oom")]
            top = max(ok, key=lambda r: r["batch"])
            per_rep.append((top["batch"], top["decode_tok_s"]))
        return per_rep[0][0], statistics.median(v for _, v in per_rep)

    items = [("Full cache", *best("full", 1.0), "reference")]
    for budget, name in ((0.5, "50% budget"), (0.25, "25% budget"), (0.125, "12.5% budget"), (0.0625, "6.25% budget")):
        cost = f"+{100 * (h2o_ppl[budget] / full_ppl[budget] - 1):.0f}% perplexity" if budget in h2o_ppl else "perplexity not measured"
        items.append((name, *best("h2o", budget), cost))

    w, h, x0, top, step, thick = 760, 300, 150, 84, 40, 22
    scale = (w - x0 - 300) / max(v for _, _, v, _ in items)
    body = ""
    for i, (name, batch, tok_s, cost) in enumerate(items):
        y = top + i * step
        body += text(x0 - 12, y + 15, name, t["ink"], 12, "end")
        body += hbar(x0, y, tok_s * scale, thick, t["s1"])
        body += text(x0 + tok_s * scale + 8, y + 15, f"{tok_s:,.0f} tok/s", t["ink"], 12, weight="600")
        body += text(w - 24, y + 15, f"batch {batch} · {cost}", t["ink2"], 11, "end")
    body += f'<line x1="{x0}" y1="{top - 8}" x2="{x0}" y2="{top + len(items) * step - 8}" stroke="{t["axis"]}" stroke-width="1"/>'
    body += text(24, h - 16, "DeepSeek-coder-1.3B, fp16, RTX 2080 Ti · 8,192-token agent sessions · largest batch tested that fits in 11 GB · median of 3 runs",
                 t["muted"], 10.5)
    return frame(w, h, t, "Evicting cache entries raises throughput by fitting more sequences",
                 "Decode throughput with H2O eviction at each cache budget, and what it costs in perplexity", body)


# ------------------------------------------------------------------ figure 2: wfl vs H2O
def fig_redundancy(t):
    core = rows("core8k_late")
    budgets = [0.25, 0.125, 0.0625, 0.03125]
    key = "nll_post_masked_per_seq"
    ref = {r["budget"]: r for r in core if r["policy"] == "h2o"}
    series = []
    for mix, name in ((0.25, "redundancy weight 0.25"), (0.5, "redundancy weight 0.5"), (1.0, "redundancy weight 1.0")):
        pts = []
        for b in budgets:
            r = next((x for x in core if x["policy"] == "wfl" and x["budget"] == b and x["settings"]["mix"] == mix), None)
            if r is None:
                break
            ratio, lo, hi, _ = paired_ci(r[key], ref[b][key], n_boot=4000)
            pts.append((100 * (ratio - 1), 100 * (lo - 1), 100 * (hi - 1)))
        if len(pts) == len(budgets):
            series.append((name, pts))
    series = series[:2] if len(series) < 3 else [series[0], series[2]]  # keep two: the pre-set one and the strongest

    w, h, left, right, top, bottom = 760, 400, 70, 190, 96, 84
    lo_y = min(p[1] for _, pts in series for p in pts)
    ymin = -(int(-lo_y * 2) + 1) / 2  # round down to the next half percent
    ymax = 0.5
    px = lambda i: left + i * (w - left - right) / (len(budgets) - 1)
    py = lambda v: top + (ymax - v) / (ymax - ymin) * (h - top - bottom)
    body = ""
    tick = ymin
    while tick <= ymax + 1e-9:
        col = t["axis"] if abs(tick) < 1e-9 else t["grid"]
        body += f'<line x1="{left}" y1="{py(tick):.1f}" x2="{w - right}" y2="{py(tick):.1f}" stroke="{col}" stroke-width="1"/>'
        body += text(left - 10, py(tick) + 4, f"{tick:+.1f}%" if abs(tick) > 1e-9 else "0%", t["muted"], 11, "end")
        tick += 0.5
    body += text(w - right + 10, py(0) + 4, "H2O (baseline)", t["ink2"], 11)
    for i, b in enumerate(budgets):
        body += text(px(i), h - bottom + 20, BUDGET_LABEL[b], t["muted"], 11, "middle")
    body += text((left + w - right) / 2, h - bottom + 38, "cache budget (share of tokens kept)", t["ink2"], 11, "middle")
    colors = [t["s1"], t["s2"]]
    for (name, pts), color in zip(series, colors):
        for i, (mid, lo, hi) in enumerate(pts):
            body += f'<line x1="{px(i):.1f}" y1="{py(lo):.1f}" x2="{px(i):.1f}" y2="{py(hi):.1f}" stroke="{color}" stroke-width="1.5" opacity="0.55"/>'
        path = " ".join(f'{"M" if i == 0 else "L"}{px(i):.1f},{py(p[0]):.1f}' for i, p in enumerate(pts))
        body += f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        for i, p in enumerate(pts):
            body += f'<circle cx="{px(i):.1f}" cy="{py(p[0]):.1f}" r="4.5" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>'
        body += text(w - right + 10, py(pts[-1][0]) + 4, f"{name}: {pts[-1][0]:+.1f}%", t["ink"], 11)
    for i, ((name, _), color) in enumerate(zip(series, colors)):  # legend
        x = left + i * 210
        body += f'<line x1="{x}" y1="74" x2="{x + 18}" y2="74" stroke="{color}" stroke-width="2"/><circle cx="{x + 9}" cy="74" r="4" fill="{color}"/>'
        body += text(x + 26, 78, name, t["ink2"], 11)
    body += text(24, h - 14, "Perplexity on the agent's own tokens, relative to H2O at the same budget · 64 paired sessions · bars are 95% intervals",
                 t["muted"], 10.5)
    return frame(w, h, t, "A redundancy term beats H2O on repetitive agent sessions, by a small margin",
                 "Lower is better. Importance-weighted facility location on top of H2O's score", body)


# ------------------------------------------------------------------ figure 3: gain vs repetition
def fig_repetition(t):
    budgets = [0.25, 0.125, 0.0625, 0.03125]
    sets = [("core8k_wiki", "WikiText", "0.2% repeated"), ("core8k_start", "Agent session start", "8% repeated"),
            ("core8k_late", "Agent session, 24k tokens in", "26% repeated")]
    data = []
    for file, name, rep in sets:
        rs = rows(file)
        ref = {r["budget"]: r for r in rs if r["policy"] == "h2o"}
        vals = []
        for b in budgets:
            r = next(x for x in rs if x["policy"] == "wfl" and x["budget"] == b and x["settings"]["mix"] == 0.25)
            ratio, _, _, _ = paired_ci(r["nll_post_per_seq"], ref[b]["nll_post_per_seq"], n_boot=500)
            vals.append(100 * (ratio - 1))
        data.append((name, rep, vals))

    w, h, left, right, top, bottom = 760, 380, 70, 30, 100, 84
    ymax = max(0.2, max(v for _, _, vals in data for v in vals))
    ymax = (int(ymax * 10) + 1) / 10
    ymin = -0.4
    py = lambda v: top + (ymax - v) / (ymax - ymin) * (h - top - bottom)
    group = (w - left - right) / len(data)
    bar, gap = 22, 2
    body = ""
    tick = ymin
    while tick <= ymax + 1e-9:
        col = t["axis"] if abs(tick) < 1e-9 else t["grid"]
        body += f'<line x1="{left}" y1="{py(tick):.1f}" x2="{w - right}" y2="{py(tick):.1f}" stroke="{col}" stroke-width="1"/>'
        body += text(left - 10, py(tick) + 4, f"{tick:+.1f}%" if abs(tick) > 1e-9 else "0%", t["muted"], 11, "end")
        tick += 0.2
    for g, (name, rep, vals) in enumerate(data):
        start = left + g * group + (group - (4 * bar + 3 * gap)) / 2
        for i, v in enumerate(vals):
            body += vbar(start + i * (bar + gap), py(0), py(0) - py(v), bar, t["ramp"][i])
        cx = left + g * group + group / 2
        body += text(cx, h - bottom + 24, name, t["ink"], 12, "middle")
        body += text(cx, h - bottom + 41, rep, t["ink2"], 11, "middle")
    for i, b in enumerate(budgets):  # legend
        x = left + i * 120
        body += f'<rect x="{x}" y="68" width="12" height="12" rx="2" fill="{t["ramp"][i]}"/>' + text(x + 18, 78, f"{BUDGET_LABEL[b]} budget", t["ink2"], 11)
    body += text(24, h - 14, "Perplexity on all tokens relative to H2O, redundancy weight 0.25 · 64 sequences of 8,192 tokens per dataset · "
                             "below zero is better", t["muted"], 10.5)
    return frame(w, h, t, "The gain appears only where the context repeats itself",
                 "Change in perplexity from adding the redundancy term, by dataset and cache budget", body)


# ------------------------------------------------------------------ figure 4: selection cost
def fig_selection(t):
    speed = rows("speed")

    def share(tag, policy, budget):
        ok = [r for r in speed if r["settings"]["tag"] == tag and r["policy"] == policy
              and abs(r["budget"] - budget) < 1e-9 and not r.get("oom")]
        top = max(ok, key=lambda r: r["batch"])
        return 100 * top["select_frac"], top["decode_tok_s"], top["batch"]

    items = [("H2O", *share("speed8k_rep1", "h2o", 0.125)), ("R-KV-style", *share("speed8k_rep1", "rkv", 0.125)),
             ("Redundancy term", *share("speed8k_wfl", "wfl", 0.125))]
    w, h, x0, top, step, thick = 760, 230, 150, 84, 40, 22
    scale = (w - x0 - 300) / 100
    body = ""
    for i, (name, pct, tok_s, batch) in enumerate(items):
        y = top + i * step
        body += text(x0 - 12, y + 15, name, t["ink"], 12, "end")
        body += hbar(x0, y, max(pct * scale, 2), thick, t["s1"])
        body += text(x0 + max(pct * scale, 2) + 8, y + 15, f"{pct:.0f}%" if pct >= 1 else "under 1%", t["ink"], 12, weight="600")
        body += text(w - 24, y + 15, f"{tok_s:,.0f} tok/s at batch {batch}", t["ink2"], 11, "end")
    body += f'<line x1="{x0}" y1="{top - 8}" x2="{x0}" y2="{top + len(items) * step - 8}" stroke="{t["axis"]}" stroke-width="1"/>'
    body += text(24, h - 16, "12.5% cache budget, 8,192-token agent sessions, eviction every 128 tokens · largest batch tested · single runs",
                 t["muted"], 10.5)
    return frame(w, h, t, "Choosing what to evict can cost more than it saves",
                 "Share of decode time spent on selection, and the throughput that results", body)


def main():
    out = os.path.join(ROOT, "figures")
    os.makedirs(out, exist_ok=True)
    for name, fn in (("throughput", fig_throughput), ("redundancy", fig_redundancy), ("repetition", fig_repetition),
                     ("selection", fig_selection)):
        for mode, theme in THEMES.items():
            with open(os.path.join(out, f"{name}-{mode}.svg"), "w", encoding="utf-8", newline="\n") as f:
                f.write(fn(theme))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
