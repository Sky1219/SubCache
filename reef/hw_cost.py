"""Hardware-cost arithmetic for streaming eviction on Reef. No RTL, no measurements: every number
below follows from the TinyStories-28M config and the assumptions in ASSUMPTIONS.

    python -m reef.hw_cost      -> reef/results/hw_cost.json and reef/results/hw_cost.md
"""

import json
import math
import os

from huggingface_hub import hf_hub_download

ASSUMPTIONS = {
    "dtcm_bytes": 32 * 1024,             # Reef data TCM (the 8 KB ITCM holds code)
    "hyperram_bytes_per_s": 90e6,        # off-chip HyperRAM bandwidth, ~90 MB/s
    "clock_hz": 16e6,                    # low end of Reef's 16-20 MHz
    "weight_bytes": 1,                   # int8 weights
    "kv_bytes": {"int8": 1, "fp16": 2},
    "score_bits": {"h2o": 8, "ema": 16, "rw": 8},  # TIU uses SCORE_WIDTH = 8; EMA keeps 8 fraction bits
    "attn_weight_bits": 8,               # attention probabilities fed to the scorer (TIU WEIGHT_WIDTH)
    "weights_streamed_every_token": True,  # 32 KB DTCM cannot hold any meaningful share of ~51 MB
    "argmin": "serial, 1 compare/cycle, N+1 cycles per scan (as in the TIU RTL)",
}


def config():
    with open(hf_hub_download("roneneldan/TinyStories-28M", "config.json")) as f:
        return json.load(f)


def model_numbers(cfg):
    d, L, H, V = cfg["hidden_size"], cfg["num_layers"], cfg["num_heads"], cfg["vocab_size"]
    ff = cfg["intermediate_size"] or 4 * d
    attn_w = 4 * d * d                      # q, k, v, out (no q/k/v bias in GPT-Neo)
    mlp_w = 2 * d * ff
    small = (d + ff + d) + 4 * d            # out/fc/proj biases + two layer norms (weight + bias)
    per_layer = attn_w + mlp_w + small
    lm_head = V * d                         # tied to the token embedding
    return dict(d=d, layers=L, heads=H, head_dim=d // H, vocab=V, ff=ff, window=cfg["window_size"],
                kinds=cfg["attention_layers"], ctx=cfg["max_position_embeddings"],
                per_layer_params=per_layer, block_params=L * per_layer, lm_head_params=lm_head,
                final_ln=2 * d)


def kv_numbers(m, a):
    out = {}
    for name, b in a["kv_bytes"].items():
        per_layer = 2 * m["d"] * b          # one K and one V vector of d elements
        total = m["layers"] * per_layer
        out[name] = dict(bytes_per_token_per_layer=per_layer, bytes_per_token=total,
                         tokens_in_dtcm_all_layers=a["dtcm_bytes"] // total,
                         tokens_in_dtcm_one_layer=a["dtcm_bytes"] // per_layer)
    return out


def traffic(m, a, budgets, story_len=201):
    """Bytes read from HyperRAM per decoded token, and the implied time at the assumed bandwidth.
    KV read = (budget + 1) slots per layer (the cache plus the new token), int8."""
    bw = a["hyperram_bytes_per_s"]
    w = (m["block_params"] + m["lm_head_params"] + m["final_ln"]) * a["weight_bytes"] + 2 * m["d"]  # + wte/wpe rows
    kvb = 2 * m["d"] * a["kv_bytes"]["int8"]
    n_local = sum(k == "local" for k in m["kinds"])
    n_global = m["layers"] - n_local
    rows = []
    cases = [(f"budget {b}", [b + 1] * m["layers"]) for b in budgets]
    cases.append((f"full cache, median story ({story_len} tok)", [story_len] * m["layers"]))
    cases.append((f"full cache at ctx {m['ctx']}", [m["ctx"]] * n_global + [m["window"]] * n_local))
    for name, slots in cases:
        kv = sum(slots) * kvb
        rows.append(dict(case=name, kv_read_bytes=kv, kv_ms=1e3 * kv / bw, weight_bytes=w, weight_ms=1e3 * w / bw,
                         kv_share=kv / (kv + w), ms_per_token=1e3 * (kv + w) / bw))
    return rows, w


def policy_costs(m, a, N=128, W=16):
    """Per (layer, head) slot state, scorer work per decoded token, and totals for the whole model
    at a budget of N slots, per-head and with one keep set per layer ('shared')."""
    LH = m["layers"] * m["heads"]
    sb = a["score_bits"]
    lg = math.ceil(math.log2(m["ctx"]))
    rows = [
        dict(policy="sliding window + sinks", state_bits_per_slot=0, extra="ring pointer per (layer, head) (" + str(lg) + " bits)",
             adds=0, muls=0, compares=0, argmin=False, buffer_bits=0),
        dict(policy="H2O (float) / TIU (8-bit sat.)", state_bits_per_slot=sb["h2o"], extra="none",
             adds=N + 1, muls=0, compares=N, argmin=True, buffer_bits=0),
        dict(policy="EMA, alpha = 1 - 2^-k", state_bits_per_slot=sb["ema"], extra="none (shift-subtract)",
             adds=2 * (N + 1), muls=0, compares=N, argmin=True, buffer_bits=0),
        dict(policy="EMA, general alpha (0.9, 0.99)", state_bits_per_slot=sb["ema"], extra="none",
             adds=N + 1, muls=N + 1, compares=N, argmin=True, buffer_bits=0),
        dict(policy=f"recent-window attention, W={W}", state_bits_per_slot=sb["rw"],
             extra=f"last W attention rows ({W} x N x {a['attn_weight_bits']} b)",
             adds=2 * (N + 1), muls=0, compares=N, argmin=True, buffer_bits=W * (N + 1) * a["attn_weight_bits"]),
    ]
    for r in rows:
        per_head = (r["state_bits_per_slot"] * (N + 1) + r["buffer_bits"]) / 8
        r["state_bytes_per_head"] = per_head
        r["state_bytes_total_per_head_mode"] = per_head * LH
        r["state_bytes_total_shared_mode"] = per_head * m["layers"]
        r["ops_per_token_per_head_mode"] = (r["adds"] + r["muls"] + r["compares"]) * LH
        r["argmin_cycles_per_token_per_head_mode"] = (N + 1) * LH if r["argmin"] else 0
        r["argmin_cycles_per_token_shared"] = (N + 1) * m["layers"] if r["argmin"] else 0
        r["argmin_ms_per_token_per_head_mode"] = 1e3 * r["argmin_cycles_per_token_per_head_mode"] / a["clock_hz"]
    return rows


def fmt_bytes(x):
    if x >= 1 << 20:
        return f"{x / (1 << 20):.2f} MiB"
    if x >= 1 << 10:
        return f"{x / (1 << 10):.1f} KiB"
    return f"{x:.0f} B"


def main():
    a = ASSUMPTIONS
    m = model_numbers(config())
    kv = kv_numbers(m, a)
    budgets = [16, 32, 64, 128]
    tr, w = traffic(m, a, budgets)
    pc = policy_costs(m, a)
    out = dict(assumptions=a, model=m, kv=kv, traffic=tr, policies=pc)
    os.makedirs(os.path.join("reef", "results"), exist_ok=True)
    with open(os.path.join("reef", "results", "hw_cost.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = ["**KV footprint** (K and V, d = %d, %d layers)\n" % (m["d"], m["layers"]),
          "| precision | per token per layer | per token, all layers | tokens of KV in 32 KB DTCM (all layers) | (one layer only) |",
          "|---|---|---|---|---|"]
    for name, k in kv.items():
        md.append(f"| {name} | {k['bytes_per_token_per_layer']} B | {fmt_bytes(k['bytes_per_token'])} | "
                  f"{k['tokens_in_dtcm_all_layers']} | {k['tokens_in_dtcm_one_layer']} |")
    md += ["", "**HyperRAM read traffic per decoded token** (int8 weights and KV, %.0f MB/s)\n" % (a["hyperram_bytes_per_s"] / 1e6),
           "| case | KV read | KV time | weights read | weight time | KV share | total ms/token |", "|---|---|---|---|---|---|---|"]
    for r in tr:
        md.append(f"| {r['case']} | {fmt_bytes(r['kv_read_bytes'])} | {r['kv_ms']:.1f} ms | {fmt_bytes(r['weight_bytes'])} | "
                  f"{r['weight_ms']:.0f} ms | {100 * r['kv_share']:.1f}% | {r['ms_per_token']:.0f} |")
    md += ["", "**Scorer cost per policy** at N = 128 slots per (layer, head), %d layers x %d heads\n" % (m["layers"], m["heads"]),
           "| policy | state / slot | extra buffering | ops per token per (layer, head) | scorer state, per-head keep sets | shared per layer | serial argmin, per-head |",
           "|---|---|---|---|---|---|---|"]
    for r in pc:
        ops = f"{r['adds']} add" + (f", {r['muls']} mul" if r["muls"] else "") + (f", {r['compares']} cmp" if r["compares"] else "")
        am = (f"{r['argmin_cycles_per_token_per_head_mode']:,} cyc = {r['argmin_ms_per_token_per_head_mode']:.2f} ms"
              if r["argmin"] else "-")
        md.append(f"| {r['policy']} | {r['state_bits_per_slot']} b | {r['extra']} | {ops if r['adds'] else '-'} | "
                  f"{fmt_bytes(r['state_bytes_total_per_head_mode'])} | {fmt_bytes(r['state_bytes_total_shared_mode'])} | {am} |")
    with open(os.path.join("reef", "results", "hw_cost.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nweights per token: {w:,} B; params: blocks {m['block_params']:,}, lm_head/wte {m['lm_head_params']:,}")


if __name__ == "__main__":
    main()
