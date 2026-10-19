"""Streaming-eviction sweep on TinyStories-28M: one JSONL line per (config, story).

    python -m reef.run --out reef/results/main.jsonl
    python -m reef.run --out reef/results/int8.jsonl --quant int8 --grid int8

Stories are decoded token by token. For every batch of stories, a full-cache pass gives the
reference distribution at every position; then each policy family runs all of its configs
(budget x protection x parameter) as rows of one batch. Every metric is summed only over
positions whose logits were computed after the first eviction (position t >= budget + 1),
predicting token t + 1. See WRITEUP.md for the metric definitions.
"""

import argparse
import json
import os
import time

import torch

from reef.data import load_stories, recall_targets
from reef.engine import MODEL, NeoEngine, load_model
from reef.policies import Streaming
from reef.kvquant import KVQuant, calibrate_outliers
from reef.quant import quantize_weights_int8

SINK = 4
RECENT_FRAC = 0.5  # SubCache's convention: half the budget is a protected recent window


def grid(name, budgets):
    """List of (family, granularity, rows); a row is a dict describing one config."""
    def rows(budgets, protects, params, label):
        out = []
        for protect in protects:
            for b in budgets:
                for p in params:
                    sink, recent = SINK, (int(round(RECENT_FRAC * b)) if protect == "sink+recent" else 0)
                    if protect == "gold":
                        # The TIU team's gold config (token-importance-unit h2o_analysis.py): no sinks,
                        # a recent window of L = round(0.5 C) tokens counting the current one, the
                        # rest heavy hitters, with C = budget + 1 tokens seen by each query.
                        sink, recent = 0, max(1, round(RECENT_FRAC * (b + 1))) - 1
                    out.append(dict(budget=b, sink=sink, recent=recent, param=p, protect=protect, policy=label(p)))
        return out

    both = ("sink+recent", "sink")
    one = ("sink+recent",)
    tiu_label = lambda p: "tiu" if p == 255 else f"tiu/{255 // p}"  # noqa: E731
    if name == "main":
        return [
            ("window", "head", rows(budgets, one, [None], lambda p: "window")),
            ("h2o", "head", rows(budgets, both, [None], lambda p: "h2o")),
            ("tiu", "head", rows(budgets, both, [255, 32], tiu_label)),
            ("rw", "head", rows(budgets, both, [8, 16, 32], lambda p: f"rw{p}")),
            ("ema", "head", rows(budgets, both, [0.9, 0.99], lambda p: f"ema{p}")),
            ("h2o", "layer", rows(budgets, one, [None], lambda p: "h2o")),
            ("tiu", "layer", rows(budgets, one, [255], tiu_label)),
            ("rw", "layer", rows(budgets, one, [8, 16, 32], lambda p: f"rw{p}")),
            ("ema", "layer", rows(budgets, one, [0.9, 0.99], lambda p: f"ema{p}")),
        ]
    if name == "global":  # sensitivity check: budget on global layers only, local layers keep their window
        return [
            ("window", "head", rows(budgets, one, [None], lambda p: "window")),
            ("h2o", "head", rows(budgets, one, [None], lambda p: "h2o")),
            ("rw", "head", rows(budgets, one, [16], lambda p: f"rw{p}")),
        ]
    if name == "gold":  # the TIU team's gold protection config, fp32
        gold = ("gold",)
        return [
            ("window", "head", rows(budgets, gold, [None], lambda p: "window")),
            ("h2o", "head", rows(budgets, gold, [None], lambda p: "h2o")),
            ("tiu", "head", rows(budgets, gold, [255, 32], tiu_label)),
            ("rw", "head", rows(budgets, gold, [16], lambda p: f"rw{p}")),
            ("ema", "head", rows(budgets, gold, [0.9], lambda p: f"ema{p}")),
        ]
    if name == "kvq":  # main policies under the KV Cache Engine codecs (run with --kv-quant)
        two = ("sink+recent", "gold")
        return [
            ("window", "head", rows(budgets, one, [None], lambda p: "window")),
            ("h2o", "head", rows(budgets, two, [None], lambda p: "h2o")),
            ("tiu", "head", rows(budgets, two, [255], tiu_label)),
            ("rw", "head", rows(budgets, two, [16], lambda p: f"rw{p}")),
            ("ema", "head", rows(budgets, two, [0.9], lambda p: f"ema{p}")),
        ]
    if name == "int8":
        return [
            ("window", "head", rows(budgets, one, [None], lambda p: "window")),
            ("h2o", "head", rows(budgets, one, [None], lambda p: "h2o")),
            ("tiu", "head", rows(budgets, one, [255], tiu_label)),
            ("rw", "head", rows(budgets, one, [16], lambda p: f"rw{p}")),
            ("ema", "head", rows(budgets, one, [0.99], lambda p: f"ema{p}")),
        ]
    raise ValueError(f"unknown grid {name!r}")


def split(groups, k, n):
    """The k-th of n shares of the groups, balanced by row count (largest first, to the lightest share)."""
    shares = [[] for _ in range(n)]
    load = [0] * n
    for g in sorted(groups, key=lambda g: -len(g[2])):
        i = load.index(min(load))
        shares[i].append(g)
        load[i] += len(g[2])
    return shares[k]


def sums(n_rows, device):
    keys = ["n", "nll", "nll_full", "kl", "agree", "correct", "correct_full",
            "rn", "r_nll", "r_nll_full", "r_correct", "r_correct_full",
            "rn_name", "r_nll_name", "r_nll_full_name", "r_correct_name", "r_correct_full_name"]
    return {k: torch.zeros(n_rows, dtype=torch.float64, device=device) for k in keys}


@torch.inference_mode()
def reference(engine, ids, lens):
    """Full-cache pass. Returns log-probs [B, T, V] (fp32), plus per position the model's top-1
    token and the NLL of the true next token (both [B, T])."""
    B, T = ids.shape
    cache = engine.new_cache(B, T)
    logp = torch.empty(B, T, engine.config.vocab_size, device=engine.device)
    for t in range(T):
        h, _, _ = engine.step(ids[:, t], cache)
        logp[:, t] = torch.log_softmax(engine.logits(h), -1)
    nxt = torch.cat([ids[:, 1:], ids[:, -1:]], 1)
    nll = -logp.gather(2, nxt[..., None]).squeeze(-1)
    return logp, logp.argmax(-1), nll


@torch.inference_mode()
def run_group(engine, family, gran, rows, ids, lens, recall, ref, evict_layers=None):
    """Run one family's configs (rows) on a batch of stories. Returns per-(config, story) sums.
    evict_layers: per-layer bools (default all layers), see policies.Streaming."""
    ref_logp, ref_top1, ref_nll = ref
    B, T = ids.shape
    C = len(rows)
    R = C * B
    dev = engine.device
    story = torch.arange(R, device=dev) % B  # row = config * B + story
    cfg = torch.arange(R, device=dev) // B
    budget = torch.tensor([rows[c]["budget"] for c in cfg.tolist()], device=dev)
    sink = [rows[c]["sink"] for c in cfg.tolist()]
    recent = [rows[c]["recent"] for c in cfg.tolist()]
    param = None if rows[0]["param"] is None else [rows[c]["param"] for c in cfg.tolist()]
    capacity = int(budget.max()) + 1
    if evict_layers is not None and not all(evict_layers):
        capacity = max(capacity, engine.window + 1)
    cache = engine.new_cache(R, capacity)
    pol = Streaming(family, cache, engine.local, engine.window, budget.tolist(), sink, recent, param, gran,
                    evict_layers)
    row_ids = ids[story]
    row_len = lens[story]
    nxt = torch.cat([row_ids[:, 1:], row_ids[:, -1:]], 1)
    # recall[b]: {position j: (kind, dist)} -> masks over prediction steps t = j - 1
    is_recall = torch.zeros(B, T, dtype=torch.bool, device=dev)
    is_name = torch.zeros(B, T, dtype=torch.bool, device=dev)
    for b, targets in enumerate(recall):
        for j, (kind, _) in targets.items():
            is_recall[b, j - 1] = True
            is_name[b, j - 1] = kind == "name"
    acc = sums(R, dev)
    for t in range(T):
        h, attns, slot = engine.step(row_ids[:, t], cache)
        pol.on_load(slot)
        ev = (t >= budget + 1) & (t + 1 < row_len)
        if bool(ev.any()):
            r = ev.nonzero().squeeze(1)
            b = story[r]
            logp = torch.log_softmax(engine.logits(h[r]), -1)
            target = nxt[r, t]
            nll = -logp.gather(1, target[:, None]).squeeze(1).double()
            rl = ref_logp[b, t]
            kl = (rl.exp() * (rl - logp)).sum(-1).double()
            top1 = logp.argmax(-1)
            correct = (top1 == target).double()
            correct_full = (ref_top1[b, t] == target).double()
            nll_full = ref_nll[b, t].double()
            rec = is_recall[b, t].double()
            nam = is_name[b, t].double()
            add = {"n": torch.ones_like(nll), "nll": nll, "nll_full": nll_full, "kl": kl,
                   "agree": (top1 == ref_top1[b, t]).double(), "correct": correct, "correct_full": correct_full,
                   "rn": rec, "r_nll": rec * nll, "r_nll_full": rec * nll_full, "r_correct": rec * correct,
                   "r_correct_full": rec * correct_full, "rn_name": nam, "r_nll_name": nam * nll,
                   "r_nll_full_name": nam * nll_full, "r_correct_name": nam * correct,
                   "r_correct_full_name": nam * correct_full}
            for k, v in add.items():
                acc[k].index_add_(0, r, v)
        pol.update(attns, t)
        pol.evict(cache)
    extra = {}
    if family == "tiu":
        extra = {"tiu_decisions": pol.n_decisions, "tiu_saturated": pol.n_saturated, "tiu_tied": pol.n_tied}
    cpu = lambda d: {k: v.cpu() for k, v in d.items()}  # noqa: E731
    return cfg.cpu(), story.cpu(), cpu(acc), cpu(extra)


def full_lines(budgets, ids, lens, recall, ref):
    """Full-cache reference sums over the positions each budget evaluates (one line per budget)."""
    _, ref_top1, ref_nll = ref
    ref_top1, ref_nll, ids, lens = ref_top1.cpu(), ref_nll.cpu(), ids.cpu(), lens.cpu()
    B, T = ids.shape
    nxt = torch.cat([ids[:, 1:], ids[:, -1:]], 1)
    out = []
    for budget in budgets:
        for b in range(B):
            t = torch.arange(budget + 1, int(lens[b]) - 1)
            c = (ref_top1[b, t] == nxt[b, t]).double()
            nll = ref_nll[b, t].double()
            rec = torch.tensor([j - 1 >= budget + 1 for j in recall[b]], dtype=torch.bool)
            rj = torch.tensor([j - 1 for j in recall[b]], dtype=torch.long)[rec]
            names = torch.tensor([k == "name" for (k, _) in recall[b].values()], dtype=torch.bool)[rec]
            rc = (ref_top1[b, rj] == nxt[b, rj]).double()
            rn = ref_nll[b, rj].double()
            s = {"n": len(t), "nll": nll.sum().item(), "kl": 0.0, "agree": len(t), "correct": c.sum().item(),
                 "rn": int(len(rj)), "r_nll": rn.sum().item(), "r_correct": rc.sum().item(),
                 "rn_name": int(names.sum()), "r_nll_name": rn[names].sum().item(),
                 "r_correct_name": rc[names].sum().item()}
            for k in ("nll", "correct", "r_nll", "r_correct", "r_nll_name", "r_correct_name"):
                s[k + "_full"] = s[k]
            out.append((budget, b, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="main")
    ap.add_argument("--quant", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="16,32,64,128")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gap", type=int, default=32)
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (ignored by CUDA kernels)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda; the gate must pass on the same device")
    ap.add_argument("--kv-quant", default="none", choices=["none", "cq4p_wht3"],
                    help="KV Cache Engine codecs: ChannelQuant CQ-4+ keys, CQ-3-rot (WHT INT3) values")
    ap.add_argument("--key-group", type=int, default=128, help="key group size G (their default 128)")
    ap.add_argument("--outlier-k", type=int, default=2, help="FP16 outlier key channels per (layer, head)")
    ap.add_argument("--calib-n", type=int, default=64, help="held-out stories for the outlier mask")
    ap.add_argument("--evict-layers", default="all", choices=["all", "global"],
                    help="layers the budget applies to; 'global' leaves local layers at their 256-token window")
    ap.add_argument("--part", default="0/1", help="k/n: run the k-th of n balanced shares of the grid "
                    "(part 0 also writes the full-cache lines); parts can run as parallel processes")
    ap.add_argument("--no-full", action="store_true",
                    help="skip the full-cache lines (when another file of the same setting has them)")
    ap.add_argument("--limit-batches", type=int, default=0, help="debug: stop after this many batches")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 matmuls fp32 on Ampere+ GPUs
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    budgets = [int(b) for b in args.budgets.split(",")]
    tok, model = load_model(device=args.device)
    if args.quant == "int8":
        quantize_weights_int8(model)
    engine = NeoEngine(model)
    stories = load_stories(tok, n=args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kv_tag = "none"
    if args.kv_quant != "none":
        # Static outlier mask calibrated offline, as the KVCE's ROM is: on stories disjoint from
        # the evaluation set, with the unquantized engine.
        calib = load_stories(tok, n=args.calib_n, seed=args.seed + 1, exclude={s["story"] for s in stories})
        outlier = calibrate_outliers(engine, calib, k=args.outlier_k)
        engine.kvq = KVQuant(outlier, group=args.key_group)
        kv_tag = f"{args.kv_quant}_G{args.key_group}"
        with open(os.path.splitext(args.out)[0] + ".outliers.json", "w") as f:
            json.dump({"calibration_stories": [s["story"] for s in calib],
                       "outlier_channels": outlier.nonzero().tolist()}, f)
    stories.sort(key=lambda s: len(s["ids"]))  # length-bucketed batches waste little on padding
    groups = split(grid(args.grid, budgets), *map(int, args.part.split("/")))
    with_full = args.part.split("/")[0] == "0" and not args.no_full

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                d = json.loads(line)
                done.add((d["family"], d["granularity"], d["batch"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = dict(model=MODEL, quant=args.quant, grid=args.grid, seed=args.seed, gap=args.gap,
                evict_layers=args.evict_layers, kv=kv_tag)
    evict = None if args.evict_layers == "all" else [not local for local in engine.local]

    batches = [stories[i:i + args.batch] for i in range(0, len(stories), args.batch)]
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    t_start = time.time()
    for bi, batch in enumerate(batches):
        todo = [g for g in [("full", "head", None)] * with_full + groups if (g[0], g[1], bi) not in done]
        if not todo:
            continue
        T = max(len(s["ids"]) for s in batch)
        ids = torch.full((len(batch), T), 50256)
        for b, s in enumerate(batch):
            ids[b, :len(s["ids"])] = torch.tensor(s["ids"])
        lens = torch.tensor([len(s["ids"]) for s in batch])
        ids, lens = ids.to(engine.device), lens.to(engine.device)
        recall = [recall_targets(s["ids"], tok, args.gap) for s in batch]
        ref = reference(engine, ids, lens)
        lines = []
        for family, gran, rows in todo:
            t0 = time.time()
            if family == "full":
                for budget, b, s in full_lines(budgets, ids, lens, recall, ref):
                    lines.append(dict(meta, family="full", granularity="head", batch=bi, policy="full",
                                      budget=budget, sink=0, recent=0, param=None, protect="none",
                                      story=batch[b]["story"], len=int(lens[b]), **s))
            else:
                cfg, story, acc, extra = run_group(engine, family, gran, rows, ids, lens, recall, ref, evict)
                for r in range(len(cfg)):
                    row = rows[int(cfg[r])]
                    b = int(story[r])
                    s = {k: (v[r].item() if v.dtype.is_floating_point else int(v[r])) for k, v in acc.items()}
                    s["n"] = int(s["n"])
                    s["rn"], s["rn_name"] = int(s["rn"]), int(s["rn_name"])
                    s.update({k: int(v[r]) for k, v in extra.items()})
                    lines.append(dict(meta, family=family, granularity=gran, batch=bi, story=batch[b]["story"],
                                      len=int(lens[b]), **row, **s))
            # Write per group so a crash loses at most one group of one batch.
            with open(args.out, "a") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            lines = []
            print(f"batch {bi + 1}/{len(batches)} T={T} {family}/{gran}: {time.time() - t0:.1f}s "
                  f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
