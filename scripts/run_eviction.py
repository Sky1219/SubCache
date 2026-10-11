"""Compare KV eviction policies at matched budgets: streaming perplexity and needle retrieval.

    python scripts/run_eviction.py                                  # default sweep on cuda:0, fp16
    python scripts/run_eviction.py --policies full h2o com --budgets 0.5 0.25 --n-seqs 4
    python scripts/run_eviction.py --device cpu --dtype float32 --text builtin \
        --seq-len 192 --n-seqs 1 --interval 32 --window 8 --needle-trials 0     # smoke test

Budget b means each (layer, kv_head) may hold b * seq_len live slots; every policy also keeps
the newest recent_frac of that and the first --sink slots. One JSON line per (policy, budget) is appended to --out with
the full settings, so every number can be traced to how it was produced.
"""

import argparse
import json
import os
import sys
import time

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from subcache.engine import Engine, load_model  # noqa: E402
from subcache.evals import load_text, needle, perplexity  # noqa: E402
from subcache.policies import POLICIES, RKV_BETA, RKV_POOL, RKV_T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--text", default="wikitext", help="'wikitext', 'builtin', or a path to a .txt file")
    ap.add_argument("--data", default=None, help="a .pt file from scripts/prepare_agent_data.py (overrides --text)")
    ap.add_argument("--resume", action="store_true", help="skip (policy, budget) rows already in --out with this tag")
    ap.add_argument("--wfl-mode", default="select", choices=["select", "prune"])
    ap.add_argument("--prune-step", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.1, help="rkv: weight of importance against redundancy")
    ap.add_argument("--policies", nargs="+", default=["full", "recency", "random", "h2o", "gcmi", "com", "flqmi"],
                    choices=POLICIES)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--interval", type=int, default=64, help="evict after every N tokens")
    ap.add_argument("--window", type=int, default=32, help="number of queries kept as the query set")
    ap.add_argument("--window-mode", default="recent", choices=["recent", "reservoir"],
                    help="'recent' = newest queries; 'reservoir' = uniform sample of all queries so far")
    ap.add_argument("--mix", type=float, default=0.5, help="wfl: weight of the redundancy term (0 = h2o)")
    ap.add_argument("--sim-power", type=float, default=1.0, help="wfl: sharpening exponent on the similarity")
    ap.add_argument("--recent-frac", type=float, default=0.5, help="share of the budget reserved for the newest slots")
    ap.add_argument("--sink", type=int, default=4, help="first N slots are never evicted (attention sinks)")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--psi", default="sqrt", choices=["sqrt", "log"])
    ap.add_argument("--eps", type=float, default=None, help="stochastic greedy epsilon; omit for exact greedy")
    ap.add_argument("--value-aware", action="store_true", help="weight the kernel by value-vector norms")
    ap.add_argument("--compact", action="store_true", help="physically remove evicted slots instead of masking them")
    ap.add_argument("--needle-trials", type=int, default=4, help="trials per depth; 0 skips the needle test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/eviction.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model, args.device, dtype)
    engine = Engine(model)
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"

    masks = None
    if args.data:
        data = torch.load(args.data)
        if data["ids"].shape[0] < args.n_seqs or data["ids"].shape[1] < args.seq_len:
            raise SystemExit(f"{args.data} holds {tuple(data['ids'].shape)}, need {args.n_seqs} x {args.seq_len}")
        sequences = [data["ids"][i:i + 1, :args.seq_len].to(args.device) for i in range(args.n_seqs)]
        masks = [data["mask"][i, :args.seq_len] for i in range(args.n_seqs)]
        haystack, args.needle_trials = None, 0
    else:
        ids = tok(load_text(args.text), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        need = args.n_seqs * (args.seq_len - 1)
        if len(ids) < need + args.seq_len:
            raise SystemExit(f"text too short: {len(ids)} tokens, need {need + args.seq_len}")
        bos = torch.tensor([tok.bos_token_id])
        sequences = [
            torch.cat([bos, ids[i * (args.seq_len - 1):(i + 1) * (args.seq_len - 1)]]).unsqueeze(0).to(args.device)
            for i in range(args.n_seqs)
        ]
        haystack = ids[need:]  # needle contexts come from text the perplexity runs did not use

    settings = {k: v for k, v in vars(args).items() if k not in ("policies", "budgets", "out", "resume")}
    settings.update(gpu=gpu, torch=str(torch.__version__), transformers=transformers.__version__,
                    rkv_threshold=RKV_T, rkv_beta=RKV_BETA, rkv_pool=RKV_POOL)
    cfg = model.config
    bytes_per_token = (cfg.num_hidden_layers * cfg.num_key_value_heads * 2 * engine.head_dim
                       * torch.finfo(dtype).bits // 8)
    token_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "tokens")
    os.makedirs(token_dir, exist_ok=True)
    print(json.dumps(settings))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 'full' is run at every budget too: it never evicts, but ppl_post is averaged over the
    # positions past that budget's capacity, so each budget needs its own reference number.
    runs = [(p, b) for b in args.budgets for p in args.policies]
    if args.resume and os.path.exists(args.out):
        done = set()
        for line in open(args.out):
            row = json.loads(line)
            if row["settings"]["tag"] == args.tag:
                done.add((row["policy"], row["budget"]))
        runs = [r for r in runs if r not in done]
        print(f"resume: {len(done)} rows already present, {len(runs)} to run")

    header = f"{'policy':8} {'budget':>6} {'ppl':>8} {'ppl_post':>9} {'needle':>7} {'retained':>9} {'sel s/evict':>11} {'wall s':>7}"
    print(header)
    for policy, budget in runs:
        capacity = max(2, int(round(budget * args.seq_len)))
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        kw = dict(interval=args.interval, window=args.window, recent_frac=args.recent_frac, sink=args.sink, eta=args.eta,
                  psi=args.psi, eps=args.eps, value_aware=args.value_aware, window_mode=args.window_mode,
                  mix=args.mix, sim_power=args.sim_power, compact=args.compact, wfl_mode=args.wfl_mode,
                  prune_step=args.prune_step, lam=args.lam, generator=gen)
        t0 = time.perf_counter()
        res = perplexity(engine, sequences, policy, capacity, masks=masks, **kw)
        if args.needle_trials:
            res.update(needle(engine, tok, haystack, policy, capacity, context_len=args.seq_len - 96,
                              trials=args.needle_trials, seed=args.seed, **kw))
        res["wall_s"] = time.perf_counter() - t0
        token_nll = res.pop("token_nll", None)
        if token_nll is not None:
            name = f"{os.path.splitext(os.path.basename(args.out))[0]}__{args.tag}__{policy}__{budget:g}.pt"
            torch.save(token_nll, os.path.join(token_dir, name))
            res["token_nll_file"] = os.path.join("tokens", name)
        live = args.seq_len if policy == "full" else min(args.seq_len, capacity + args.interval)
        res["cache_bytes_per_seq"] = live * bytes_per_token  # peak KV memory one sequence needs
        res["loadavg"] = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        row = {"policy": policy, "budget": budget, "capacity": capacity, **res, "settings": settings}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

        per_evict = res["select_s_total"] / res["evictions"] if res.get("evictions") else float("nan")
        print(f"{policy:8} {budget:6.3f} {res['ppl']:8.3f} {res.get('ppl_post', float('nan')):9.3f} "
              f"{res.get('needle_acc', float('nan')):7.2f} {res.get('retained_mean', float('nan')):9.3f} "
              f"{per_evict:11.4f} {res['wall_s']:7.1f}")
    print(f"appended {len(runs)} rows to {args.out}")


if __name__ == "__main__":
    main()
