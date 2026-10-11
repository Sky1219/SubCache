"""Speed and memory of batched decoding under KV eviction with physical compaction.

    python scripts/bench_speed.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --context 8192

For every (policy, budget, batch size): stream `context` tokens per sequence into the cache in
chunks, evicting and compacting after each chunk, then decode `new` tokens greedily for the
whole batch, evicting again whenever a sequence has grown by `interval` tokens. Reported:

    decode ms/step   wall time per decode step, selection and compaction included
    tok/s            batch * new / decode time: the throughput number
    select %         share of decode time spent choosing what to evict
    cache GB         memory held by the K and V tensors
    peak GB          peak GPU memory, weights included

Batch sizes are tried in increasing order and a (policy, budget) stops at the first one that
runs out of memory, so the largest batch listed is the largest that fits. policy 'full' never
evicts and is the reference.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from subcache.engine import Engine, load_model  # noqa: E402
from subcache.evals import load_text  # noqa: E402
from subcache.policies import NEEDS_WINDOW, POLICIES, evict  # noqa: E402


def run_one(engine, ids, policy, capacity, args):
    """ids [batch, context]. Returns a dict of measurements."""
    batch, context = ids.shape
    evicting = policy != "full"
    slots = (capacity + args.interval + 8) if evicting else (context + args.new + 8)
    window = args.window if policy in NEEDS_WINDOW else 0
    kw = dict(recent_frac=args.recent_frac, sink=args.sink, mix=args.mix, eps=args.eps, compact=True,
              wfl_mode=args.wfl_mode, prune_step=args.prune_step)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache = engine.new_cache(batch, slots, window=window)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, context, args.interval):
        logits = engine.forward(ids[:, start:start + args.interval], cache, track=evicting)
        if evicting:
            evict(cache, policy, capacity, **kw)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    select_s = compact_s = 0.0
    evictions = 0
    t0 = time.perf_counter()
    for _ in range(args.new):
        next_id = logits.argmax(dim=-1)
        logits = engine.forward(next_id.unsqueeze(1), cache, track=evicting)
        if evicting and cache.length >= capacity + args.interval:
            st = evict(cache, policy, capacity, **kw)
            select_s += st["select_s"]
            compact_s += st["compact_s"]
            evictions += 1
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    return {
        "prefill_tok_s": batch * context / prefill_s,
        "decode_ms_step": 1000 * decode_s / args.new,
        "decode_tok_s": batch * args.new / decode_s,
        "select_frac": select_s / decode_s,
        "compact_frac": compact_s / decode_s,
        "decode_evictions": evictions,
        "live_slots": cache.length,
        "cache_gb": cache.allocated_bytes / 2**30,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text", default="wikitext")
    ap.add_argument("--context", type=int, default=8192, help="tokens per sequence before decoding")
    ap.add_argument("--new", type=int, default=256, help="tokens decoded per sequence")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    ap.add_argument("--policies", nargs="+", default=["full", "h2o"], choices=POLICIES)
    ap.add_argument("--interval", type=int, default=128, help="evict after every N tokens")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--recent-frac", type=float, default=0.5)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--mix", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--wfl-mode", default="select", choices=["select", "prune"])
    ap.add_argument("--prune-step", type=int, default=8)
    ap.add_argument("--data", default=None, help="a .pt file from scripts/prepare_agent_data.py (overrides --text)")
    ap.add_argument("--out", default="results/speed.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.device, torch.float16)
    engine = Engine(model)
    if args.data:
        ids = torch.load(args.data)["ids"][:, :args.context]
        ids = ids.repeat(max(args.batches) // ids.shape[0] + 1, 1)[:max(args.batches)].to(args.device)
    else:
        ids = tok(load_text(args.text), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        need = max(args.batches) * args.context
        ids = ids.repeat(need // len(ids) + 1)[:need].view(max(args.batches), args.context).to(args.device)

    try:
        load = os.getloadavg()[0]
        others = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.split()
    except Exception:
        load, others = None, []
    settings = {k: v for k, v in vars(args).items() if k not in ("policies", "budgets", "batches", "out")}
    settings.update(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                    transformers=transformers.__version__, loadavg=load, gpu_processes=len(others))
    print(json.dumps(settings))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    run_one(engine, ids[:1, :512], "h2o", 128, args)  # warm-up
    runs = [("full", 1.0)] if "full" in args.policies else []
    runs += [(p, b) for b in args.budgets for p in args.policies if p != "full"]
    print(f"{'policy':8} {'budget':>6} {'batch':>5} {'decode ms/step':>14} {'tok/s':>8} {'select %':>8} "
          f"{'cache GB':>8} {'peak GB':>8} {'prefill tok/s':>13}")
    for policy, budget in runs:
        capacity = int(round(budget * args.context))
        for batch in args.batches:
            try:
                res = run_one(engine, ids[:batch], policy, capacity, args)
            except torch.OutOfMemoryError:
                print(f"{policy:8} {budget:6.2f} {batch:5d}   out of memory")
                res = {"oom": True}
            row = {"policy": policy, "budget": budget, "capacity": capacity, "batch": batch, **res, "settings": settings}
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if res.get("oom"):
                torch.cuda.empty_cache()
                break
            print(f"{policy:8} {budget:6.2f} {batch:5d} {res['decode_ms_step']:14.1f} {res['decode_tok_s']:8.0f} "
                  f"{100 * res['select_frac']:8.1f} {res['cache_gb']:8.2f} {res['peak_gb']:8.2f} {res['prefill_tok_s']:13.0f}")


if __name__ == "__main__":
    main()
