"""Task benchmarks for the eviction policies: python scripts/run_bench.py --bench longbench|ruler|math

  longbench  LongBench tasks, prompt compression (see subcache/longctx.py), budgets as a fraction
             of the prompt, greedy decoding, LongBench's metric per task.
  ruler      RULER-style needle and variable-tracking tasks at the given context lengths, same
             protocol as longbench, RULER's string-match-all score.
  math       MATH-500 with chain-of-thought from a reasoning model, eviction every `interval`
             generated tokens, budgets in slots, top-p sampling with common random numbers.

Writes one JSON line per example (all policies and budgets together). Each record carries the
settings that affect its scores and a short hash of them ("config"). A rerun skips examples
already in --out with the same config, so a stopped run resumes where it left off; records
from other settings are kept but not reused. Summarize with scripts/analyze_bench.py.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from subcache import benchmarks as B  # noqa: E402
from subcache.engine import Engine, load_model  # noqa: E402
from subcache.longctx import capacity_for, evict_rows, gather_rows, generate, keep_mask, prefill  # noqa: E402


def eos_ids(model, tok):
    ids = model.generation_config.eos_token_id
    ids = list(ids) if isinstance(ids, (list, tuple)) else [ids]
    return sorted({i for i in ids + [tok.eos_token_id] if i is not None})


def policy_kw(policy, args, compress=False):
    """evict() arguments for a policy; compress=True adds the prompt-compression options."""
    kw = dict(recent_frac=args.recent_frac, sink=args.sink)
    if policy == "wfl":
        kw.update(mix=args.mix, wfl_mode="prune", prune_step=args.prune_step)
        if compress:
            kw.update(pool=args.pool, prune_rounds=args.prune_rounds, sim_bytes=args.sim_bytes)
    if policy == "rkv":
        kw.update(lam=args.lam, rkv_chunk=args.rkv_chunk)
    return kw


# Arguments that only choose which examples run, or how memory is chunked, not what they score.
NOT_RESULT_SETTINGS = ("out", "device", "tasks", "limit", "lengths", "levels", "sim_bytes")


def result_settings(args, eos):
    """The settings a record's scores depend on, and a short hash of them."""
    s = {k: v for k, v in vars(args).items() if k not in NOT_RESULT_SETTINGS}
    s["eos"] = eos
    return s, hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]


def read_records(path):
    """Records in path. A last line cut off by a killed run is dropped from the file, so the
    next append starts on a fresh line."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines[-1].strip():  # no newline at the end: the last write never finished
        print(f"{path}: dropping an incomplete last line ({len(lines[-1])} bytes)", flush=True)
        with open(path, "r+b") as f:
            f.truncate(len(data) - len(lines[-1]))
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def done_keys(path, config):
    """(task, length, idx) of the records in path made with `config`."""
    keys, other = set(), 0
    for r in read_records(path):
        if r.get("config") == config:
            keys.add((r["task"], r.get("length"), r["idx"]))
        else:
            other += 1
    if other:
        print(f"{path}: {other} records come from other settings; they are kept but not reused", flush=True)
    return keys


def compress_and_decode(engine, tok, ids, max_gen, eos, args):
    """Prompt-compression protocol for one example. Returns a list of result dicts."""
    dev = engine.device
    ids_t = torch.tensor([ids], device=dev)
    t0 = time.perf_counter()
    logits, cache = prefill(engine, ids_t, extra=max_gen + 8, window=args.window, chunk=args.chunk)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    n = cache.length
    results = []
    for budget in args.budgets:
        cap = capacity_for(budget, n, args.sink)
        masks, sel = [], []
        for policy in args.policies:
            m, st = keep_mask(cache, policy, cap, **policy_kw(policy, args, compress=True))
            masks.append(m)
            sel.append(st["select_s"] if st else 0.0)
        rows = gather_rows(engine, cache, masks, extra=max_gen + 8)
        t1 = time.perf_counter()
        toks, trunc = generate(engine, rows, logits, max_gen, eos)
        dec_s = time.perf_counter() - t1
        del rows
        for policy, t, tr, s in zip(args.policies, toks, trunc, sel):
            results.append({"policy": policy, "budget": budget, "capacity": cap, "tokens": t, "truncated": tr,
                            "select_s": s, "decode_s": dec_s})
    t1 = time.perf_counter()
    toks, trunc = generate(engine, cache, logits, max_gen, eos)
    results.append({"policy": "full", "budget": 1.0, "capacity": n, "tokens": toks[0], "truncated": trunc[0],
                    "select_s": 0.0, "decode_s": time.perf_counter() - t1})
    del cache
    torch.cuda.empty_cache()
    for r in results:
        r["pred"] = tok.decode(r.pop("tokens"), skip_special_tokens=True)
    return results, n, prefill_s


def run_math(engine, tok, args, eos, out, settings, config, done):
    rows = B.load_math500(limit=args.limit, seed=args.seed, levels=args.levels)
    configs = [{"policy": "full", "capacity": 0}] + [
        dict(policy=p, capacity=int(b), **policy_kw(p, args)) for b in args.budgets for p in args.policies]
    batch = len(configs)
    for i, row in enumerate(rows):
        if ("math500", None, row["unique_id"]) in done:
            continue
        ids = B.math_ids(tok, row)
        t0 = time.perf_counter()
        ids_t = torch.tensor([ids], device=engine.device).expand(batch, -1)
        logits, cache = prefill(engine, ids_t, extra=args.max_new + 8, window=args.window, chunk=len(ids))
        sel = [0.0] * batch
        n_evict = [0] * batch

        def on_step(c, step):
            if (step + 1) % args.interval == 0:
                for b, st in enumerate(evict_rows(c, configs)):
                    if st:
                        sel[b] += st["select_s"]
                        n_evict[b] += 1

        toks, trunc = generate(engine, cache, logits, args.max_new, eos, temperature=args.temperature,
                               top_p=args.top_p, seed=args.seed * 100003 + i, track=True, on_step=on_step)
        wall = time.perf_counter() - t0
        del cache
        torch.cuda.empty_cache()
        results = []
        for cfg, t, tr, s, ne in zip(configs, toks, trunc, sel, n_evict):
            text = tok.decode(t, skip_special_tokens=True)
            results.append({"policy": cfg["policy"], "budget": cfg["capacity"] or 1.0, "capacity": cfg["capacity"] or None,
                            "gen_len": len(t), "truncated": tr, "boxed": B.last_boxed(text),
                            "correct": B.math_correct(text, row["answer"]), "tail": text[-300:],
                            "select_s": s, "evictions": ne})
        rec = {"task": "math500", "length": None, "idx": row["unique_id"], "level": row.get("level"),
               "subject": row.get("subject"), "answer": row["answer"], "n_tokens": len(ids), "wall_s": wall,
               "results": results, "config": config, "settings": settings}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        acc = {r["policy"] + "@" + str(r["budget"]): int(r["correct"]) for r in results}
        print(f"[{i + 1}/{len(rows)}] {row['unique_id']} wall {wall:.0f}s maxlen {max(r['gen_len'] for r in results)} "
              f"full={acc['full@1.0']} correct={sum(acc.values())}/{len(acc)}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["longbench", "ruler", "math"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (math: problems)")
    ap.add_argument("--max-len", type=int, default=32000, help="longbench: prompt tokens after middle truncation")
    ap.add_argument("--lengths", type=int, nargs="*", default=[8192, 16384], help="ruler: context lengths")
    ap.add_argument("--budgets", type=float, nargs="*", default=[0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--policies", nargs="*", default=["recency", "h2o", "rkv", "wfl"])
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--recent-frac", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk (tokens)")
    ap.add_argument("--mix", type=float, default=1.0)
    ap.add_argument("--prune-step", type=int, default=8)
    ap.add_argument("--prune-rounds", type=int, default=64)
    ap.add_argument("--pool", type=float, default=2.0, help="wfl candidate pool, a multiple (>= 1) of the slots to fill")
    ap.add_argument("--sim-bytes", type=float, default=6e9, help="memory for wfl similarity matrices at once")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--rkv-chunk", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=8192, help="math: generation cap")
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="math: only these difficulty levels (1-5)")
    ap.add_argument("--interval", type=int, default=64, help="math: evict every this many generated tokens")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.pool and args.pool < 1:
        ap.error("--pool must be at least 1 (or 0 for no pool): a smaller pool keeps fewer slots than the budget")

    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model, args.device, dtype)
    engine = Engine(model)
    eos = eos_ids(model, tok)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    settings, config = result_settings(args, eos)
    settings["gpu"] = torch.cuda.get_device_name(engine.device)  # recorded, not part of the hash
    print(json.dumps(dict(vars(args), eos=eos, gpu=settings["gpu"], config=config)), flush=True)

    done = done_keys(args.out, config)
    with open(args.out, "a") as out:
        if args.bench == "math":
            run_math(engine, tok, args, eos, out, settings, config, done)
            return
        if args.bench == "longbench":
            jobs = [(task, None, i, row) for task in (args.tasks or list(B.LONGBENCH_PROMPT))
                    for i, row in enumerate(B.load_longbench(task, limit=args.limit))]
        else:
            from subcache.evals import load_text

            maker = B.RulerMaker(tok, load_text("wikitext-train"))
            jobs = [(task, length, i, None) for length in args.lengths for task in (args.tasks or B.RULER_TASKS)
                    for i in range(args.limit or 50)]
        for j, (task, length, i, row) in enumerate(jobs):
            if (task, length, i) in done:
                continue
            if args.bench == "longbench":
                ids, max_gen = B.longbench_ids(tok, task, row, args.max_len), B.LONGBENCH_MAXGEN[task]
                answers = row["answers"]
                extra = {"length_words": row.get("length")}
            else:
                ex = maker.make(task, length, seed=args.seed * 1_000_003 + length * 101 + i * 7919 + B.RULER_TASKS.index(task) * 13)
                ids, max_gen, answers = ex["ids"], B.RULER_MAXGEN[task], ex["answers"]
                extra = {"depths": ex["depths"]}
            t0 = time.perf_counter()
            results, n, prefill_s = compress_and_decode(engine, tok, ids, max_gen, eos, args)
            for r in results:
                r["score"] = (B.longbench_score(task, r["pred"], answers) if args.bench == "longbench"
                              else B.ruler_score(r["pred"], answers))
            rec = {"task": task, "length": length, "idx": i, "n_tokens": n, "answers": answers, "prefill_s": prefill_s,
                   "wall_s": time.perf_counter() - t0, "results": results, **extra, "config": config,
                   "settings": settings}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            sc = {r["policy"] + "@" + str(r["budget"]): round(r["score"]) for r in results}
            print(f"[{j + 1}/{len(jobs)}] {task} {length or ''} #{i} n={n} prefill {prefill_s:.1f}s "
                  f"wall {rec['wall_s']:.1f}s full={sc['full@1.0']} {sc}", flush=True)


if __name__ == "__main__":
    main()
