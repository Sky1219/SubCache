"""Correctness gate: the manual engine must reproduce HF greedy generation token for token.

    python scripts/gate.py                       # fp16 on cuda:0, 128 new tokens
    python scripts/gate.py --device cpu --dtype float32 --max-new 24 --lengths 64

Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES before running on edmonds.
A mismatch is reported with the top-2 logit gap at the first differing step, so an fp16
near-tie can be told apart from a real bug.
"""

import argparse
import os
import sys
import time

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from subcache.engine import Engine, load_model  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "World War Two ended in",
    "The best selling fruit in the worlds is",
    "Bananas go well with",
    " I love",
    "def fibonacci(n):\n    ",
]
FILLER = (
    "The history of the printing press begins in the fifteenth century, when movable type "
    "changed how books were made, copied, and shared across Europe and beyond. "
) * 200
NEAR_TIE = 0.05


def eos_ids(model, tok):
    """Every id generate() treats as end-of-sequence (some models list more than one)."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def hf_generate(model, tok, ids, max_new):
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=max_new,
            min_new_tokens=max_new,
            repetition_penalty=1.0,  # some models ship a penalty in their generation config
            pad_token_id=tok.eos_token_id,
        )
    return out[:, ids.shape[1]:]


def top2_gap(model, tok, ids, ref_new, step):
    """Top-2 logit gap from a full HF forward on the reference prefix up to `step`."""
    prefix = torch.cat([ids, ref_new[:step].unsqueeze(0)], dim=1)
    with torch.inference_mode():
        logits = model(prefix).logits[0, -1].float()
    logits[eos_ids(model, tok)] = float("-inf")
    vals = torch.topk(logits, 2).values
    return (vals[0] - vals[1]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--lengths", type=int, nargs="*", default=[128, 512, 1024, 1900])
    ap.add_argument("--batch", type=int, default=4, help="rows in the batched-decode check; 1 skips it")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model, args.device, dtype)
    engine = Engine(model)
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
    print(f"model={args.model} dtype={args.dtype} device={gpu} "
          f"torch={torch.__version__} transformers={transformers.__version__}")

    cases = [(repr(p), tok(p, return_tensors="pt")["input_ids"]) for p in PROMPTS]
    filler_ids = tok(FILLER, return_tensors="pt")["input_ids"]
    cases += [(f"filler L={n}", filler_ids[:, :n]) for n in args.lengths]

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    # Logit-level check, teacher forced: one prefill, then compare every position to HF.
    ids = cases[2][1].to(args.device)
    with torch.inference_mode():
        hf_logits = model(ids).logits[0].float()
    mine = engine.forward(ids, engine.new_cache(1, ids.shape[1]), all_logits=True)[0].float()
    if not (torch.isfinite(hf_logits).all() and torch.isfinite(mine).all()):
        # Two paths that both produce NaN "match" token for token, so this must fail loudly.
        print(f"NON-FINITE LOGITS: reference has {int((~torch.isfinite(hf_logits)).sum())}, engine has "
              f"{int((~torch.isfinite(mine)).sum())} bad values. The model is not usable in {args.dtype}; gate void.")
        sys.exit(2)
    print(f"prefill logits max|diff| vs HF: {(mine - hf_logits).abs().max().item():.4f} "
          f"(argmax agree at {(mine.argmax(-1) == hf_logits.argmax(-1)).float().mean().item():.0%} of positions)")

    eos = eos_ids(model, tok)
    exact = ties = bugs = 0
    for name, ids in cases:
        ids = ids.to(args.device)
        ref_new = hf_generate(model, tok, ids, args.max_new)[0]
        with torch.inference_mode():
            if not torch.isfinite(model(ids).logits[0, -1]).all():
                print(f"  {name[:44]:44} NON-FINITE reference logits; gate void")
                sys.exit(2)
        sync()
        t0 = time.perf_counter()
        new, cache = engine.greedy(ids, args.max_new, suppress_token=eos)
        sync()
        ms = 1000 * (time.perf_counter() - t0) / args.max_new
        new = new[0]
        if torch.equal(new, ref_new):
            exact += 1
            status = "MATCH"
        else:
            step = (new != ref_new).nonzero()[0].item()
            gap = top2_gap(model, tok, ids, ref_new, step)
            if gap < NEAR_TIE:
                ties += 1
                status = f"near-tie at step {step} (top-2 gap {gap:.4f})"
            else:
                bugs += 1
                status = f"MISMATCH at step {step} (top-2 gap {gap:.4f})"
        print(f"  {name[:44]:44} T={ids.shape[1]:5d}  {ms:6.1f} ms/tok  {status}")

    # Batched decode: several equal-length prompts in one batch must match HF on the same batch.
    total = len(cases)
    if args.batch > 1:
        for length in (64, 512):
            ids = torch.stack([filler_ids[0, i * 97:i * 97 + length] for i in range(args.batch)]).to(args.device)
            ref_new = hf_generate(model, tok, ids, args.max_new)
            new, _ = engine.greedy(ids, args.max_new, suppress_token=eos)
            status = []
            for b in range(args.batch):
                if torch.equal(new[b], ref_new[b]):
                    exact += 1
                    status.append("MATCH")
                    continue
                step = (new[b] != ref_new[b]).nonzero()[0].item()
                gap = top2_gap(model, tok, ids[b:b + 1], ref_new[b], step)
                if gap < NEAR_TIE:
                    ties += 1
                    status.append(f"near-tie at step {step} (gap {gap:.4f})")
                else:
                    bugs += 1
                    status.append(f"MISMATCH at step {step} (gap {gap:.4f})")
            print(f"  batch of {args.batch}, L={length}: " + "; ".join(status))
        total += 2 * args.batch

    print(f"{exact} exact, {ties} near-tie, {bugs} real mismatch out of {total}")
    sys.exit(1 if bugs else 0)


if __name__ == "__main__":
    main()
