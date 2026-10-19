"""The TIU team's H2O simulation (token-importance-unit, analysis/h2o_analysis.py) run on this model
and data, so it can be compared position for position with true streaming eviction.

    python -m reef.sim --out reef/results/sim.jsonl [--device cuda]

Their simulation is one teacher-forced forward pass. In every layer and head, each query row i
keeps a recent window plus the top tokens by attention mass accumulated over all queries so far,
computed from the attention the layer actually produced in that pass, then attends over that set
only. Nothing is evicted permanently: every row re-selects from scratch, so a token dropped for
one query can come back for the next.

Kept the same as theirs: per-head selection, scores from A.cumsum over query rows (row i
included, a one-row lookahead), renormalized attention over the kept set. Adapted to compare
with reef/run.py at equal budget: query i sees itself, the 4 sinks, the newest round(B/2) earlier
tokens, and the top B - 4 - round(B/2) others (B + 1 tokens, as in streaming), and in local
layers only tokens inside the 256-token window are candidates (their code targets models with
no local layers). Variant `h2o_sim_nolook` ranks with rows < i only, which removes the lookahead.

Writes the run.py per-story JSONL schema (policy h2o_sim / h2o_sim_nolook), so analyze.py pairs
it with the full-cache lines and streaming H2O from the main sweep.
"""

import argparse
import json
import os

import torch
import transformers.models.gpt_neo.modeling_gpt_neo as neo

from reef.data import load_stories, recall_targets
from reef.engine import MODEL, load_model
from reef.run import RECENT_FRAC, SINK

_CFG = {}


def keep_mask(allowed, acc, budget, sink, recent):
    """[..., T, T] bool: keys query row i attends to. allowed = what the model's own mask permits
    (causal, local window); acc = per-row importance. Row i keeps itself, the `sink` first tokens,
    the `recent` tokens before it, and the top budget - sink - recent other allowed tokens by acc."""
    T = acc.shape[-1]
    i = torch.arange(T, device=acc.device)[:, None]
    j = torch.arange(T, device=acc.device)[None, :]
    protected = allowed & ((j < sink) | (j >= i - recent))
    eligible = allowed & ~protected
    m = budget - sink - recent
    score = acc.masked_fill(~eligible, float("-inf"))
    top = score.topk(min(m, T), dim=-1)
    heavy = torch.zeros(score.shape, dtype=torch.bool, device=acc.device)
    heavy.scatter_(-1, top.indices, top.values > float("-inf"))
    return protected | heavy


def sim_attn(self, query, key, value, attention_mask=None, *args, **kwargs):
    """Drop-in for GPTNeoSelfAttention._attn implementing the simulation (batch size 1)."""
    query, key = query.float(), key.float()
    w = torch.matmul(query, key.transpose(-1, -2))  # [1, H, T, T], no 1/sqrt(d) in GPT-Neo
    T = w.shape[-1]
    allowed = self.bias[:, :, :T, :T]  # causal, plus the local window in local layers
    if _CFG:
        A = torch.softmax(w.masked_fill(~allowed, float("-inf")), dim=-1)
        acc = A.cumsum(dim=-2)  # acc[..., i, j] = sum over rows q <= i
        if not _CFG["lookahead"]:
            acc = torch.cat([torch.zeros_like(acc[..., :1, :]), acc[..., :-1, :]], dim=-2)
        allowed = keep_mask(allowed, acc, _CFG["budget"], _CFG["sink"], _CFG["recent"])
    w = w.masked_fill(~allowed, torch.finfo(w.dtype).min)
    if attention_mask is not None:
        w = w + attention_mask
    p = torch.softmax(w, dim=-1).to(value.dtype)
    return torch.matmul(p, value), p


@torch.inference_mode()
def story_logits(model, ids, cfg):
    """Log-probs [T, V] for one story under the simulation (cfg) or the full cache (cfg=None)."""
    _CFG.clear()
    if cfg:
        _CFG.update(cfg)
    orig = neo.GPTNeoSelfAttention._attn
    neo.GPTNeoSelfAttention._attn = sim_attn
    try:
        out = model(torch.tensor([ids], device=model.device)).logits[0].float()
    finally:
        neo.GPTNeoSelfAttention._attn = orig
        _CFG.clear()
    return torch.log_softmax(out, -1)


def sums(logp, ref, ids, budget, recall):
    """run.py's per-story sums over positions t >= budget + 1 (predicting ids[t + 1])."""
    T = len(ids)
    t = torch.arange(budget + 1, T - 1, device=logp.device)
    nxt = torch.tensor(ids, device=logp.device)[t + 1]
    lp, rl = logp[t], ref[t]
    nll = -lp.gather(1, nxt[:, None]).squeeze(1).double()
    nll_full = -rl.gather(1, nxt[:, None]).squeeze(1).double()
    kl = (rl.exp() * (rl - lp)).sum(-1).double()
    top1, rtop1 = lp.argmax(-1), rl.argmax(-1)
    correct, correct_full = (top1 == nxt).double(), (rtop1 == nxt).double()
    pos = {int(x): k for k, x in enumerate(t.tolist())}
    rec = torch.zeros(len(t), dtype=torch.double, device=logp.device)
    nam = torch.zeros_like(rec)
    for j, (kind, _) in recall.items():
        if j - 1 in pos:
            rec[pos[j - 1]] = 1
            nam[pos[j - 1]] = kind == "name"
    s = {"n": len(t), "nll": nll.sum(), "nll_full": nll_full.sum(), "kl": kl.sum(),
         "agree": (top1 == rtop1).double().sum(), "correct": correct.sum(), "correct_full": correct_full.sum(),
         "rn": rec.sum(), "r_nll": (rec * nll).sum(), "r_nll_full": (rec * nll_full).sum(),
         "r_correct": (rec * correct).sum(), "r_correct_full": (rec * correct_full).sum(),
         "rn_name": nam.sum(), "r_nll_name": (nam * nll).sum(), "r_nll_full_name": (nam * nll_full).sum(),
         "r_correct_name": (nam * correct).sum(), "r_correct_full_name": (nam * correct_full).sum()}
    out = {k: (v if isinstance(v, int) else v.item()) for k, v in s.items()}
    for k in ("rn", "rn_name"):
        out[k] = int(out[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="16,32,64,128")
    ap.add_argument("--gap", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok, model = load_model(device=args.device)
    stories = load_stories(tok, n=args.n, seed=args.seed)
    budgets = [int(b) for b in args.budgets.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["story"] for line in f}
    meta = dict(model=MODEL, quant="fp32", grid="sim", seed=args.seed, gap=args.gap, evict_layers="all",
                family="sim", granularity="head", protect="sink+recent", batch=0)
    for k, s in enumerate(stories):
        if s["story"] in done:
            continue
        ids = s["ids"]
        recall = recall_targets(ids, tok, args.gap)
        ref = story_logits(model, ids, None)
        lines = []
        for budget in budgets:
            recent = int(round(RECENT_FRAC * budget))
            for lookahead, name in ((True, "h2o_sim"), (False, "h2o_sim_nolook")):
                cfg = dict(budget=budget, sink=SINK, recent=recent, lookahead=lookahead)
                logp = story_logits(model, ids, cfg)
                lines.append(dict(meta, policy=name, budget=budget, sink=SINK, recent=recent, param=None,
                                  story=s["story"], len=len(ids), **sums(logp, ref, ids, budget, recall)))
        with open(args.out, "a") as f:  # one story at a time, so a re-run resumes
            for line in lines:
                f.write(json.dumps(line) + "\n")
        if k % 50 == 0:
            print(f"story {k + 1}/{len(stories)}", flush=True)


if __name__ == "__main__":
    main()
