"""Generation with an evicted cache, for task benchmarks (LongBench, RULER-style, MATH-500).

Two protocols:

  Prompt compression (LongBench, RULER). The prompt is read once with the full cache, tracking
  what the policies need (attention mass per slot, attention rows of the last `window` prompt
  tokens). Every policy then picks its kept slots from that same cache (`keep_mask`), the kept
  slots are packed into a fresh batched cache, one row per policy (`gather_rows`), and the
  answer is decoded for all rows at once with no further eviction. This is the setup of SnapKV,
  PyramidKV and most LongBench tables; the first answer token comes from the full prefill.

  Decode-time eviction (MATH-500). The prompt is short and the output long, so the cache is
  evicted every `interval` generated tokens. Each batch row is a different (policy, capacity)
  on the same prompt; rows stay in one masked cache (evicted slots hidden, not removed) and are
  evicted one at a time through a view of their row (`evict_rows`).

Sampling uses common random numbers: every row draws its token with the same uniform number
at each step, so rows whose next-token distributions agree pick the same token, and the
comparison between policies is paired.
"""

import torch

from .policies import evict, select_keep


@torch.inference_mode()
def prefill(engine, ids, extra, window=32, chunk=1024, track=True):
    """Run ids ([1, T]) into a new cache in chunks of `chunk` tokens, with room for `extra` more.
    Returns (logits of the last position [1, vocab], cache)."""
    cache = engine.new_cache(ids.shape[0], ids.shape[1] + extra, window=window if track else 0)
    logits = None
    for s in range(0, ids.shape[1], chunk):
        logits = engine.forward(ids[:, s:s + chunk], cache, track=track)
    return logits, cache


def capacity_for(budget, n, sink=4):
    """Slots per head for a budget given as a fraction of n (<= 1) or as a slot count (> 1)."""
    cap = int(round(budget * n)) if budget <= 1 else int(budget)
    return max(sink + 1, min(cap, n))


@torch.inference_mode()
def keep_mask(cache, policy, capacity, **kw):
    """The keep bits `policy` would leave on `cache` ([layers, batch, kv, length]), without
    changing the cache. Returns (mask, stats)."""
    mask, stats = select_keep(cache, policy, capacity, **kw)
    return (cache.keep[..., :cache.length].clone() if mask is None else mask), stats


@torch.inference_mode()
def gather_rows(engine, cache, masks, extra):
    """New cache with one row per mask, holding only that mask's kept slots (in order), plus
    room for `extra` tokens. Every mask must keep the same number of slots in every head."""
    n = cache.length
    live = int(masks[0].sum(-1).max())
    out = engine.new_cache(len(masks), live + extra)
    dim = cache.k.shape[-1]
    for b, m in enumerate(masks):
        m = m[:, 0]  # [layers, kv, n]
        if int(m.sum(-1).min()) != live or int(m.sum(-1).max()) != live:
            raise ValueError("gather_rows needs the same number of kept slots in every head and row")
        idx = m.float().topk(live, dim=-1).indices.sort(dim=-1).values  # [layers, kv, live]
        take = idx[..., None].expand(-1, -1, -1, dim)
        out.k[:, b, :, :live] = cache.k[:, 0, :, :n].gather(2, take)
        out.v[:, b, :, :live] = cache.v[:, 0, :, :n].gather(2, take)
    out.length = live
    out.pos = cache.pos
    return out


@torch.inference_mode()
def evict_rows(cache, configs):
    """Evict each row of a masked cache with its own config: configs[b] is a dict with
    'policy', 'capacity' and evict() keyword arguments. Returns per-row stats (None = nothing
    evicted)."""
    stats = []
    for b, cfg in enumerate(configs):
        cfg = dict(cfg)
        policy, capacity = cfg.pop("policy"), cfg.pop("capacity")
        st = evict(cache.row(b), policy, capacity, **cfg)
        if st is not None:
            cache.has_evicted = True
        stats.append(st)
    return stats


def _pick(logits, temperature, top_p, u):
    """Greedy when temperature == 0; otherwise top-p sampling where every row uses the same
    uniform u (common random numbers)."""
    if not temperature:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    sp = sp.masked_fill(cum - sp > top_p, 0.0)  # keep the smallest prefix with mass >= top_p
    cum = sp.cumsum(dim=-1)
    target = u * cum[:, -1:]
    j = torch.searchsorted(cum, target).clamp_max(sp.shape[-1] - 1)
    return si.gather(1, j).squeeze(1)


@torch.inference_mode()
def generate(engine, cache, logits, max_new, eos_ids, temperature=0.0, top_p=1.0, seed=0,
             track=False, on_step=None):
    """Decode up to max_new tokens for every row of `cache`, starting from `logits`
    ([batch, vocab], or [1, vocab] shared by all rows). Rows that emit an EOS id stop counting;
    the loop ends when all have. on_step(cache, step) runs after each new token is written.
    Returns a list of token-id lists (EOS excluded) and whether each row hit max_new."""
    batch = cache.k.shape[1]
    if logits.shape[0] != batch:
        logits = logits.expand(batch, -1)
    eos = torch.tensor(sorted(set(eos_ids)), device=logits.device)
    gen = torch.Generator(device=logits.device).manual_seed(seed)
    done = [False] * batch
    out = [[] for _ in range(batch)]
    for step in range(max_new):
        u = torch.rand(1, generator=gen, device=logits.device)
        nxt = _pick(logits, temperature, top_p, u)
        # One device-to-host copy per step, not one per row.
        for b, (t, e) in enumerate(zip(nxt.tolist(), torch.isin(nxt, eos).tolist())):
            if not done[b] and not e:
                out[b].append(t)
            done[b] = done[b] or e
        if all(done) or step == max_new - 1:
            break
        logits = engine.forward(nxt[:, None], cache, track=track)
        if on_step is not None:
            on_step(cache, step)
    return out, [not d for d in done]
