"""KV eviction policies, run per (layer, kv_head) and batched over all of them as tensors.

Setup, following PRISM (Kothawade et al., arXiv 2103.00128):
    ground set V  = live cached slots of one (layer, kv_head)
    query set Q   = the most recent queries (cache window) of the query heads sharing that kv head
    kernel S[j,i] = attention weight query j gave slot i          -> shape [problems, Q, slots]
    subset A      = slots to keep, |A| = capacity

Policies
    full     keep everything (reference)
    recency  keep the newest `capacity` slots (sliding window)
    random   keep the protected recent slots + a uniform random choice of the rest
    h2o      top-k by attention mass received over the whole history (Zhang et al., H2O)
    gcmi     top-k by attention mass received from the window: PRISM's graph-cut MI, which is
             modular, so it is H2O restricted to the window
    com      PRISM concave-over-modular: eta*sum_{i in A} psi(sum_j S_ji) + sum_j psi(sum_{i in A} S_ji)
    flqmi    PRISM facility-location MI: sum_j max_{i in A} S_ji + eta*sum_{i in A} max_j S_ji
    wfl      redundancy-aware H2O: (1-mix)*sum_{i in A} w_i + mix*sum_{i in V} w_i max_{j in A} sim(i,j)
             with w = H2O's score and sim a key-and-value similarity between cached slots.
             mix=0 is exactly h2o; mix>0 prefers that every important slot has a similar slot kept.
             wfl_mode='select' builds the kept set greedily; 'prune' starts from everything and
             greedily removes the cheapest slots, which is far faster when only a few must go.
    rkv      R-KV-style (Cai et al., NeurIPS 2025): lam*importance - (1-lam)*redundancy, with
             importance = attention from the window queries (max over the query heads sharing a
             kv head, max-pooled over neighbouring slots) and redundancy = softmax of each key's
             mean cosine similarity to the other keys. One selection per layer, shared by its
             heads. The paper's threshold, recency count and pooling width were not available,
             so RKV_T, RKV_BETA and RKV_POOL below are this code's choices, and the protected
             recent/sink slots follow this harness, not the paper's 8 observation tokens.

Every policy protects the first `sink` slots and the newest `recent_frac * capacity` slots, and
chooses the rest, so they differ only in how the middle slots are chosen. (The first tokens act
as attention sinks; dropping them wrecks the model regardless of policy, see StreamingLLM. With
them protected, `recency` is the StreamingLLM baseline.) Eviction is permanent: an evicted slot
is never a candidate again.
"""

import math
import time

import torch

POLICIES = ("full", "recency", "random", "h2o", "gcmi", "com", "flqmi", "wfl", "rkv")
NEEDS_WINDOW = ("gcmi", "com", "flqmi", "rkv")
RKV_T, RKV_BETA, RKV_POOL = 0.5, 1, 7


def _psi(name):
    if name == "sqrt":
        return torch.sqrt
    if name == "log":
        return lambda x: torch.log1p(x / 0.01)
    raise ValueError(f"unknown psi {name!r}")


class COM:
    """eta * sum_{i in A} psi(sum_j S_ji)  +  sum_j psi(sum_{i in A} S_ji).

    The second term has diminishing returns in each query's retained attention mass; the first
    is modular in A (a per-slot relevance bonus). Monotone submodular in A for S >= 0.
    """

    def __init__(self, eta=1.0, psi="sqrt"):
        self.eta = eta
        self.psi = _psi(psi)

    def start(self, S, held):
        self.S = S
        self.mass = (S * held[:, None, :]).sum(-1)  # [P, Q] mass each query keeps via `held`
        self.modular = self.eta * self.psi(S.sum(1))  # [P, n]

    def gains(self, cols=None):
        S, modular = self.S, self.modular
        if cols is not None:
            S = S.gather(2, cols[:, None, :].expand(-1, S.shape[1], -1))
            modular = modular.gather(1, cols)
        return modular + (self.psi(self.mass[:, :, None] + S) - self.psi(self.mass)[:, :, None]).sum(1)

    def add(self, rows, pick):
        self.mass += self.S[rows, :, pick]


class FLQMI:
    """sum_j max_{i in A} S_ji  +  eta * sum_{i in A} max_j S_ji."""

    def __init__(self, eta=1.0, psi=None):
        self.eta = eta

    def start(self, S, held):
        self.S = S
        self.best = S.masked_fill(~held[:, None, :], 0.0).amax(-1)  # [P, Q]
        self.modular = self.eta * S.amax(1)  # [P, n]

    def gains(self, cols=None):
        S, modular = self.S, self.modular
        if cols is not None:
            S = S.gather(2, cols[:, None, :].expand(-1, S.shape[1], -1))
            modular = modular.gather(1, cols)
        return modular + (S - self.best[:, :, None]).clamp_min(0.0).sum(1)

    def add(self, rows, pick):
        self.best = torch.maximum(self.best, self.S[rows, :, pick])


OBJECTIVES = {"com": COM, "flqmi": FLQMI}


def greedy_select(objective, S, cand, held, m, eps=None, generator=None):
    """Pick m slots per problem from `cand` to add to `held`, greedily maximizing `objective`.

    S [P, Q, n]; cand, held [P, n] bool. eps=None is exact greedy (every candidate scored each
    step). With eps, stochastic greedy (Mirzasoleiman et al. 2015): each step scores a random
    sample of (n_cand/m)*log(1/eps) candidates, for a (1 - 1/e - eps) guarantee in expectation.
    Returns selected [P, n] bool.
    """
    P = S.shape[0]
    rows = torch.arange(P, device=S.device)
    n_cand = int(cand.sum(1).min())
    m = min(m, n_cand)
    selected = torch.zeros_like(cand)
    if m <= 0:
        return selected
    avail = cand.clone()
    objective.start(S, held)
    sample = None if eps is None else max(1, math.ceil(n_cand / m * math.log(1.0 / eps)))
    for step in range(m):
        if sample is None or sample >= n_cand - step:
            pick = objective.gains().masked_fill(~avail, float("-inf")).argmax(1)
        else:
            cols = torch.multinomial(avail.float(), sample, replacement=False, generator=generator)
            pick = cols.gather(1, objective.gains(cols).argmax(1, keepdim=True)).squeeze(1)
        selected[rows, pick] = True
        avail[rows, pick] = False
        objective.add(rows, pick)
    return selected


def slot_similarity(keys, values, power=1.0):
    """sim [P, n, n] in [0, 1]: two slots are interchangeable only if both their keys and their
    values point the same way. Vectors are centred per problem first (raw keys share a large
    common component, which would make everything look alike), and `power` sharpens the result.
    """
    def cosine(x):
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.normalize(x, dim=-1)
        return torch.matmul(x, x.transpose(1, 2)).clamp_min(0.0)

    sim = (cosine(keys) * cosine(values)) ** power
    eye = torch.eye(sim.shape[1], device=sim.device, dtype=torch.bool)
    return sim.masked_fill(eye, 1.0)


def wfl_select(sim, w, cand, held, m, mix=0.5, eps=None, generator=None):
    """Greedy for (1-mix)*sum_{j in A} w_j + mix*sum_i w_i max_{j in A} sim[i, j], A = held + picks.

    sim [P, n, n], w [P, n] >= 0, cand / held [P, n] bool. Monotone submodular (a modular term
    plus weighted facility location). Returns selected [P, n] bool.
    """
    P, n = w.shape
    rows = torch.arange(P, device=w.device)
    n_cand = int(cand.sum(1).min())
    m = min(m, n_cand)
    selected = torch.zeros_like(cand)
    if m <= 0:
        return selected
    avail = cand.clone()
    cover = sim.masked_fill(~held[:, None, :], 0.0).amax(-1)  # [P, n] how well slot i is covered
    sample = None if eps is None else max(1, math.ceil(n_cand / m * math.log(1.0 / eps)))
    for step in range(m):
        if sample is None or sample >= n_cand - step:
            gain = (w[:, :, None] * (sim - cover[:, :, None]).clamp_min(0.0)).sum(1)  # [P, n] per column j
            gain = (mix * gain + (1.0 - mix) * w).masked_fill(~avail, float("-inf"))
            pick = gain.argmax(1)
        else:
            cols = torch.multinomial(avail.float(), sample, replacement=False, generator=generator)
            sub = sim.gather(2, cols[:, None, :].expand(-1, n, -1))  # [P, n, sample]
            gain = (w[:, :, None] * (sub - cover[:, :, None]).clamp_min(0.0)).sum(1)
            gain = mix * gain + (1.0 - mix) * w.gather(1, cols)
            pick = cols.gather(1, gain.argmax(1, keepdim=True)).squeeze(1)
        selected[rows, pick] = True
        avail[rows, pick] = False
        cover = torch.maximum(cover, sim[rows, :, pick])
    return selected


def wfl_prune(sim, w, cand, held, m, mix=0.5, step=8, rounds=None):
    """Same objective as wfl_select, optimized from the other side: start with every candidate
    kept and repeatedly remove the `step` slots whose removal loses the least, until m remain.

    Removing slot j loses (1-mix)*w_j, plus mix * w_i * (best_i - second_i) for every slot i
    whose most similar kept slot is j. Cost is one top-2 over [P, n, n] per round, and the
    number of rounds is (slots to remove) / step, so it suits streaming eviction where each
    call removes few slots. No approximation guarantee (greedy removal), unlike wfl_select.

    rounds caps the number of rounds by removing more per round when many slots must go, as when
    a whole prompt is compressed at once: step becomes max(step, ceil(to_remove / rounds)).
    """
    kept = cand | held
    n_remove = int(cand.sum(1).min()) - m
    if rounds:
        step = max(step, math.ceil(n_remove / rounds))
    removed = 0
    while removed < n_remove:
        # Best and second-best kept neighbour of every slot, as two max passes (much faster
        # than topk(2) on long rows). With ties, best - second is 0 whichever index wins.
        masked = sim.masked_fill(~kept[:, None, :], -1.0)
        best, arg = masked.max(dim=-1)
        second = masked.scatter_(-1, arg[..., None], -1.0).amax(dim=-1).clamp_min(0.0)
        del masked
        loss = torch.zeros_like(w).scatter_add_(1, arg, w * (best - second).clamp_min(0.0))
        loss = (mix * loss + (1.0 - mix) * w).masked_fill(~(kept & cand), float("inf"))
        k = min(step, n_remove - removed)
        kept.scatter_(1, loss.topk(k, dim=1, largest=False).indices, False)
        removed += k
    return kept & cand


def rkv_scores(S, keys, groups, lam=0.1):
    """R-KV-style score per slot. S [P, Q, n] window attention, keys [P, n, d]. Returns [P, n]."""
    P, Q, n = S.shape
    attn = S.view(P, groups, Q // groups, n).amax(1)  # max over the query heads of a group
    attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-9)
    attn = torch.nn.functional.max_pool1d(attn, RKV_POOL, stride=1, padding=RKV_POOL // 2)
    importance = attn.mean(1)  # [P, n]

    kn = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(kn, kn.transpose(1, 2))  # sim[p, j, i]
    sim.diagonal(dim1=1, dim2=2).zero_()
    # For each slot i, forgive its similarity to the RKV_BETA most recent slots that resemble
    # it, so the newest copy of a repeated pattern is not itself marked redundant.
    order = torch.arange(n, device=sim.device, dtype=sim.dtype)[None, :, None]
    for _ in range(RKV_BETA):
        latest = torch.where(sim > RKV_T, order, torch.full_like(sim, -1.0)).argmax(dim=1, keepdim=True)
        has = sim.gather(1, latest) > RKV_T
        sim.scatter_(1, latest, sim.gather(1, latest).masked_fill(has, 0.0))
    redundancy = torch.softmax(sim.mean(1), dim=-1)  # [P, n]
    return lam * importance - (1.0 - lam) * redundancy


def top_m(score, cand, m):
    """Keep the m highest-scoring candidates per problem. Returns [P, n] bool."""
    m = min(m, score.shape[1])
    if m <= 0:
        return torch.zeros_like(cand)
    idx = score.masked_fill(~cand, float("-inf")).topk(m, dim=1).indices
    return torch.zeros_like(cand).scatter_(1, idx, True) & cand


def window_kernel(cache):
    """S [P, Q, n] float32 from the cache's attention window; P = layers*batch*kv_heads,
    Q = (query heads per kv head) * (rows in the window), n = cache.length."""
    n, fill = cache.length, cache.win_fill
    n_layers, batch, n_kv, _ = cache.keep.shape
    rows = cache.window[:, :, :, :fill, :n].float()  # [layers, batch, heads, fill, n]
    return rows.reshape(n_layers * batch * n_kv, -1, n)


@torch.inference_mode()
def evict(cache, policy, capacity, compact=False, **kw):
    """Evict down to `capacity` live slots per (layer, kv_head). Returns a stats dict, or None
    if nothing needed evicting. With compact=True the evicted slots are physically removed
    (see KVCache.compact); otherwise they are only masked. Selection options: select_keep."""
    if compact and cache.is_row_view:  # checked before any keep bit changes
        raise RuntimeError("cannot compact a row view (KVCache.row); evict rows with compact=False")
    new_keep, stats = select_keep(cache, policy, capacity, **kw)
    if new_keep is None:
        return None
    cache.keep[..., :cache.length] = new_keep
    cache.has_evicted = True
    if compact:
        t0 = time.perf_counter()
        cache.compact()  # physically drop the evicted slots
        if cache.keep.device.type == "cuda":
            torch.cuda.synchronize()
        stats["compact_s"] = time.perf_counter() - t0
    return stats


@torch.inference_mode()
def select_keep(cache, policy, capacity, recent_frac=0.5, sink=4, eta=1.0, psi="sqrt", eps=None,
                value_aware=False, mix=0.5, sim_power=1.0, wfl_mode="select", prune_step=8, lam=0.1,
                chunk=16, generator=None, pool=None, prune_rounds=None, rkv_chunk=None, sim_bytes=None):
    """The keep bits ([layers, batch, kv_heads, length]) that evicting down to `capacity` live
    slots per (layer, kv_head) would leave, and stats, without changing the cache. Returns
    (None, None) if nothing needs evicting.

    Stats: selection time, and (when the cache has a window) the fraction of each recent
    query's attention mass that survives, as mean / std / min across heads.

    For compressing a long prompt in one call (thousands of slots to drop):
      pool         wfl only: candidates are limited to the pool*m highest-H2O-score ones, and the
                   facility-location ground set to those plus the protected slots, so the
                   similarity matrix is that size instead of [n, n]. The rest are evicted.
      prune_rounds wfl prune only: at most this many removal rounds (see wfl_prune).
      sim_bytes    wfl only: pick `chunk` so the similarity matrices handled at once take about
                   this much memory.
      rkv_chunk    rkv only: heads scored at a time (default: a whole layer). Same result, less
                   memory: the per-head scores are averaged over the layer either way.
    """
    if pool and pool < 1:
        # A pool smaller than the slots to fill would keep fewer than `capacity` slots.
        raise ValueError(f"pool must be at least 1 (or None), got {pool}")
    n = cache.length
    if policy == "full" or n <= capacity:
        return None, None
    if policy in NEEDS_WINDOW and (cache.window is None or cache.win_fill == 0):
        raise ValueError(f"policy {policy!r} needs a cache created with window > 0 and track=True")
    device = cache.keep.device
    n_layers, batch, n_kv, _ = cache.keep.shape
    P = n_layers * batch * n_kv

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    keep = cache.keep[..., :n].reshape(P, n).clone()  # clone: reshape can return a view
    sink = min(sink, capacity)
    recent = min(capacity - sink, int(round(recent_frac * capacity)))
    slot = torch.arange(n, device=device)
    protected = keep & ((slot >= n - recent) | (slot < sink))
    cand = keep & ~protected
    m = capacity - recent - sink

    S = window_kernel(cache) if cache.window is not None and cache.win_fill else None
    kernel = S
    if value_aware and S is not None:
        # Weight each slot by the size of its value vector: what the slot contributes to an
        # attention output is weight * value, not weight alone.
        vnorm = cache.v[..., :n, :].float().norm(dim=-1).reshape(P, n)
        kernel = S * (vnorm / vnorm.mean(dim=1, keepdim=True).clamp_min(1e-9))[:, None, :]

    if policy == "recency":
        selected = top_m(torch.arange(n, device=device, dtype=torch.float32).expand(P, n), cand, m)
    elif policy == "random":
        selected = top_m(torch.rand(P, n, device=device, generator=generator), cand, m)
    elif policy == "h2o":
        selected = top_m(cache.score[..., :n].reshape(P, n), cand, m)
    elif policy == "gcmi":
        selected = top_m(kernel.sum(1), cand, m)
    elif policy in ("wfl", "rkv"):
        # Work on the live slots only: every head has the same number of them, so a similarity
        # matrix is [problems, live, live] instead of [problems, n, n]. Problems are handled a
        # few at a time (`chunk`) to bound memory; rkv needs whole layers together.
        live = int(keep.sum(1).min())
        if int(keep.sum(1).max()) != live:
            raise ValueError(f"{policy} expects the same number of live slots in every head")
        idx = keep.float().topk(live, dim=1).indices.sort(dim=1).values  # [P, live] slot ids
        if policy == "wfl" and pool:
            n_prot = int(protected.sum(1).max())
            size_pool = min(live, n_prot + min(int(pool * m), live - n_prot))
            rank = cache.score[..., :n].reshape(P, n).masked_fill(~cand, float("-inf"))
            rank = rank.masked_fill(protected, float("inf"))
            idx = rank.topk(size_pool, dim=1).indices.sort(dim=1).values
        if policy == "wfl" and sim_bytes:
            chunk = max(1, int(sim_bytes / (12 * idx.shape[1] ** 2)))
        dim = cache.k.shape[-1]
        all_k = cache.k[..., :n, :].reshape(P, n, dim)
        all_v = cache.v[..., :n, :].reshape(P, n, dim)
        all_w = cache.score[..., :n].reshape(P, n)
        picked = torch.zeros_like(idx, dtype=torch.bool)
        groups = kernel.shape[1] // cache.win_fill if policy == "rkv" else None
        size = n_kv if policy == "rkv" else chunk  # P is ordered layer, batch, head
        for a in range(0, P, size):
            sl = slice(a, min(a + size, P))
            c_cand = cand[sl].gather(1, idx[sl])
            if policy == "rkv":
                step_h = rkv_chunk or (sl.stop - sl.start)
                z = []
                for h in range(sl.start, sl.stop, step_h):
                    ids = idx[h:h + step_h]
                    win = kernel[h:h + step_h].gather(2, ids[:, None, :].expand(-1, kernel.shape[1], -1))
                    keys = all_k[h:h + step_h].gather(1, ids[:, :, None].expand(-1, -1, dim)).float()
                    z.append(rkv_scores(win, keys, groups, lam))
                z = torch.cat(z)
                picked[sl] = top_m(z.mean(0, keepdim=True).expand_as(z), c_cand, m)
                continue
            take = idx[sl, :, None].expand(-1, -1, dim)
            keys = all_k[sl].gather(1, take).float()
            values = all_v[sl].gather(1, take).float()
            c_held = protected[sl].gather(1, idx[sl])
            w = all_w[sl].gather(1, idx[sl])
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)
            sim = slot_similarity(keys, values, sim_power)
            if wfl_mode == "prune":
                picked[sl] = wfl_prune(sim, w, c_cand, c_held, m, mix=mix, step=prune_step, rounds=prune_rounds)
            else:
                picked[sl] = wfl_select(sim, w, c_cand, c_held, m, mix=mix, eps=eps, generator=generator)
        selected = torch.zeros_like(keep).scatter_(1, idx, picked)
    elif policy in OBJECTIVES:
        selected = greedy_select(OBJECTIVES[policy](eta=eta, psi=psi), kernel, cand, protected, m,
                                 eps=eps, generator=generator)
    else:
        raise ValueError(f"unknown policy {policy!r}")

    new_keep = protected | selected

    if device.type == "cuda":
        torch.cuda.synchronize()
    stats = {"length": n, "evicted": int(keep.sum() - new_keep.sum()), "select_s": time.perf_counter() - t0}
    if S is not None:
        before = (S * keep[:, None, :]).sum(-1)
        after = (S * new_keep[:, None, :]).sum(-1)
        retained = (after / before.clamp_min(1e-9)).mean(1)  # [P] one number per head
        stats.update(retained_mean=retained.mean().item(), retained_std=retained.std().item(),
                     retained_min=retained.min().item())
    return new_keep.view(n_layers, batch, n_kv, n), stats
