"""Streaming eviction policies over a SlotCache, batched over (layer, row, head) as tensors.

One `Streaming` object drives one policy family over many batch rows at once; each row has its
own budget, protection and family parameter (W or alpha), so a whole sweep of one family runs as
one batch. Per decoded token:

    slot  = cache.claim()            new token takes the lowest empty slot (engine.step does this)
    policy.on_load(slot)             its score starts at 0
    attn  = engine.step(...)         the token attends to every visible slot, itself included
    policy.update(attn)              scores absorb this step's attention
    policy.evict(cache)              local layers drop slots that left the window; then, wherever
                                     a (layer, row, head) holds more than `budget` tokens, the
                                     lowest-scoring unprotected slot is evicted (one per step)

So a query attends over at most budget + 1 tokens: `budget` older ones plus itself.

Families (score per cached slot; the argmin is evicted)
    full    never evicts (budget = capacity)
    window  score = position: evict the oldest unprotected token. With sinks this is StreamingLLM.
    h2o     score += attention received (float). Zhang et al. 2023, and the TIU's statistic.
    tiu     the TIU datapath (LonghornSilicon/tt-longhorn-tiu, token_importance_unit.sv):
            weight = round(attention * scale) clamped to 8 bits, score = min(score + weight, 255),
            evict the lowest score, ties to the lowest physical slot index. The RTL does not say
            how attention mass maps onto the 8-bit weight port, so `scale` is a per-row parameter
            (255 = full scale, attention 1.0 -> 255). The core has no protection input; here it
            ranks only the unprotected slots, as if protected tokens were held outside it.
    rw      score = attention received from the last W queries (W per row). The streaming analog
            of SubCache's question-aware window score.
    ema     score = alpha * score + attention (alpha per row).

Protection (per row): the first `sink` positions and the newest `recent` positions are never
candidates. Granularity: 'head' evicts per (layer, head); 'layer' ranks slots by the score summed
over the layer's heads (tiu: by one 8-bit score fed the head-mean attention) and evicts the same
token from every head, so all heads of a layer share one keep set.
"""

import torch

FAMILIES = ("full", "window", "h2o", "tiu", "rw", "ema")
TIU_MAX = 255  # SCORE_WIDTH = WEIGHT_WIDTH = 8 in the RTL


class Streaming:
    def __init__(self, family, cache, local, window, budget, sink, recent, param=None, granularity="head",
                 evict_layers=None):
        """budget, sink, recent, param: per-row sequences (param = W for rw, alpha for ema, weight
        scale for tiu, default 255).
        local: per-layer bools; window: local attention window (positions).
        evict_layers: per-layer bools, the layers the budget applies to (default: all). A layer left
        out keeps everything it can still attend to; only local layers may be left out, since a
        local layer never holds more than `window` tokens."""
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        if granularity not in ("head", "layer"):
            raise ValueError(f"unknown granularity {granularity!r}")
        L, R, H, C = cache.pos.shape
        dev = cache.pos.device
        self.family, self.granularity = family, granularity
        self.local = torch.tensor(local, device=dev)
        if evict_layers is None:
            evict_layers = [True] * L
        self.evict_layers = torch.tensor(evict_layers, device=dev)
        if not bool(self.evict_layers.all()) and family != "full":
            if bool((~self.evict_layers & ~self.local).any()):
                raise ValueError("a global layer without eviction needs a full-length cache")
            if C < window + 1:
                raise ValueError(f"capacity {C} cannot hold an un-evicted local layer (window {window})")
        self.window = window
        self.budget = torch.as_tensor(budget, device=dev, dtype=torch.long)
        self.sink = torch.as_tensor(sink, device=dev, dtype=torch.long)
        self.recent = torch.as_tensor(recent, device=dev, dtype=torch.long)
        if bool((self.sink + self.recent >= self.budget).any()):
            raise ValueError("sink + recent must leave at least one evictable slot under the budget")
        if bool((self.budget + 1 > C).any()):
            raise ValueError(f"capacity {C} too small for a budget of {int(self.budget.max())} (+1 incoming token)")
        heads = H if granularity == "head" else 1
        if family == "tiu":
            self.score = torch.zeros(L, R, heads, C, device=dev, dtype=torch.int32)
            scale = [TIU_MAX] * R if param is None else param
            self.scale = torch.as_tensor(scale, device=dev, dtype=torch.float32)[None, :, None, None]
            # Per row: eviction decisions, those where every candidate was saturated at 255, and
            # those where the minimum score was shared by more than one candidate (a tie).
            self.n_decisions = torch.zeros(R, dtype=torch.long, device=dev)
            self.n_saturated = torch.zeros(R, dtype=torch.long, device=dev)
            self.n_tied = torch.zeros(R, dtype=torch.long, device=dev)
        elif family in ("h2o", "rw", "ema"):
            self.score = torch.zeros(L, R, H, C, device=dev, dtype=torch.float32)
        if family == "rw":
            self.W = torch.as_tensor(param, device=dev, dtype=torch.long)
            self.wmax = int(self.W.max())
            self.buf = torch.zeros(self.wmax, L, R, H, C, device=dev, dtype=torch.float32)  # ring of attention rows
        if family == "ema":
            self.alpha = torch.as_tensor(param, device=dev, dtype=torch.float32)[None, :, None, None]

    def on_load(self, slot):
        """A new token took `slot` ([L, R, H]) in every head: its score starts at zero."""
        if self.family in ("full", "window"):
            return
        s = slot if self.score.shape[2] == slot.shape[2] else slot[:, :, :1]
        self.score.scatter_(-1, s[..., None], 0)
        if self.family == "rw":
            idx = slot[None, ..., None].expand(self.wmax, -1, -1, -1, 1)
            self.buf.scatter_(-1, idx, 0.0)

    def update(self, attns, t):
        """attns: list over layers of [R, H, C] attention weights of the query at position t."""
        if self.family in ("full", "window"):
            return
        a = torch.stack(attns)  # [L, R, H, C]
        if self.family == "h2o":
            self.score += a
        elif self.family == "ema":
            self.score.mul_(self.alpha).add_(a)
        elif self.family == "rw":
            # Running sum of the last W rows: add this row, subtract the row W steps back (still in
            # the ring because W <= wmax), then store this row over the oldest one.
            rows = torch.arange(a.shape[1], device=a.device)
            old = (t - self.W) % self.wmax
            leaving = self.buf[old, :, rows].permute(1, 0, 2, 3)  # [L, R, H, C]
            leaving = leaving * (t - self.W >= 0).to(a.dtype)[None, :, None, None]
            self.score += a - leaving
            self.buf[t % self.wmax] = a
        elif self.family == "tiu":
            if self.granularity == "layer":
                a = a.mean(2, keepdim=True)
            w = torch.round(a * self.scale).clamp_(0, TIU_MAX).to(torch.int32)
            self.score = torch.clamp_max(self.score + w, TIU_MAX)

    def ranking(self, pos):
        """[L, R, heads, C] score whose argmin over candidates is evicted."""
        if self.family == "window":
            r = pos.float()
        elif self.family == "tiu":
            r = self.score.float()
        else:
            r = self.score
        if self.granularity == "layer" and r.shape[2] > 1:
            r = r.sum(2, keepdim=True)
        return r

    def evict(self, cache):
        """Drop slots that left a local layer's window, then evict down to the budget. Returns the
        number of slots evicted by the policy (window drops not counted)."""
        pos = cache.pos
        t_last = cache.t - 1  # position of the token just processed
        # Local layers: the next query (position t_last + 1) sees positions > t_last + 1 - window.
        stale = (pos >= 0) & (pos <= t_last + 1 - self.window) & self.local[:, None, None, None]
        pos.masked_fill_(stale, -1)
        if self.family == "full":
            return 0
        occupied = pos >= 0
        need = (occupied.sum(-1) > self.budget[None, :, None]) & self.evict_layers[:, None, None]  # [L, R, H]
        if not bool(need.any()):
            return 0
        sink = self.sink[None, :, None, None]
        recent = self.recent[None, :, None, None]
        protected = (pos < sink) | (pos > t_last - recent)
        cand = occupied & ~protected
        if self.granularity == "layer":
            cand = cand[:, :, :1]
            need = need[:, :, :1]
        r = self.ranking(pos).masked_fill(~cand, float("inf"))
        victim = r.argmin(-1)  # first minimum: lowest slot index on ties (the TIU's tie rule)
        if self.family == "tiu":
            low = r.gather(-1, victim[..., None])
            dims = (0, 2)
            self.n_decisions += need.sum(dims)
            self.n_saturated += (need & (low.squeeze(-1) >= TIU_MAX)).sum(dims)
            self.n_tied += (need & ((r == low).sum(-1) > 1)).sum(dims)
        if not bool(cand.gather(-1, victim[..., None]).squeeze(-1)[need].all()):
            raise RuntimeError("over budget with no evictable slot")
        victim = victim.expand(pos.shape[:3])
        need = need.expand(pos.shape[:3])
        cur = pos.gather(-1, victim[..., None])
        pos.scatter_(-1, victim[..., None], torch.where(need[..., None], torch.full_like(cur, -1), cur))
        return int(need.sum())
