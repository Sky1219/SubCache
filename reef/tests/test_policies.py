"""Model-free checks of the streaming policies: run with `python reef/tests/test_policies.py`.

Random attention rows are pushed through a SlotCache + Streaming policy and compared with direct
re-implementations: a scalar port of the TIU RTL (token_importance_unit.sv) driven the way the
TIU's H2OScheduler wrapper drives it, and per-token score histories for h2o / rw / ema.
"""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from reef.engine import SlotCache  # noqa: E402
from reef.policies import Streaming  # noqa: E402


class RTLTiu:
    """Scalar port of token_importance_unit.sv: sat_add per slot, LOAD zeroes, EVICT is a serial
    strict-less argmin seeded with the first slot it scans.

    The RTL seeds with slot 0 (or the SCORE_MAX sentinel if slot 0 is invalid), so if slot 0 were
    excluded and every other slot sat at 255 it would return slot 0. The core has no protection
    input, so this port models a core that only sees the unprotected slots (in slot order)."""

    def __init__(self, n):
        self.score, self.valid = [0] * n, [False] * n

    def load(self, k):
        self.score[k], self.valid[k] = 0, True

    def acc(self, k, w):
        self.score[k] = min(self.score[k] + (w & 255), 255)

    def evict(self, allowed):
        """`allowed` stands in for the control wrapper's protection: protected slots are skipped."""
        seen = [i for i in range(len(self.valid)) if self.valid[i] and allowed[i]]
        min_s, min_i = self.score[seen[0]], seen[0]
        for i in seen:
            if self.score[i] < min_s:
                min_s, min_i = self.score[i], i
        self.valid[min_i] = False
        return min_i


def drive(family, budget, sink, recent, param, steps, granularity="head", L=2, H=3, seed=0, local=None,
          window=10**9):
    """Random attention through one row. Returns (per-step attention tensors, pos snapshots
    after eviction, victims as (step, layer, head, slot, position))."""
    g = torch.Generator().manual_seed(seed)
    cache = SlotCache(L, 1, H, budget + 1, 4, "cpu", torch.float32)
    pol = Streaming(family, cache, local or [False] * L, window, [budget], [sink], [recent],
                    None if param is None else [param], granularity)
    attns, snaps, victims = [], [], []
    for t in range(steps):
        slot = cache.claim()
        cache.t = t + 1
        pol.on_load(slot)
        live = cache.pos >= 0
        logits = torch.randn(L, 1, H, budget + 1, generator=g) * 2.0
        a = torch.softmax(logits.masked_fill(~live, float("-inf")), -1)
        attns.append(a)
        pol.update(list(a), t)
        before = cache.pos.clone()
        pol.evict(cache)
        for l, r, h, s in ((before >= 0) & (cache.pos < 0)).nonzero().tolist():
            victims.append((t, l, h, s, int(before[l, r, h, s])))
        snaps.append(cache.pos.clone())
    return attns, snaps, victims


def test_tiu_matches_rtl():
    for budget, recent, seed in [(4, 0, 0), (7, 2, 1), (12, 3, 2)]:
        sink, steps, L, H = 1, 80, 2, 3
        attns, snaps, victims = drive("tiu", budget, sink, recent, None, steps, L=L, H=H, seed=seed)
        for l in range(L):
            for h in range(H):
                rtl, slot_pos, got = RTLTiu(budget + 1), [-1] * (budget + 1), []
                for t in range(steps):
                    k = next(i for i in range(budget + 1) if not rtl.valid[i])  # lowest free slot
                    rtl.load(k)
                    slot_pos[k] = t
                    a = attns[t][l, 0, h]
                    for i in range(budget + 1):
                        if rtl.valid[i]:
                            rtl.acc(i, int(torch.round(a[i] * 255).clamp(0, 255)))
                    if sum(rtl.valid) > budget:
                        allowed = [sink <= slot_pos[i] <= t - recent for i in range(budget + 1)]
                        v = rtl.evict(allowed)
                        got.append((t, v, slot_pos[v]))
                want = [(t, s, p) for t, ll, hh, s, p in victims if ll == l and hh == h]
                assert got == want, (budget, l, h, got[:5], want[:5])


def test_scores_match_history():
    for family, param in [("h2o", None), ("rw", 5), ("ema", 0.9)]:
        g = torch.Generator().manual_seed(3)
        L, H, budget, steps = 2, 2, 6, 40
        cache = SlotCache(L, 1, H, budget + 1, 4, "cpu", torch.float32)
        pol = Streaming(family, cache, [False] * L, 10**9, [budget], [1], [0], None if param is None else [param])
        attns = []
        for t in range(steps):
            slot = cache.claim()
            cache.t = t + 1
            pol.on_load(slot)
            a = torch.softmax((torch.randn(L, 1, H, budget + 1, generator=g)).masked_fill(cache.pos < 0, float("-inf")), -1)
            attns.append(a)
            pol.update(list(a), t)
            snap = cache.pos.clone()
            # Compare before eviction, on every live slot.
            live = snap >= 0
            exp = torch.zeros(snap.shape)
            for l, r, h, s in live.nonzero().tolist():
                p = int(snap[l, r, h, s])
                hist = [float(attns[u][l, r, h, s]) for u in range(p, t + 1)]
                if family == "h2o":
                    v = sum(hist)
                elif family == "rw":
                    v = sum(hist[-param:])
                else:
                    v = 0.0
                    for x in hist:
                        v = param * v + x
                exp[l, r, h, s] = v
            got = pol.score.masked_fill(~live, 0)
            assert torch.allclose(got, exp, atol=1e-5), (family, t, (got - exp).abs().max())
            pol.evict(cache)


def test_invariants():
    for family, param, gran in [("window", None, "head"), ("h2o", None, "head"), ("tiu", None, "layer"),
                                ("rw", 4, "head"), ("ema", 0.99, "layer"), ("h2o", None, "layer")]:
        for budget, sink, recent in [(8, 4, 3), (8, 4, 0), (16, 4, 8)]:
            attns, snaps, victims = drive(family, budget, sink, recent, param, 60, gran, L=2, H=4,
                                          local=[False, True], window=20)
            gone = set()
            for t, pos in enumerate(snaps):
                live = pos >= 0
                assert bool((live.sum(-1) <= budget).all())
                for l in range(pos.shape[0]):
                    for h in range(pos.shape[2]):
                        held = set(pos[l, 0, h][live[l, 0, h]].tolist())
                        lo = t - 20 + 2 if l == 1 else 0  # local layer keeps positions > t + 1 - window
                        for p in range(max(lo, 0), min(sink, t + 1)):
                            assert p in held, ("sink lost", family, l, h, t, p)
                        for p in range(max(lo, t - recent + 1, 0), t + 1):
                            assert p in held, ("recent lost", family, l, h, t, p)
                        assert not (held & {p for (ll, hh, p) in gone if ll == l and hh == h})
                        if l == 1:
                            assert all(p > t + 1 - 20 for p in held), "local layer kept a stale slot"
                if gran == "layer":
                    assert bool((pos == pos[:, :, :1]).all())
                for (tt, l, h, s, p) in victims:
                    if tt == t:
                        gone.add((l, h, p))


def test_window_evicts_oldest():
    _, snaps, _ = drive("window", 8, 2, 0, None, 30, L=1, H=1)
    # After warm-up the held set is exactly sinks + the newest budget - sink tokens.
    for t in range(10, 30):
        held = sorted(snaps[t][0, 0, 0][snaps[t][0, 0, 0] >= 0].tolist())
        assert held == [0, 1] + list(range(t - 5, t + 1)), (t, held)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
