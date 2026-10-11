"""Checks for the selection code that need no model: run with `python tests/test_policies.py`."""

import itertools
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from subcache.engine import KVCache  # noqa: E402
from subcache.policies import (COM, FLQMI, POLICIES, evict, greedy_select, rkv_scores, slot_similarity,  # noqa: E402
                               top_m, wfl_prune, wfl_select)


def value(name, S, members, eta=1.0):
    """Objective value of a set for ONE problem, written directly from the PRISM formulas.
    S [Q, n]; members: list of slot indices."""
    if not members:
        return 0.0
    sub = S[:, members]
    if name == "com":
        return (eta * torch.sqrt(S.sum(0)[members]).sum() + torch.sqrt(sub.sum(1)).sum()).item()
    return (sub.amax(1).sum() + eta * S.amax(0)[members].sum()).item()


def check_objective(name, cls):
    torch.manual_seed(0)
    P, Q, n, m = 4, 6, 9, 4
    S = torch.rand(P, Q, n).softmax(-1)
    held = torch.zeros(P, n, dtype=torch.bool)
    held[:, -2:] = True  # two protected slots
    cand = ~held

    # Incremental gains equal f(A + i) - f(A) computed from scratch.
    obj = cls(eta=0.7)
    obj.start(S, held)
    gains = obj.gains()
    for p in range(P):
        base = value(name, S[p], [n - 2, n - 1], 0.7)
        for i in range(n - 2):
            direct = value(name, S[p], [n - 2, n - 1, i], 0.7) - base
            assert abs(gains[p, i].item() - direct) < 1e-5, (name, p, i, gains[p, i].item(), direct)

    # Diminishing returns: the gain of i given A is at least its gain given a superset of A.
    for p in range(P):
        for i in range(n - 2):
            small = value(name, S[p], [0, i]) - value(name, S[p], [0])
            big = value(name, S[p], [0, 1, 2, i]) - value(name, S[p], [0, 1, 2])
            if i not in (0, 1, 2):
                assert small >= big - 1e-6, (name, "not submodular", p, i)

    # Exact greedy: right count, only candidates, and within (1 - 1/e) of the brute-force optimum.
    sel = greedy_select(cls(eta=0.7), S, cand, held, m)
    assert (sel.sum(1) == m).all() and not (sel & ~cand).any()
    for p in range(P):
        chosen = [n - 2, n - 1] + sel[p].nonzero().flatten().tolist()
        base = value(name, S[p], [n - 2, n - 1], 0.7)
        best = max(value(name, S[p], [n - 2, n - 1] + list(c), 0.7)
                   for c in itertools.combinations(range(n - 2), m))
        got = value(name, S[p], chosen, 0.7)
        assert got - base >= (1 - 1 / 2.718281828) * (best - base) - 1e-6, (name, p, got, best)

    # Stochastic greedy with a sample covering every candidate is exact greedy.
    sel2 = greedy_select(cls(eta=0.7), S, cand, held, m, eps=1e-9)
    assert torch.equal(sel, sel2)
    # And with a small sample it still returns m distinct candidates.
    sel3 = greedy_select(cls(eta=0.7), S, cand, held, m, eps=0.5, generator=torch.Generator().manual_seed(1))
    assert (sel3.sum(1) == m).all() and not (sel3 & ~cand).any()
    print(f"{name}: gains, submodularity, greedy quality, stochastic greedy OK")


def check_top_m():
    score = torch.tensor([[5.0, 1.0, 9.0, 3.0, 7.0]])
    cand = torch.tensor([[True, True, False, True, True]])
    assert top_m(score, cand, 2).tolist() == [[True, False, False, False, True]]
    assert top_m(score, cand, 10).tolist() == cand.tolist()  # asks for more than exist
    print("top_m OK")


def wfl_value(sim, w, members, mix):
    if not members:
        return 0.0
    return ((1 - mix) * w[members].sum() + mix * (w * sim[:, members].amax(1)).sum()).item()


def check_wfl():
    torch.manual_seed(0)
    P, n, m, mix = 3, 9, 4, 0.6
    sim = slot_similarity(torch.randn(P, n, 8), torch.randn(P, n, 8))
    assert torch.allclose(sim, sim.transpose(1, 2)) and (sim.diagonal(dim1=1, dim2=2) == 1).all()
    assert sim.min() >= 0 and sim.max() <= 1
    w = torch.rand(P, n)
    held = torch.zeros(P, n, dtype=torch.bool)
    held[:, -2:] = True
    cand = ~held

    # mix = 0 is top-m by weight, i.e. H2O.
    assert torch.equal(wfl_select(sim, w, cand, held, m, mix=0.0), top_m(w, cand, m))

    sel = wfl_select(sim, w, cand, held, m, mix=mix)
    assert (sel.sum(1) == m).all() and not (sel & ~cand).any()
    for p in range(P):
        base_set = [n - 2, n - 1]
        base = wfl_value(sim[p], w[p], base_set, mix)
        got = wfl_value(sim[p], w[p], base_set + sel[p].nonzero().flatten().tolist(), mix)
        best = max(wfl_value(sim[p], w[p], base_set + list(c), mix) for c in itertools.combinations(range(n - 2), m))
        assert got - base >= (1 - 1 / 2.718281828) * (best - base) - 1e-6, (p, got, best)
        for i in range(3, n - 2):  # diminishing returns
            small = wfl_value(sim[p], w[p], [0, i], mix) - wfl_value(sim[p], w[p], [0], mix)
            big = wfl_value(sim[p], w[p], [0, 1, 2, i], mix) - wfl_value(sim[p], w[p], [0, 1, 2], mix)
            assert small >= big - 1e-6

    # A slot that duplicates an already-held slot is worth less than an equally weighted new one.
    keys = torch.randn(1, 4, 8)
    keys[0, 1] = keys[0, 3]
    vals = torch.randn(1, 4, 8)
    vals[0, 1] = vals[0, 3]
    s2 = slot_similarity(keys, vals)
    h2 = torch.tensor([[False, False, False, True]])
    pick = wfl_select(s2, torch.tensor([[0.2, 0.3, 0.2, 0.3]]), ~h2, h2, 1, mix=1.0)
    assert not pick[0, 1], "picked the duplicate of a held slot"
    assert torch.equal(wfl_select(sim, w, cand, held, m, mix=mix, eps=1e-9), sel)
    print("wfl: similarity, mix=0 equals h2o, greedy quality, submodularity, duplicate handling OK")


def check_prune():
    torch.manual_seed(1)
    P, n, mix = 4, 10, 0.6
    sim = slot_similarity(torch.randn(P, n, 8), torch.randn(P, n, 8))
    w = torch.rand(P, n)
    held = torch.zeros(P, n, dtype=torch.bool)
    held[:, -2:] = True
    cand = ~held
    base = [n - 2, n - 1]
    # mix = 0 removes the lowest weights: the same set H2O keeps.
    assert torch.equal(wfl_prune(sim, w, cand, held, 5, mix=0.0, step=1), top_m(w, cand, 5))
    # Removing one slot, one at a time, is exactly optimal.
    one = wfl_prune(sim, w, cand, held, n - 3, mix=mix, step=1)
    assert (one.sum(1) == n - 3).all() and not (one & ~cand).any()
    gaps = []
    for p in range(P):
        got = wfl_value(sim[p], w[p], base + one[p].nonzero().flatten().tolist(), mix)
        best = max(wfl_value(sim[p], w[p], base + list(c), mix) for c in itertools.combinations(range(n - 2), n - 3))
        assert abs(got - best) < 1e-6, (p, got, best)
        # Removing several: not guaranteed optimal, so only require it to be close on average.
        sel = wfl_prune(sim, w, cand, held, 4, mix=mix, step=1)
        got = wfl_value(sim[p], w[p], base + sel[p].nonzero().flatten().tolist(), mix)
        best = max(wfl_value(sim[p], w[p], base + list(c), mix) for c in itertools.combinations(range(n - 2), 4))
        floor = wfl_value(sim[p], w[p], base, mix)
        gaps.append((got - floor) / (best - floor))
    assert min(gaps) > 0.9, gaps
    batched = wfl_prune(sim, w, cand, held, 4, mix=mix, step=2)
    assert (batched.sum(1) == 4).all() and not (batched & ~cand).any()
    print(f"prune: mix=0 equals h2o, single removal optimal, multi-removal reaches {min(gaps):.0%}+ of optimum")


def check_rkv():
    torch.manual_seed(0)
    P, groups, fill, n = 2, 4, 3, 12
    S = torch.rand(P, groups * fill, n).softmax(-1)
    keys = torch.randn(P, n, 64)
    keys[:, 5] = keys[:, 2] * 1.5   # slot 5 repeats slot 2 (same direction)
    keys[:, 9] = keys[:, 2] * 0.7   # and so does slot 9
    z = rkv_scores(S, keys, groups, lam=0.0)  # redundancy only
    assert z.shape == (P, n)
    # With importance switched off, the repeated slots score below every unrelated slot.
    others = [i for i in range(n) if i not in (2, 5, 9)]
    assert (z[:, [2, 5, 9]].max(1).values < z[:, others].min(1).values).all(), z
    z1 = rkv_scores(S, keys, groups, lam=1.0)  # importance only
    assert torch.allclose(z1.sum(1) / z1.sum(1), torch.ones(P)) and (z1 >= 0).all()
    print("rkv: shapes, repeated keys marked redundant, importance-only is non-negative")


def check_window_plan():
    cfg = types.SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=2, hidden_size=8, head_dim=4)
    c = KVCache(cfg, 1, 16, torch.device("cpu"), torch.float32)
    c.enable_window(4, 2, "recent")
    assert c.window_plan(6) == ([2, 3, 4, 5], [0, 1, 2, 3]) and c.win_fill == 4
    assert c.window_plan(1) == ([0], [0]) and c.win_ptr == 1
    # Reservoir: every query should end up in the window about equally often.
    counts = [0] * 40
    for seed in range(3000):
        c = KVCache(cfg, 1, 16, torch.device("cpu"), torch.float32)
        c.enable_window(4, 2, "reservoir", seed)
        held = {}
        for start in range(0, 40, 8):  # five forwards of 8 queries
            rows, slots = c.window_plan(8)
            assert len(set(slots)) == len(slots)
            for r, sl in zip(rows, slots):
                held[sl] = start + r
        assert c.win_fill == 4 and len(held) == 4
        for q in held.values():
            counts[q] += 1
    expected = 3000 * 4 / 40
    assert all(abs(x - expected) < 0.25 * expected for x in counts), counts
    print("window plan: recent ring and uniform reservoir OK")


def check_evict_all():
    cfg = types.SimpleNamespace(num_hidden_layers=3, num_key_value_heads=2, num_attention_heads=8, hidden_size=64, head_dim=8)
    for value_aware in (False, True):
        for eps in (None, 0.2):
            for policy in POLICIES:
                torch.manual_seed(0)
                c = KVCache(cfg, 1, 100, torch.device("cpu"), torch.float32)
                c.enable_window(6, 8)
                c.length = 80
                c.k.normal_()
                c.v.normal_()
                c.score.uniform_()
                c.window[..., :6, :80] = torch.rand(3, 1, 8, 6, 80).softmax(-1)
                c.win_fill = 6
                st = evict(c, policy, 30, value_aware=value_aware, eps=eps, generator=torch.Generator().manual_seed(0),
                           wfl_mode="prune" if (policy == "wfl" and eps) else "select", chunk=4)
                live = c.keep[..., :80].sum(-1)
                if policy == "full":
                    assert st is None and int(live.min()) == 80
                    continue
                assert int(live.min()) == int(live.max()) == 30, (policy, live)
                assert c.keep[..., :4].all() and c.keep[..., 65:80].all()
                assert st["evicted"] == 6 * 50 and 0 < st["retained_mean"] <= 1
                c.length = 90  # a later eviction, when some slots are already gone
                evict(c, policy, 30, value_aware=value_aware, eps=eps, generator=torch.Generator().manual_seed(1))
                assert int(c.keep[..., :90].sum(-1).min()) == int(c.keep[..., :90].sum(-1).max()) == 30, policy
    print("evict: every policy keeps exactly `capacity`, protects sink + recent, twice in a row")


def check_compact():
    cfg = types.SimpleNamespace(num_hidden_layers=3, num_key_value_heads=2, num_attention_heads=8, hidden_size=64, head_dim=8)

    def make():
        torch.manual_seed(0)
        c = KVCache(cfg, 2, 100, torch.device("cpu"), torch.float32)
        c.enable_window(6, 8)
        c.length = c.pos = 80
        c.k.normal_()
        c.v.normal_()
        c.score.uniform_()
        c.window[..., :6, :80] = torch.rand(3, 2, 8, 6, 80).softmax(-1)
        c.win_fill = 6
        return c

    for policy in ("h2o", "com", "wfl"):
        masked, packed = make(), make()
        evict(masked, policy, 30)
        st = evict(packed, policy, 30, compact=True)
        assert packed.length == 30 and packed.pos == 80 and not packed.has_evicted and "compact_s" in st
        assert packed.keep[..., :80].all()
        for li in range(3):
            for b in range(2):
                for h in range(2):
                    live = masked.keep[li, b, h, :80].nonzero().flatten()
                    assert torch.equal(packed.k[li, b, h, :30], masked.k[li, b, h, live]), policy
                    assert torch.equal(packed.v[li, b, h, :30], masked.v[li, b, h, live])
                    assert torch.equal(packed.score[li, b, h, :30], masked.score[li, b, h, live])
                    for g in range(4):
                        assert torch.equal(packed.window[li, b, h * 4 + g, :, :30], masked.window[li, b, h * 4 + g][:, live])
        assert (packed.window[..., 30:80] == 0).all()
    print("compact: packed cache holds exactly the kept keys, values, scores and window columns")


if __name__ == "__main__":
    check_prune()
    check_rkv()
    check_compact()
    check_objective("com", COM)
    check_objective("flqmi", FLQMI)
    check_top_m()
    check_wfl()
    check_window_plan()
    check_evict_all()
