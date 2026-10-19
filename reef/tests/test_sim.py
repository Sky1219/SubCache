"""Checks for reef/sim.py (the TIU team's non-permanent H2O simulation): run with
`python reef/tests/test_sim.py [--device cuda]` from the SubCache root.

1. keep_mask equals a brute-force per-row selection on random scores, with a local window.
2. With a budget larger than the story, the simulation reproduces HF's full-cache forward.
3. With a small budget it changes the output (so test 2 is not vacuous) and every row past the
   budget attends to exactly budget + 1 tokens in global layers.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from reef.data import load_stories  # noqa: E402
from reef.engine import load_model  # noqa: E402
from reef.sim import keep_mask, story_logits  # noqa: E402

DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else os.environ.get("REEF_DEVICE", "cpu")


def test_keep_mask_brute_force():
    g = torch.Generator().manual_seed(0)
    T, H, window = 40, 3, 25
    for budget, sink, recent in [(10, 2, 3), (16, 4, 8), (12, 4, 0)]:
        i = torch.arange(T)[:, None]
        j = torch.arange(T)[None, :]
        allowed = (j <= i) & (j > i - window)
        acc = torch.rand(H, T, T, generator=g)
        got = keep_mask(allowed, acc, budget, sink, recent)
        for h in range(H):
            for r in range(T):
                ok = [c for c in range(T) if allowed[r, c]]
                prot = [c for c in ok if c < sink or c >= r - recent]
                rest = sorted((c for c in ok if c not in prot), key=lambda c: -float(acc[h, r, c]))
                want = set(prot) | set(rest[: budget - sink - recent])
                have = set(got[h, r].nonzero().flatten().tolist())
                assert have == want, (budget, h, r, sorted(have), sorted(want))
                if r >= budget + 1 and r < window:
                    assert len(have) == budget + 1


def test_sim_matches_full_cache_and_changes_output():
    tok, model = load_model(device=DEVICE)
    s = max(load_stories(tok, n=40, seed=1), key=lambda s: len(s["ids"]))  # > 256: local layers matter
    ids = s["ids"]
    with torch.no_grad():
        hf = torch.log_softmax(model(torch.tensor([ids], device=model.device)).logits[0].float(), -1)
    full = story_logits(model, ids, None)
    big = story_logits(model, ids, dict(budget=len(ids) + 10, sink=4, recent=8, lookahead=True))
    d_full, d_big = (full - hf).abs().max().item(), (big - hf).abs().max().item()
    print(f"  story len {len(ids)}: |patched - HF| {d_full:.1e}, |sim at budget > T - HF| {d_big:.1e}")
    assert d_full < 1e-4 and d_big < 1e-4
    for look in (True, False):
        small = story_logits(model, ids, dict(budget=32, sink=4, recent=16, lookahead=look))
        diff = (small - hf).abs().max().item()
        print(f"  budget 32, lookahead={look}: max |diff| vs full {diff:.2f}")
        assert diff > 0.1


if __name__ == "__main__":
    print("device:", DEVICE)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
