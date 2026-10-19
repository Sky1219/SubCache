"""Checks for reef/kvquant.py (the KV Cache Engine codecs): run with
`python reef/tests/test_kvquant.py [--device cuda]` from the SubCache root.

1. CQ-3-rot values: bit-exact against a scalar port of kv-cache-engine/analysis/wht_value_pyref.py.
2. ChannelQuant keys: bit-exact against a scalar port of the ISA's per-channel INT4 rule.
3. In the engine, with streaming eviction, the plumbing (when a key group closes, which tokens
   it covers, outlier channels, interaction with eviction) matches one HF forward that applies
   the codecs from the eviction record, exactly when the codecs are swapped for smooth stand-ins.
   With the real codecs the two agree only approximately: the engine's and HF's projections differ
   in the last bits (~1e-7) and INT3/INT4/FP16 rounding turns rare ones into whole-code flips.
"""

import os
import sys

import numpy as np
import torch
import transformers.models.gpt_neo.modeling_gpt_neo as neo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from reef.kvquant import EPS, KVQuant, fp16, quantize_key_group, wht3_roundtrip  # noqa: E402

DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else os.environ.get("REEF_DEVICE", "cpu")


def ref_wht3_row(row):
    """Scalar port of compress_row + decompress_row from wht_value_pyref.py (numpy fp16)."""
    def fwht(r):
        D, h = len(r), 1
        while h < D:
            for i in range(0, D, 2 * h):
                for j in range(i, i + h):
                    a, b = np.float64(r[j]), np.float64(r[j + h])
                    r[j], r[j + h] = np.float16(a + b), np.float16(a - b)
            h *= 2
        return r
    r = fwht(np.array(row, dtype=np.float16))
    s = np.float64(np.float16(np.max(np.abs(r)))) / 3
    s = np.float16(max(s, EPS))
    codes = [int(max(-4, min(3, np.rint(np.float64(x) / np.float64(s))))) for x in r]
    d = np.array([np.float16(np.float64(c) * np.float64(s)) for c in codes], dtype=np.float16)
    d = fwht(d)
    return [np.float32(np.float64(x) / len(row)) for x in d]


def test_wht3_bit_exact():
    g = torch.Generator().manual_seed(0)
    for D in (32, 64):
        x = torch.randn(200, D, generator=g) * torch.rand(200, 1, generator=g) * 4
        x[::7, 3] *= 40  # an outlier channel: the case the rotation is for
        got = wht3_roundtrip(x)
        want = torch.tensor([ref_wht3_row(row.numpy()) for row in x])
        assert torch.equal(got, want), (D, (got - want).abs().max())


def test_key_group_bit_exact():
    g = torch.Generator().manual_seed(1)
    H, C, D = 3, 20, 32
    k = fp16(torch.randn(H, C, D, generator=g) * 3)
    members = torch.rand(H, C, generator=g) < 0.6
    outlier = torch.zeros(H, D, dtype=torch.bool)
    outlier[:, [5, 17]] = True
    got = quantize_key_group(k, members, outlier)
    for h in range(H):
        for d in range(D):
            col = [float(k[h, c, d]) for c in range(C) if members[h, c]]
            a = np.float16(max(abs(v) for v in col)) if col else np.float16(0)
            s = np.float16(max(np.float64(a) / 7, EPS))
            for c in range(C):
                v = float(k[h, c, d])
                if members[h, c] and not outlier[h, d]:
                    v = float(np.float32(max(-8, min(7, np.rint(v / np.float64(s)))) * np.float64(s)))
                assert float(got[h, c, d]) == v, (h, c, d, float(got[h, c, d]), v)


def hf_kvq_logits(model, ids, kvq, evicted_at):
    """One HF forward whose attention reproduces streaming KV quantization + eviction.
    evicted_at [layers, heads, T]: step at whose end each token was evicted (> T if never)."""
    orig = neo.GPTNeoSelfAttention._attn
    G = kvq.group

    def attn(self, query, key, value, attention_mask=None, *a, **kw):
        l = self.layer_id
        q = query.float()[0]  # [H, T, D]
        k16, v = fp16(key.float()[0]), wht3_roundtrip(value.float()[0])
        H, T, D = k16.shape
        ev = evicted_at[l].to(q.device)
        pos = torch.arange(T, device=q.device)
        kq = k16.clone()
        closed_after = torch.full((T,), T + 10, device=q.device)  # first query that reads kq
        for start in range(0, T - G + 1, G):
            end = start + G - 1
            grp = (pos >= start) & (pos <= end)
            members = grp[None, :] & (ev >= end)  # still cached when the group closed
            kq = torch.where(grp[None, :, None], quantize_key_group(k16, members, kvq.outlier[l].to(q.device)), kq)
            closed_after[grp] = end + 1
        s16 = q @ k16.transpose(-1, -2)
        sq = q @ kq.transpose(-1, -2)
        use_q = pos[:, None] >= closed_after[None, :]
        scores = torch.where(use_q[None], sq, s16)
        i = pos[:, None]
        visible = self.bias[0, 0, :T, :T] & (i <= ev[:, None, :])  # [H, T, T]
        scores = scores.masked_fill(~visible, float("-inf"))
        p = torch.softmax(scores, -1)
        return (p @ v)[None].to(value.dtype), p[None]

    neo.GPTNeoSelfAttention._attn = attn
    try:
        with torch.no_grad():
            return model(torch.tensor([ids], device=model.device)).logits[0].float().cpu()
    finally:
        neo.GPTNeoSelfAttention._attn = orig


def smooth_codecs():
    """Stand-ins with the same structure but no rounding: keys of a closed group are rescaled by a
    smooth function of the group's per-channel amax (outlier channels untouched), values by a
    smooth function of the token's amax after a WHT. Group timing, membership, outliers and their
    interaction with eviction then show up as differences, without rounding cliffs."""
    import reef.kvquant as KQ

    def key_group(k, members, outlier):
        kd = k.double()
        amax = (kd.abs() * members[..., None]).amax(dim=-2, keepdim=True)
        use = members[..., None] & ~outlier[..., None, :]
        return torch.where(use, kd * (0.8 + 0.1 * torch.tanh(amax)), kd).to(k.dtype)

    def values(v):
        amax = v.double().abs().amax(-1, keepdim=True)
        return (v.double() * (0.9 + 0.05 * torch.tanh(amax))).float()

    return {"fp16": lambda x: x, "quantize_key_group": key_group, "wht3_roundtrip": values}, KQ


def compare(model, engine, ids, G, family, budget, sink, recent, outlier):
    import reef.tests.test_gate as gate
    engine.kvq = KVQuant(outlier, group=G)
    try:
        logits, evicted_at, _ = gate.run_streaming(engine, ids, family, budget, sink, recent, None, "head")
    finally:
        engine.kvq = None
    ref = hf_kvq_logits(model, ids, KVQuant(outlier.cpu(), group=G), evicted_at[:, 0])
    diff = (logits[0] - ref).abs().amax(-1)
    agree = (logits[0].argmax(-1) == ref.argmax(-1)).float().mean().item()
    return diff, agree


CASES = [(16, "h2o", [16], [4], [8]), (32, "tiu", [24], [0], [11]), (128, "h2o", [400], [4], [8])]


def test_engine_kvq_plumbing_matches_hf():
    """Exact check of streaming group-close / membership / outlier / eviction logic."""
    import reef.tests.test_gate as gate
    gate.DEVICE = DEVICE
    tok, model, engine, stories = gate.setup()
    ids = stories[-1]["ids"]  # > 300 tokens
    outlier = torch.zeros(engine.n_layers, engine.n_heads, engine.head_dim, dtype=torch.bool, device=engine.device)
    outlier[..., :2] = True
    subs, KQ = smooth_codecs()
    saved = {n: getattr(KQ, n) for n in subs}
    here = sys.modules[__name__]
    saved_here = {n: getattr(here, n) for n in subs}
    for n, f in subs.items():
        setattr(KQ, n, f)
        setattr(here, n, f)
    try:
        for G, family, budget, sink, recent in CASES:
            diff, agree = compare(model, engine, ids, G, family, budget, sink, recent, outlier)
            print(f"  smooth codecs, G={G} {family} budget {budget[0]}: max |logit diff| {diff.max().item():.1e}")
            assert diff.max().item() < 1e-3, (G, family)
    finally:
        for n, f in saved.items():
            setattr(KQ, n, f)
        for n, f in saved_here.items():
            setattr(here, n, f)


def test_engine_kvq_real_codecs_close_to_hf():
    """With the real codecs, last-bit differences between the engine's and HF's projections flip
    occasional INT3/INT4 codes, so only near-agreement is expected."""
    import reef.tests.test_gate as gate
    gate.DEVICE = DEVICE
    tok, model, engine, stories = gate.setup()
    ids = stories[-1]["ids"]
    outlier = torch.zeros(engine.n_layers, engine.n_heads, engine.head_dim, dtype=torch.bool, device=engine.device)
    outlier[..., :2] = True
    for G, family, budget, sink, recent in CASES:
        diff, agree = compare(model, engine, ids, G, family, budget, sink, recent, outlier)
        print(f"  real codecs, G={G} {family} budget {budget[0]}: top-1 agreement {agree:.4f}, "
              f"median |logit diff| {diff.median().item():.2f}")
        assert agree > 0.9


if __name__ == "__main__":
    print("device:", DEVICE)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
