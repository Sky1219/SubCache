"""Longhorn Silicon's KV Cache Engine codecs, applied to the engine's cache in streaming order.

Source of truth: github.com/LonghornSilicon/kv-cache-engine (docs/isa/kv_cache_engine_isa.md,
docs/wht_value_rotation.md, analysis/wht_value_pyref.py).

Keys, ChannelQuant CQ-4+: tokens are written as FP16 into a residual buffer; when a group of G
consecutive positions is complete, the engine freezes one FP16 scale per channel,
s_c = fp16(max(amax_c / 7, 2^-14)), and stores round-half-even(k / s_c) clamped to INT4 [-8, 7].
The top-k outlier channels of each (layer, head), from a static calibrated mask, stay FP16.
Until its group closes a key is read back as FP16.

Values, CQ-3-rot: per token, at write time. The FP16 row goes through a raw Walsh-Hadamard
butterfly in FP16 (add/sub, no normalization), gets a per-token FP16 scale
s = fp16(max(amax / 3, 2^-14)) and INT3 codes in [-4, 3]; decoding multiplies back to FP16,
applies the same butterfly and scales by 1/D.

Two choices the spec leaves open in a streaming cache with eviction, made here and stated in
WRITEUP.md: the group's scales use the keys of its tokens still cached when the group closes
(an evicted token has no storage left), and the group closes after the step that wrote its last
token (that token's own query still reads the FP16 keys).
"""

import torch

EPS = 2.0 ** -14
K_QMIN, K_QMAX = -8, 7
V_QMIN, V_QMAX = -4, 3


def fp16(x):
    """Round to FP16 and return in the input's dtype (round-half-even, as numpy/C++ do)."""
    return x.to(torch.float16).to(x.dtype)


def fwht_raw_fp16(x):
    """Raw Walsh-Hadamard butterfly over the last dim (a power of two), every add/sub rounded to
    FP16 as in fwht_raw_f16 of the reference: computed exactly in float64, then rounded."""
    x = fp16(x.double())
    D = x.shape[-1]
    h = 1
    while h < D:
        y = x.view(*x.shape[:-1], D // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = fp16(torch.stack([a + b, a - b], dim=-2)).reshape(*x.shape[:-1], D)
        h *= 2
    return x


def wht3_roundtrip(v):
    """CQ-3-rot compress + decompress of value rows (last dim = head dim). Returns fp32."""
    D = v.shape[-1]
    r = fwht_raw_fp16(v)
    amax = fp16(r.abs().amax(-1, keepdim=True))
    s = fp16(torch.clamp(amax / V_QMAX, min=EPS))
    codes = torch.round(r / s).clamp_(V_QMIN, V_QMAX)  # torch.round is half-to-even
    deq = fp16(codes * s)
    return (fwht_raw_fp16(deq) * (1.0 / D)).float()


def quantize_key_group(k, members, outlier):
    """ChannelQuant CQ-4+ for one group. k [..., C, D] fp16-rounded keys, members [..., C] bool
    (the group's live slots), outlier [..., D] bool. Returns k with members' non-outlier channels
    replaced by their dequantized INT4 values; everything else unchanged."""
    kd = k.double()
    amax = (kd.abs() * members[..., None]).amax(dim=-2, keepdim=True)  # [..., 1, D]
    s = fp16(torch.clamp(fp16(amax) / K_QMAX, min=EPS))
    q = torch.round(kd / s).clamp_(K_QMIN, K_QMAX) * s
    use = members[..., None] & ~outlier[..., None, :]
    return torch.where(use, q, kd).to(k.dtype)


class KVQuant:
    """Applied by NeoEngine.step when set as engine.kvq.
    outlier: [layers, heads, head_dim] bool (static mask); group: key group size G."""

    def __init__(self, outlier, group=128):
        self.outlier = outlier
        self.group = group

    def on_write(self, k, v):
        """New token's K, V [rows, heads, D] as stored: FP16 residual key, CQ-3-rot value."""
        return fp16(k), wht3_roundtrip(v)

    def close_group(self, cache, layer, t):
        """Called after the step that wrote position t: if t ends a key group, quantize it."""
        G = self.group
        if (t + 1) % G:
            return
        pos = cache.pos[layer]  # [rows, heads, C]
        members = (pos >= t + 1 - G) & (pos <= t)
        out = self.outlier[layer][None].expand(pos.shape[0], -1, -1)
        cache.k[layer] = quantize_key_group(cache.k[layer], members, out)


@torch.inference_mode()
def calibrate_outliers(engine, stories, k=2):
    """Static outlier mask: per (layer, head), the k key channels with the largest |key| over
    full-cache passes of `stories` (which must not overlap the evaluation stories)."""
    amax = torch.zeros(engine.n_layers, engine.n_heads, engine.head_dim, device=engine.device)
    for s in stories:
        ids = torch.tensor(s["ids"], device=engine.device)
        cache = engine.new_cache(1, len(ids))
        for t in range(len(ids)):
            engine.step(ids[t:t + 1], cache)
        live = (cache.pos >= 0)[..., None]  # [L, 1, H, C, 1]
        amax = torch.maximum(amax, (cache.k.abs() * live).amax(dim=(1, 3)))
    idx = amax.topk(k, dim=-1).indices
    return torch.zeros_like(amax, dtype=torch.bool).scatter_(-1, idx, True)
