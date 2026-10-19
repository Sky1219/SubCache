"""Simulated int8 weights (weight-only fake quantization) to approximate Reef's int8 model.

Every Linear weight in the blocks (q, k, v, out, MLP in and out), the token embedding (tied to
lm_head) and the position embedding are rounded to symmetric int8 with one fp32 scale per output
channel (per row of the embedding tables), then dequantized in place. Biases, layer norms,
activations, attention and the KV cache stay fp32. Reef's actual scheme (granularity, activation
and KV precision) may differ; this is a first-order check that the policy ranking survives int8
weights, not a model of Reef's numerics.
"""

import torch


@torch.no_grad()
def fake_quant_rows(w):
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    return (torch.round(w / scale).clamp_(-127, 127) * scale).to(w.dtype)


@torch.no_grad()
def quantize_weights_int8(model):
    tr = model.transformer
    tr.wte.weight.copy_(fake_quant_rows(tr.wte.weight))  # also lm_head: the weights are tied
    tr.wpe.weight.copy_(fake_quant_rows(tr.wpe.weight))
    for block in tr.h:
        a = block.attn.attention
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.out_proj, block.mlp.c_fc, block.mlp.c_proj):
            lin.weight.copy_(fake_quant_rows(lin.weight))  # nn.Linear weight is [out, in]
    assert model.lm_head.weight.data_ptr() == tr.wte.weight.data_ptr(), "lm_head no longer tied to wte"
    return model
