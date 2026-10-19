"""Manual GPT-Neo decode step with a slot-addressed KV cache, for streaming eviction.

Follows subcache/engine.py: the weights and small modules (embeddings, layer norms, projections,
MLP) come from the Hugging Face model; the layer loop, attention, cache writes and masking are
done here so the K/V tensors are ours to evict.

GPT-Neo specifics, read from transformers' modeling_gpt_neo.py and checked by tests/test_gate.py:
  * Learned absolute position embeddings are added to the input embedding. There is no rotary or
    relative term, so a cached key/value already carries its position: a slot can be evicted and
    its storage reused by a later token without touching any other slot.
  * Attention scores are NOT scaled by 1/sqrt(head_dim), and are computed in fp32.
  * Layers alternate global / local. A local layer's query at position t sees keys at positions
    t-window+1 .. t (window_size = 256 for TinyStories-28M).

Cache layout. Every batch row (one story under one eviction config) owns `capacity` slots per
(layer, head). A slot holds one token's K and V and that token's position (`pos`, -1 = empty).
A new token goes into the lowest empty slot of each (layer, head), as the TIU control wrapper in
LonghornSilicon/token-importance-unit does, so after evictions the slot order is scrambled.
Attention does not care about slot order. Eviction is permanent: an evicted slot only ever
comes back holding a new token.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "roneneldan/TinyStories-28M"


def load_model(name=MODEL, device="cpu", dtype=torch.float32):
    """Tokenizer + eval-mode model with eager attention (the path the engine is gated against)."""
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="eager")
    return tok, model.to(device).eval()


class SlotCache:
    """k, v : [layers, rows, heads, capacity, head_dim]
    pos  : [layers, rows, heads, capacity] long, position of the token in each slot, -1 = empty
    t    : position of the next token (all rows advance together)."""

    def __init__(self, n_layers, rows, n_heads, capacity, head_dim, device, dtype):
        self.k = torch.zeros(n_layers, rows, n_heads, capacity, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.pos = torch.full((n_layers, rows, n_heads, capacity), -1, device=device, dtype=torch.long)
        self.capacity = capacity
        self.t = 0

    @property
    def live(self):
        """[layers, rows, heads] number of occupied slots."""
        return (self.pos >= 0).sum(-1)

    def claim(self):
        """Lowest empty slot of every (layer, row, head) for the next token; marks it occupied by
        position t. Returns the slot ids, [layers, rows, heads]."""
        empty = self.pos < 0
        if not bool(empty.any(-1).all()):
            raise ValueError(f"cache overflow at t={self.t}: a (layer, row, head) has no empty slot "
                             f"(capacity {self.capacity}); evict before the next token")
        slot = empty.to(torch.uint8).argmax(-1)  # argmax returns the first maximum: lowest empty slot
        self.pos.scatter_(-1, slot[..., None], self.t)
        return slot


class NeoEngine:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        tr = model.transformer
        self.wte, self.wpe, self.blocks, self.ln_f = tr.wte, tr.wpe, tr.h, tr.ln_f
        self.lm_head = model.lm_head
        self.n_layers = self.config.num_layers
        self.n_heads = self.config.num_heads
        self.head_dim = self.config.hidden_size // self.n_heads
        self.window = self.config.window_size
        self.local = [kind == "local" for kind in self.config.attention_layers]
        self.device = self.wte.weight.device
        self.dtype = self.wte.weight.dtype

    def new_cache(self, rows, capacity):
        return SlotCache(self.n_layers, rows, self.n_heads, capacity, self.head_dim, self.device, self.dtype)

    def visible(self, cache, layer):
        """[rows, heads, capacity] bool: slots the current query (position cache.t) may attend to."""
        pos = cache.pos[layer]
        ok = pos >= 0
        if self.local[layer]:
            ok &= pos > cache.t - self.window
        return ok

    @torch.inference_mode()
    def step(self, ids, cache):
        """Feed one token per row (ids: [rows]) at position cache.t.

        Writes the token's K/V into a freshly claimed slot of every (layer, head), attends (a token
        attends to itself), and advances cache.t. Returns (hidden [rows, d] after the final layer
        norm, attention weights per layer as a list of [rows, heads, capacity] fp32 tensors with
        zeros on slots that were not visible, slot ids claimed [layers, rows, heads]).
        Logits are left to the caller (self.logits) so it can skip rows it does not score.
        """
        rows = ids.shape[0]
        t = cache.t
        if t >= self.config.max_position_embeddings:
            raise ValueError(f"position {t} beyond max_position_embeddings")
        slot = cache.claim()
        r_idx = torch.arange(rows, device=self.device)[:, None]
        h_idx = torch.arange(self.n_heads, device=self.device)[None, :]

        h = self.wte(ids) + self.wpe.weight[t]
        attns = []
        for li, block in enumerate(self.blocks):
            attn = block.attn.attention
            x = block.ln_1(h)
            q = attn.q_proj(x).view(rows, self.n_heads, self.head_dim)
            k = attn.k_proj(x).view(rows, self.n_heads, self.head_dim)
            v = attn.v_proj(x).view(rows, self.n_heads, self.head_dim)
            cache.k[li][r_idx, h_idx, slot[li]] = k
            cache.v[li][r_idx, h_idx, slot[li]] = v

            keys = cache.k[li].float()  # [rows, heads, capacity, head_dim]
            scores = torch.matmul(keys, q.float()[..., None]).squeeze(-1)  # no 1/sqrt(d) in GPT-Neo
            scores = scores.masked_fill(~self.visible(cache, li), float("-inf"))
            w = torch.softmax(scores, dim=-1)  # fp32, [rows, heads, capacity]
            attns.append(w)
            out = torch.matmul(w.to(cache.v.dtype)[:, :, None, :], cache.v[li]).squeeze(2)  # [rows, heads, hd]
            h = h + attn.out_proj(out.reshape(rows, -1))
            h = h + block.mlp(block.ln_2(h))

        cache.t = t + 1
        return self.ln_f(h), attns, slot

    @torch.inference_mode()
    def logits(self, hidden):
        return self.lm_head(hidden).float()
