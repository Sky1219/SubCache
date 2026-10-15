"""Manual forward pass for Llama-family HF models with an explicit, preallocated KV cache.

The weights and small modules (embedding, norms, projections, MLP, RoPE tables) come from the
Hugging Face model; the layer loop, attention math, cache writes and positions are done here so
the K/V tensors are ours to inspect, mask and (later) page.

Eviction has two modes. Masked: every (layer, kv_head, slot) has a `keep` bit and evicted slots
are hidden from attention but still stored, which is enough to measure accuracy. Compacted
(KVCache.compact): kept entries are packed to the front and the rest dropped, so the cache is
physically smaller and later steps read less; this is the mode for speed and memory numbers.
Every head keeps the same number of slots, so the packed cache stays a regular tensor.
"""

import copy
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def load_model(name, device, dtype):
    """Tokenizer + eval-mode model with eager attention (the path the engine is gated against)."""
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="eager")
    return tok, model.to(device).eval()


class KVCache:
    """Preallocated K/V for every layer, plus per-(layer, kv_head, slot) keep bits and scores.

    k, v   : [layers, batch, kv_heads, max_len, head_dim]   keys are stored post-RoPE
    keep   : [layers, batch, kv_heads, max_len] bool         False = evicted (masked in attention)
    score  : [layers, batch, kv_heads, max_len] float32      attention mass received, all time
    window : [layers, batch, heads, win_size, max_len]       attention rows of win_size queries,
             only allocated by enable_window. mode 'recent' keeps the newest queries (a ring
             buffer); mode 'reservoir' keeps a uniform random sample of every query seen so far
    length : number of slots in use
    pos    : number of tokens seen, i.e. the position id of the next token. Equal to `length`
             until the cache is compacted; keys are stored already rotated, so a slot never
             needs its position again.
    """

    def __init__(self, config, batch, max_len, device, dtype):
        n_layers = config.num_hidden_layers
        n_kv = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        shape = (n_layers, batch, n_kv, max_len)
        self.k = torch.zeros(*shape, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(*shape, head_dim, device=device, dtype=dtype)
        self.keep = torch.ones(shape, device=device, dtype=torch.bool)
        self.score = torch.zeros(shape, device=device, dtype=torch.float32)
        self.max_len = max_len
        self.length = 0
        self.pos = 0
        self.has_evicted = False  # any keep bit False among the slots in use (masked mode only)
        self.is_row_view = False  # set on the views returned by row()
        self.bytes_per_slot = 2 * head_dim * self.k.element_size()  # one K and one V vector
        self.window = None
        self.win_size = 0
        self.win_fill = 0
        self.win_ptr = 0
        self.win_mode = "recent"
        self.win_seen = 0
        self.win_rng = random.Random(0)

    def enable_window(self, size, n_heads, mode="recent", seed=0):
        n_layers, batch, _, max_len = self.keep.shape
        self.window = torch.zeros(n_layers, batch, n_heads, size, max_len, device=self.k.device, dtype=self.k.dtype)
        self.win_size = size
        self.win_mode = mode
        self.win_rng = random.Random(seed)

    def window_plan(self, n):
        """Which of n new query rows to store, and in which window rows. Returns (rows, slots)
        as lists of equal length and updates the fill / pointer / seen counters."""
        size = self.win_size
        if self.win_mode == "recent":
            take = min(n, size)
            rows = list(range(n - take, n))
            slots = [(self.win_ptr + i) % size for i in range(take)]
            self.win_ptr = (self.win_ptr + take) % size
            self.win_fill = min(size, self.win_fill + take)
        else:  # reservoir sampling: after t queries each one is in the window with probability size/t
            chosen = {}
            for row in range(n):
                self.win_seen += 1
                if self.win_fill < size:
                    chosen[self.win_fill] = row
                    self.win_fill += 1
                else:
                    slot = self.win_rng.randrange(self.win_seen)
                    if slot < size:
                        chosen[slot] = row
            slots, rows = list(chosen.keys()), list(chosen.values())
        return rows, slots

    def row(self, b):
        """A shallow copy that reads and writes batch row b of this cache (as batch size 1).
        Writes to its tensors land in this cache; counters such as has_evicted do not. A view
        cannot be compacted: that would move this row's slots while length stays shared."""
        view = copy.copy(self)
        view.is_row_view = True
        row = slice(b, b + 1)
        view.k, view.v, view.keep, view.score = self.k[:, row], self.v[:, row], self.keep[:, row], self.score[:, row]
        if self.window is not None:
            view.window = self.window[:, row]
        return view

    @property
    def allocated_bytes(self):
        """Memory held by the K and V tensors."""
        return 2 * self.k.numel() * self.k.element_size()

    def compact(self):
        """Pack the kept slots of every head to the front and drop the rest."""
        if self.is_row_view:
            raise RuntimeError("cannot compact a row view: the parent cache's length would no longer match "
                               "the row's slots; use masked eviction (compact=False) on rows")
        n = self.length
        keep = self.keep[..., :n]
        live = int(keep.sum(-1).max())
        if int(keep.sum(-1).min()) != live:
            raise ValueError("compact needs the same number of kept slots in every head")
        if live == n:
            return
        idx = keep.float().topk(live, dim=-1).indices.sort(dim=-1).values  # [layers, batch, kv, live]
        take = idx[..., None].expand(*idx.shape, self.k.shape[-1])
        self.k[..., :live, :] = self.k[..., :n, :].gather(3, take)
        self.v[..., :live, :] = self.v[..., :n, :].gather(3, take)
        self.score[..., :live] = self.score[..., :n].gather(3, idx)
        self.score[..., live:n] = 0
        if self.window is not None:
            groups = self.window.shape[2] // idx.shape[2]
            cols = idx.repeat_interleave(groups, dim=2)[:, :, :, None, :].expand(-1, -1, -1, self.win_size, -1)
            self.window[..., :live] = self.window[..., :n].gather(4, cols)
            self.window[..., live:n] = 0
        self.keep[..., :n] = True
        self.length = live
        self.has_evicted = False

    @property
    def kept_slots(self):
        """Number of live (layer, head, slot) entries among the slots written so far."""
        return int(self.keep[..., : self.length].sum())

    @property
    def kept_bytes(self):
        return self.kept_slots * self.bytes_per_slot

    @property
    def kept_fraction(self):
        total = self.keep[..., : self.length].numel()
        return self.kept_slots / total if total else 1.0


class Engine:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        base = model.model
        self.embed = base.embed_tokens
        self.layers = base.layers
        self.final_norm = base.norm
        self.rotary = base.rotary_emb
        self.lm_head = model.lm_head
        self.n_heads = self.config.num_attention_heads
        self.n_kv = self.config.num_key_value_heads
        self.groups = self.n_heads // self.n_kv
        self.head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.device = self.embed.weight.device
        self.dtype = self.embed.weight.dtype

    def new_cache(self, batch, max_len, window=0, window_mode="recent", seed=0):
        cache = KVCache(self.config, batch, max_len, self.device, self.dtype)
        if window:
            cache.enable_window(window, self.n_heads, window_mode, seed)
        return cache

    @torch.inference_mode()
    def forward(self, input_ids, cache, track=False, all_logits=False):
        """Run `input_ids` ([batch, n]) on top of whatever `cache` already holds.

        Prefill and decode are the same call: prefill has n = prompt length on an empty cache,
        decode has n = 1. Returns logits for the last position ([batch, vocab]) or, with
        all_logits=True, for every new position ([batch, n, vocab]).

        track=True records what eviction policies need: attention mass received per slot
        (cache.score) and, if the cache has a window, the attention rows of the newest queries.
        """
        batch, n = input_ids.shape
        start, end = cache.length, cache.length + n
        if end > cache.max_len:
            raise ValueError(f"cache overflow: need {end} slots, have {cache.max_len}")

        h = self.embed(input_ids)
        positions = torch.arange(cache.pos, cache.pos + n, device=self.device).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary(h, position_ids=positions)

        # New query i is written to slot start+i and may attend to slots 0..start+i. A single
        # query sees every slot, so decode needs no causal mask.
        causal = None
        if n > 1:
            slots = torch.arange(end, device=self.device)
            blocked = slots[None, :] > slots[start:, None]  # [n, end]
            causal = torch.zeros(n, end, device=self.device, dtype=self.dtype).masked_fill(blocked, float("-inf"))

        win_rows = win_slots = None
        if track and cache.window is not None:
            rows, slots = cache.window_plan(n)
            if rows:
                win_rows = torch.tensor(rows, device=self.device)
                win_slots = torch.tensor(slots, device=self.device)

        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(batch, n, self.n_kv, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(batch, n, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Write first, then attend: a token attends to itself.
            cache.k[li, :, :, start:end] = k
            cache.v[li, :, :, start:end] = v
            # Grouped-query attention without copying K/V once per query head: the query heads
            # that share a kv head are stacked along the query axis (rows ordered group, then
            # position), so each kv head's keys are read in place by one batched matmul.
            keys = cache.k[li, :, :, :end]  # [batch, kv, end, head_dim]
            values = cache.v[li, :, :, :end]
            qg = q.reshape(batch, self.n_kv, self.groups * n, self.head_dim)

            scores = torch.matmul(qg, keys.transpose(-1, -2)) * self.scale  # [batch, kv, groups*n, end]
            if causal is not None:
                scores.view(batch, self.n_kv, self.groups, n, end).add_(causal)
            if cache.has_evicted:
                evicted = ~cache.keep[li, :, :, :end]  # [batch, kv, end]
                scores.masked_fill_(evicted[:, :, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

            if track:
                # Attention mass each slot received, summed over the new queries and over the
                # query heads that share a kv head.
                cache.score[li, :, :, :end] += weights.sum(dim=2, dtype=torch.float32)
                if win_rows is not None:
                    rows = weights.reshape(batch, self.n_heads, n, end)[:, :, win_rows, :]
                    cache.window[li][:, :, win_slots, :end] = rows

            out = torch.matmul(weights, values).reshape(batch, self.n_heads, n, self.head_dim)
            out = out.transpose(1, 2).reshape(batch, n, -1)
            h = h + attn.o_proj(out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        cache.length = end
        cache.pos += n
        h = self.final_norm(h if all_logits else h[:, -1:])
        logits = self.lm_head(h)
        return logits if all_logits else logits[:, -1]

    @torch.inference_mode()
    def greedy(self, input_ids, max_new, suppress_token=None, cache=None, track=False, on_step=None):
        """Greedy decode. `suppress_token` (an id or a list of ids, e.g. every EOS id) is never
        picked, matching generate(min_new_tokens=max_new). `on_step(cache, step)` runs after each token is chosen
        and before it is fed back, which is where an eviction policy can hook in.
        Returns ([batch, max_new] token ids, cache).
        """
        batch, prompt_len = input_ids.shape
        if cache is None:
            cache = self.new_cache(batch, prompt_len + max_new)
        logits = self.forward(input_ids, cache, track=track)
        out = []
        for step in range(max_new):
            if suppress_token is not None:
                logits[:, suppress_token] = float("-inf")
            next_id = logits.argmax(dim=-1)
            out.append(next_id)
            if step == max_new - 1:
                break
            if on_step is not None:
                on_step(cache, step)
            logits = self.forward(next_id.unsqueeze(1), cache, track=track)
        return torch.stack(out, dim=1), cache
