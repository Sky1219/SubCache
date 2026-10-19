"""Correctness gate: the engine against Hugging Face's own GPT-Neo forward pass.

Run with `python reef/tests/test_gate.py` from the SubCache root (or with pytest). Downloads
TinyStories-28M and the validation text on first use.

1. Full cache, no eviction: token-by-token engine logits == one HF forward over the story, for
   several stories decoded together as one batch, one of them longer than the 256-token local
   window.
2. Streaming eviction: for every policy family, the engine's logits under eviction == one HF
   forward whose attention is masked so that each (layer, head) query sees exactly the tokens the
   engine still held at that step. The engine stores tokens in reused, scrambled slots while HF
   keeps them in position order with its own position ids, so agreement also shows that evicting
   a GPT-Neo slot needs no position fix-up (positions live in the cached K/V).
"""

import os
import sys

import torch
import transformers.models.gpt_neo.modeling_gpt_neo as neo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from reef.data import load_stories  # noqa: E402
from reef.engine import NeoEngine, load_model  # noqa: E402
from reef.policies import Streaming  # noqa: E402

ATOL = 1e-3  # logits are O(10); fp32 engine vs HF differ by ~1e-5
# Device under test: `python reef/tests/test_gate.py --device cuda` (or REEF_DEVICE=cuda for pytest).
DEVICE = os.environ.get("REEF_DEVICE", "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_cache = {}


def setup():
    if not _cache:
        torch.manual_seed(0)
        tok, model = load_model(device=DEVICE)
        stories = load_stories(tok, n=40, seed=1)
        stories.sort(key=lambda s: len(s["ids"]))
        picked = [stories[0], stories[len(stories) // 2], next(s for s in stories[::-1] if len(s["ids"]) > 300)]
        _cache.update(tok=tok, model=model, engine=NeoEngine(model), stories=picked)
    return _cache["tok"], _cache["model"], _cache["engine"], _cache["stories"]


def hf_logits(model, ids, head_mask=None):
    """One HF forward. head_mask: optional [layers, heads, T, T] bool of allowed (query, key)
    pairs, applied on top of HF's own causal / local-window mask by wrapping _attn."""
    orig = neo.GPTNeoSelfAttention._attn

    def masked_attn(self, query, key, value, attention_mask=None, *args, **kwargs):
        extra = torch.zeros(head_mask.shape[1:], dtype=torch.float32, device=query.device)
        extra.masked_fill_(~head_mask[self.layer_id].to(query.device), torch.finfo(torch.float32).min)
        attention_mask = extra[None] if attention_mask is None else attention_mask + extra[None]
        return orig(self, query, key, value, attention_mask, *args, **kwargs)

    if head_mask is not None:
        neo.GPTNeoSelfAttention._attn = masked_attn
    try:
        with torch.no_grad():
            return model(torch.tensor([ids], device=DEVICE)).logits[0].float().cpu()
    finally:
        neo.GPTNeoSelfAttention._attn = orig


def test_full_cache_matches_hf():
    tok, model, engine, stories = setup()
    T = max(len(s["ids"]) for s in stories)
    ids = torch.full((len(stories), T), 50256)
    for b, s in enumerate(stories):
        ids[b, : len(s["ids"])] = torch.tensor(s["ids"])
    ids = ids.to(DEVICE)
    cache = engine.new_cache(len(stories), T)
    out = []
    for t in range(T):
        h, _, _ = engine.step(ids[:, t], cache)
        out.append(engine.logits(h).cpu())
    out = torch.stack(out, 1)
    for b, s in enumerate(stories):
        n = len(s["ids"])
        ref = hf_logits(model, s["ids"])
        diff = (out[b, :n] - ref).abs().max().item()
        agree = (out[b, :n].argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"  full cache, story len {n}: max |logit diff| {diff:.2e}, top-1 agreement {agree:.4f}")
        assert diff < ATOL, diff
        assert agree == 1.0


def run_streaming(engine, ids, family, budget, sink, recent, param, granularity, evict_layers=None):
    """Decode one story under one policy (rows = configs in a list), recording when each
    (layer, row, head, position) was evicted. Returns logits [rows, T, V] and evicted_at."""
    rows, T = len(budget), len(ids)
    cap = max(budget) + 1 if evict_layers is None else max(max(budget) + 1, engine.window + 1)
    cache = engine.new_cache(rows, cap)
    pol = Streaming(family, cache, engine.local, engine.window, budget, sink, recent, param, granularity, evict_layers)
    big = T + 10
    evicted_at = torch.full((engine.n_layers, rows, engine.n_heads, T), big, dtype=torch.long)
    out, scrambled = [], False
    x = torch.tensor(ids, device=DEVICE)
    for t in range(T):
        h, attns, slot = engine.step(x[t].expand(rows), cache)
        pol.on_load(slot)
        out.append(engine.logits(h).cpu())
        pol.update(attns, t)
        before = cache.pos.clone()
        pol.evict(cache)
        before, pos = before.cpu(), cache.pos.cpu()
        live = pos >= 0
        on = torch.ones(engine.n_layers, dtype=torch.bool) if evict_layers is None else torch.tensor(evict_layers)
        assert bool((live.sum(-1) <= torch.tensor(budget)[None, :, None])[on].all())
        gone = (before >= 0) & (pos < 0)
        idx = gone.nonzero()
        evicted_at[idx[:, 0], idx[:, 1], idx[:, 2], before[gone]] = t
        # Slots stop being in position order once a freed slot is reused.
        p = pos.masked_fill(~live, -1)
        scrambled |= bool(((p[..., 1:] < p[..., :-1]) & (p[..., 1:] >= 0)).any())
    return torch.stack(out, 1), evicted_at, scrambled


def test_streaming_eviction_matches_hf_masks():
    tok, model, engine, stories = setup()
    ids = stories[-1]["ids"]  # > 300 tokens, so local layers also drop out-of-window slots
    T = len(ids)
    q = torch.arange(T)[:, None]
    full = hf_logits(model, ids)
    configs = [
        ("window", [16, 24], [4, 4], [8, 0], None, "head"),
        ("h2o", [16, 24], [4, 4], [8, 0], None, "head"),
        ("tiu", [16, 24], [4, 4], [8, 0], None, "head"),
        ("tiu", [16], [4], [4], None, "layer"),
        ("rw", [16, 24, 32], [4, 4, 4], [8, 0, 0], [8, 16, 32], "head"),
        ("ema", [16, 24], [4, 4], [0, 8], [0.9, 0.99], "layer"),
        ("h2o", [16], [4], [8], None, "head", "global"),
    ]
    for family, budget, sink, recent, param, gran, *where in configs:
        evict = [not loc for loc in engine.local] if where else None
        logits, evicted_at, scrambled = run_streaming(engine, ids, family, budget, sink, recent, param, gran, evict)
        assert scrambled, "test did not exercise slot reuse"
        for r in range(len(budget)):
            # Key p is visible to query q iff p <= q and p was not yet evicted when q ran.
            ev = evicted_at[:, r]  # [layers, heads, T]
            allowed = (q >= torch.arange(T)[None, :]) & (q[None, None] <= ev[:, :, None, :])
            ref = hf_logits(model, ids, allowed)
            diff = (logits[r] - ref).abs().max().item()
            print(f"  {family:6s} {gran:5s} {'global-only ' if where else ''}budget {budget[r]:3d} recent {recent[r]:2d}: max |logit diff| {diff:.2e}")
            assert diff < ATOL, (family, r, diff)
            # Negative control: eviction must actually change the output, or the match is vacuous.
            assert (logits[r] - full).abs().max().item() > 0.1
            if gran == "layer":
                assert bool((ev == ev[:, :1]).all()), "layer granularity must evict the same token in every head"


def test_runner_metrics_on_full_cache():
    """The sweep's metric path, fed a policy that never evicts, must reproduce the reference
    exactly: KL 0, full top-1 agreement, identical NLL and recall sums."""
    from reef.data import recall_targets
    from reef.run import full_lines, reference

    tok, model, engine, stories = setup()
    T = max(len(s["ids"]) for s in stories)
    ids = torch.full((len(stories), T), 50256)
    for b, s in enumerate(stories):
        ids[b, : len(s["ids"])] = torch.tensor(s["ids"])
    lens = torch.tensor([len(s["ids"]) for s in stories])
    recall = [recall_targets(s["ids"], tok) for s in stories]
    ids, lens = ids.to(DEVICE), lens.to(DEVICE)
    ref = reference(engine, ids, lens)
    rows = [dict(budget=16, sink=0, recent=0, param=None)]
    cfg, story, acc, _ = run_group_full_capacity(engine, rows, ids, lens, recall, ref, T)
    want = {b: s for (budget, b, s) in full_lines([16], ids, lens, recall, ref)}
    for r in range(len(story)):
        b = int(story[r])
        assert acc["n"][r].item() == want[b]["n"] > 0
        assert abs(acc["kl"][r].item()) < 1e-6 * want[b]["n"], acc["kl"][r].item()
        assert acc["agree"][r].item() == want[b]["n"]
        for k in ("nll", "correct", "r_nll", "r_correct", "rn"):
            assert abs(acc[k][r].item() - want[b][k]) < 1e-3 * max(1.0, abs(want[b][k])), (k, acc[k][r].item(), want[b][k])
        print(f"  story {b}: {want[b]['n']} positions, {want[b]['rn']} recall tokens, KL {acc['kl'][r].item():.1e}")


def test_int8_fake_quant():
    """Weights land on an int8 grid (<= 255 distinct levels per row, error <= half a step), the
    tied lm_head follows the embedding, and the model still predicts nearly as the fp32 one."""
    from reef.quant import quantize_weights_int8

    tok, model, engine, stories = setup()
    _, qmodel = load_model(device=DEVICE)
    quantize_weights_int8(qmodel)
    pairs = [(model.transformer.wte.weight, qmodel.transformer.wte.weight),
             (model.transformer.h[3].mlp.c_fc.weight, qmodel.transformer.h[3].mlp.c_fc.weight)]
    for w, q in pairs:
        step = w.abs().amax(1, keepdim=True) / 127
        assert bool(((w - q).abs() <= step / 2 + 1e-7).all())
        assert len(torch.unique(q[0])) <= 255
    assert qmodel.lm_head.weight.data_ptr() == qmodel.transformer.wte.weight.data_ptr()
    ids = stories[1]["ids"]
    a, b = hf_logits(model, ids), hf_logits(qmodel, ids)
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"  int8 vs fp32 top-1 agreement {agree:.3f}, max |logit diff| {(a - b).abs().max().item():.2f}")
    assert agree > 0.9


def run_group_full_capacity(engine, rows, ids, lens, recall, ref, T):
    """run_group with family 'full' but scoring from position budget + 1 on: the 'full' policy
    never evicts, so give it budget-sized scoring and a cache big enough for the whole story."""
    import reef.run as run
    orig = run.Streaming

    class NeverEvict(orig):
        def __init__(self, family, cache, *a, **k):
            super().__init__("full", cache, *a, **k)

    orig_new = engine.new_cache
    engine.new_cache = lambda rows_, cap: orig_new(rows_, T + 1)
    run.Streaming = NeverEvict
    try:
        return run.run_group(engine, "full", "head", rows, ids, lens, recall, ref)
    finally:
        run.Streaming = orig
        engine.new_cache = orig_new


if __name__ == "__main__":
    if "--device" in sys.argv:
        DEVICE = sys.argv[sys.argv.index("--device") + 1]
    print("device:", DEVICE)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("gate passed")
