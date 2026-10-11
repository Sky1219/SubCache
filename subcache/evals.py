"""Evaluations for eviction policies: streaming perplexity and needle retrieval.

Both feed text in chunks of `interval` tokens and call the policy after each chunk, so the cache
holds at most capacity + interval live slots per head (eviction "in chunks every N tokens").
"""

import math
import random

import torch
import torch.nn.functional as F

from .policies import evict

FILLER = (
    "The history of the printing press begins in the fifteenth century, when movable type "
    "changed how books were made, copied, and shared across Europe and beyond. "
)


def load_text(source):
    """'wikitext' / 'wikitext-train' (need the `datasets` package), a text file path, or 'builtin'."""
    if source == "builtin":
        return FILLER * 2000
    if source == "wikitext-train":  # the larger split, for runs that need more text
        from datasets import load_dataset

        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
        return "\n".join(r for r in rows if r.strip())[:6_000_000]
    if source == "wikitext":
        from datasets import load_dataset

        try:
            rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        except Exception:  # older `datasets` releases know it by the short name
            rows = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
        return "\n".join(r for r in rows if r.strip())
    with open(source, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _stream(engine, ids, policy, capacity, interval, window, all_logits, evict_kw):
    """Feed ids ([1, T]) in chunks, evicting after each. Yields (start, end, logits)."""
    track = policy != "full"
    evict_kw = dict(evict_kw)
    window_mode = evict_kw.pop("window_mode", "recent")
    # A compacting cache only ever holds capacity + one chunk; a masking cache holds everything.
    slots = capacity + interval + 64 if (track and evict_kw.get("compact")) else ids.shape[1] + 64
    cache = engine.new_cache(1, slots, window=window if track else 0, window_mode=window_mode)
    stats = []
    for start in range(0, ids.shape[1], interval):
        end = min(start + interval, ids.shape[1])
        logits = engine.forward(ids[:, start:end], cache, track=track, all_logits=all_logits)
        yield start, end, logits, cache, stats
        st = evict(cache, policy, capacity, **evict_kw)
        if st:
            stats.append(st)


@torch.inference_mode()
def streaming_nll(engine, ids, policy, capacity, interval=64, window=32, **evict_kw):
    """Teacher-forced next-token NLL for one sequence ids ([1, T]) under an eviction policy.

    Returns (nll [T-1] where nll[i] is the loss predicting token i+1, list of eviction stats).
    """
    losses, stats = [], []
    inputs = ids[:, :-1]
    for start, end, logits, cache, stats in _stream(engine, inputs, policy, capacity, interval, window, True, evict_kw):
        logp = F.log_softmax(logits[0].float(), dim=-1)
        target = ids[0, start + 1:end + 1]
        losses.append(-logp.gather(1, target[:, None]).squeeze(1))
    nll = torch.cat(losses)
    if not torch.isfinite(nll).all():
        raise FloatingPointError("non-finite loss: the model overflows in this dtype (try another model or dtype)")
    return nll, stats


def summarize_stats(stats):
    if not stats:
        return {}
    out = {"evictions": len(stats), "select_s_total": sum(s["select_s"] for s in stats)}
    if any("compact_s" in s for s in stats):
        out["compact_s_total"] = sum(s.get("compact_s", 0.0) for s in stats)
    for key in ("retained_mean", "retained_std", "retained_min"):
        vals = [s[key] for s in stats if key in s]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def perplexity(engine, sequences, policy, capacity, interval=64, window=32, masks=None, **evict_kw):
    """Perplexity over sequences (list of [1, T] id tensors).

    masks: optional list of [T] bool tensors marking the tokens of interest (e.g. the agent's
    own turns in a trajectory). When given, ppl_post_masked and nll_post_masked_per_seq cover
    only those tokens among the positions past capacity.

    ppl      : all positions
    token_nll: the loss on every token of every sequence (a tensor, for loss-by-position plots)
    ppl_post : only positions >= capacity, where the cache is over budget and eviction applies.
               Compare policies on this one; for policy='full' it is the same positions with
               nothing evicted.
    """
    all_nll, post_nll, masked_nll, stats = [], [], [], []
    for i, ids in enumerate(sequences):
        nll, st = streaming_nll(engine, ids, policy, capacity, interval, window, **evict_kw)
        all_nll.append(nll)
        post_nll.append(nll[capacity:])
        if masks is not None:
            target_mask = masks[i].to(nll.device)[1:]  # nll[j] is the loss on token j+1
            masked_nll.append(nll[capacity:][target_mask[capacity:]])
        stats += st
    per_seq = [x.mean().item() for x in post_nll if x.numel()]  # for paired comparisons
    token_nll = torch.stack(all_nll).half().cpu() if len({x.numel() for x in all_nll}) == 1 else None
    all_nll, post_nll = torch.cat(all_nll), torch.cat(post_nll)
    out = {"ppl": math.exp(all_nll.mean().item()), "tokens": all_nll.numel()}
    if post_nll.numel():
        out["ppl_post"] = math.exp(post_nll.mean().item())
        out["nll_post_per_seq"] = per_seq
    if masked_nll:
        out["nll_post_masked_per_seq"] = [x.mean().item() if x.numel() else None for x in masked_nll]
        joined = torch.cat(masked_nll)
        if joined.numel():
            out["ppl_post_masked"] = math.exp(joined.mean().item())
            out["masked_tokens"] = joined.numel()
    out.update(summarize_stats(stats))
    if token_nll is not None:
        out["token_nll"] = token_nll  # [sequences, T-1] loss on every token; the caller saves it
    return out


def needle_prompt(tok, haystack_ids, context_len, depth, code):
    """A chat prompt: haystack text with one needle sentence at `depth` (0..1), then a question.
    Returns ids [1, T]."""
    needle = f" The secret passcode is {code}. Remember it."
    cut = int(depth * context_len)
    before = tok.decode(haystack_ids[:cut], skip_special_tokens=True)
    after = tok.decode(haystack_ids[cut:context_len], skip_special_tokens=True)
    question = "\n\nWhat is the secret passcode mentioned in the text above?"
    messages = [{"role": "user", "content": before + needle + after + question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "The secret passcode is"
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def needle(engine, tok, haystack_ids, policy, capacity, context_len=1024, depths=(0.1, 0.3, 0.5, 0.7, 0.9),
           trials=4, interval=64, window=32, seed=0, **evict_kw):
    """Fraction of trials where the model reproduces a 5-digit passcode hidden in the context.

    The prompt is streamed in with eviction after each chunk, then 8 tokens are decoded greedily
    with no further eviction. Check the policy='full' score first: if the model cannot do the
    task with the whole cache, the test says nothing about eviction.
    """
    rng = random.Random(seed)
    hits, total, by_depth, stats = 0, 0, {}, []
    for depth in depths:
        ok = 0
        for trial in range(trials):
            code = str(rng.randint(10000, 99999))
            offset = rng.randint(0, max(0, len(haystack_ids) - context_len - 1))
            ids = needle_prompt(tok, haystack_ids[offset:offset + context_len], context_len, depth, code)
            ids = ids.to(engine.device)
            for start, end, logits, cache, stats_run in _stream(engine, ids, policy, capacity, interval, window, False, evict_kw):
                pass
            stats += stats_run
            out = []
            for _ in range(8):
                next_id = logits.argmax(dim=-1)
                out.append(next_id.item())
                logits = engine.forward(next_id.unsqueeze(1), cache, track=False)
            ok += code in tok.decode(out).replace(",", "").replace(" ", "")
        by_depth[depth] = ok / trials
        hits += ok
        total += trials
    out = {"needle_acc": hits / total, "needle_by_depth": by_depth}
    out.update({f"needle_{k}": v for k, v in summarize_stats(stats).items()})
    return out
