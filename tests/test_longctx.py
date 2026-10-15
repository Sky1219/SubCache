"""Checks for the task-benchmark code (subcache/longctx.py, subcache/benchmarks.py,
scripts/run_bench.py) on a tiny random Llama on the CPU: run with `python tests/test_longctx.py`.
No download is needed."""

import os
import shlex
import sys
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_bench import build_parser, done_keys, read_records, result_settings  # noqa: E402
from subcache.benchmarks import last_boxed, longbench_score, ruler_score  # noqa: E402
from subcache.engine import Engine  # noqa: E402
from subcache.longctx import _pick, evict_rows, gather_rows, keep_mask, prefill  # noqa: E402
from subcache.policies import evict, select_keep  # noqa: E402


def tiny_engine():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=101, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Engine(LlamaForCausalLM(cfg).eval())


POLICY_KW = {
    "recency": {}, "h2o": {}, "rkv": dict(lam=0.1),
    "wfl": dict(mix=1.0, wfl_mode="prune", prune_step=8, pool=2.0, prune_rounds=64),
}


def check_gather_rows():
    """Each packed row must decode exactly like the full cache with that row's mask applied:
    a misordered gather or a wrong position would change every compressed score."""
    engine = tiny_engine()
    ids = torch.randint(1, 100, (1, 40))
    logits, cache = prefill(engine, ids, extra=8, window=8, chunk=16)
    n, cap = cache.length, 16
    masks = [keep_mask(cache, p, cap, recent_frac=0.5, sink=4, **kw)[0] for p, kw in POLICY_KW.items()]
    assert len({tuple(m.flatten().tolist()) for m in masks}) > 1  # the policies disagree somewhere
    rows = gather_rows(engine, cache, masks, extra=8)
    assert rows.length == int(masks[0].sum(-1).max()) and rows.pos == n

    refs = []
    for m in masks:
        ref = engine.new_cache(1, n + 8)
        ref.k[:, :, :, :n], ref.v[:, :, :, :n] = cache.k[:, :, :, :n], cache.v[:, :, :, :n]
        ref.keep[..., :n] = m
        ref.length, ref.pos, ref.has_evicted = n, cache.pos, True
        refs.append(ref)
    for t in (7, 23, 55, 91):
        packed = engine.forward(torch.full((len(masks), 1), t), rows)
        for b, ref in enumerate(refs):
            masked = engine.forward(torch.tensor([[t]]), ref)
            assert torch.allclose(packed[b], masked[0], atol=1e-4), (list(POLICY_KW)[b], t)
    print("gather_rows: every packed row decodes like the full cache under its own mask")


def check_row_views():
    engine = tiny_engine()
    ids = torch.randint(1, 100, (2, 40))
    _, cache = prefill(engine, ids, extra=8, window=8, chunk=40)
    k0 = cache.k.clone()
    try:
        evict(cache.row(1), "h2o", 20, compact=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("compacting a row view must fail")
    assert torch.equal(cache.k, k0) and cache.keep[..., :40].all() and cache.length == 40
    try:
        evict_rows(cache, [{"policy": "full", "capacity": 0}, {"policy": "h2o", "capacity": 20, "compact": True}])
    except RuntimeError:
        pass
    else:
        raise AssertionError("evict_rows with compact must fail")

    # Masked eviction through a view changes only that row.
    evict_rows(cache, [{"policy": "full", "capacity": 0}, {"policy": "h2o", "capacity": 20}])
    assert cache.has_evicted and cache.keep[:, 0, :, :40].all()
    assert (cache.keep[:, 1, :, :40].sum(-1) == 20).all()
    print("row views: compaction refused, masked eviction touches only its own row")


def check_pool():
    engine = tiny_engine()
    _, cache = prefill(engine, torch.randint(1, 100, (1, 60)), extra=0, window=8, chunk=60)
    for pool in (2.0, 1.0):
        m, _ = select_keep(cache, "wfl", 30, mix=1.0, wfl_mode="prune", pool=pool)
        assert (m.sum(-1) == 30).all(), pool
    try:
        select_keep(cache, "wfl", 30, mix=1.0, wfl_mode="prune", pool=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("pool < 1 must be rejected")
    print("pool: >= 1 keeps exactly the capacity, < 1 is rejected")


def check_pick():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]]))
    assert _pick(logits, 0.0, 1.0, None).tolist() == [0, 2]
    # top_p 0.85 keeps {0.6, 0.3}; renormalized cumulative mass is [2/3, 1].
    for u, want in ((0.05, 0), (0.6, 0), (0.7, 1), (0.999, 1)):
        got = _pick(logits[:1], 1.0, 0.85, torch.tensor([u]))
        assert got.item() == want, (u, got)
    # Same u, same distribution: the same token on every row (common random numbers).
    same = logits[:1].expand(5, -1)
    for u in torch.linspace(0.01, 0.99, 20):
        assert len(set(_pick(same, 0.7, 0.95, u.view(1)).tolist())) == 1
    # The tail outside top_p is never drawn.
    for u in torch.linspace(0.0, 1.0, 101):
        assert _pick(logits[:1], 1.0, 0.5, u.view(1)).item() == 0
    print("_pick: greedy, top-p nucleus and shared random numbers")


def check_scoring():
    assert last_boxed(r"so the answer is \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{1} then \boxed{ {2} }") == " {2} "
    assert last_boxed(r"\boxed {7}") == "7"
    assert last_boxed(r"\boxed 5 and {x}") is None
    assert last_boxed(r"\boxed{unclosed") is None and last_boxed("no box") is None
    assert ruler_score("The numbers are 123 and 456.", ["123", "456"]) == 100
    assert ruler_score("Only 123.", ["123", "456"]) == 50
    assert longbench_score("hotpotqa", "Paris", ["Paris", "London"]) == 100
    assert longbench_score("hotpotqa", "Berlin", ["Paris"]) == 0
    print("scoring: last_boxed, RULER string-match-all, LongBench QA F1")


def check_resume():
    """Records carry their settings; resume reuses only matching ones and survives a cut-off
    last line."""
    parse = lambda c: build_parser().parse_args(shlex.split(c))  # noqa: E731
    base = "--bench ruler --model m --out x.jsonl"
    s1, c1 = result_settings(parse(base), [1])
    _, c1b = result_settings(parse(base + " --limit 3 --lengths 4096 --tasks vt"), [1])
    _, c2 = result_settings(parse(base + " --mix 0.5"), [1])
    assert c1 == c1b and c1 != c2 and "limit" not in s1 and s1["mix"] == 1.0

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.jsonl")
        with open(path, "w", newline="\n") as f:
            f.write('{"task": "vt", "length": 8192, "idx": 0, "config": "%s"}\n' % c1)
            f.write('{"task": "vt", "length": 8192, "idx": 1, "config": "%s"}\n' % c2)
            f.write('{"task": "vt", "length": 8192, "idx": 2, "con')  # killed mid-write
        assert done_keys(path, c1) == {("vt", 8192, 0)}
        assert len(read_records(path)) == 2
        with open(path, "rb") as f:
            assert f.read().endswith(b"}\n")  # the partial line is gone, appends start clean

    # The commands in scripts/bench_all.sh map to the configs stored in results/bench, so a
    # rerun of those commands resumes them instead of starting over.
    stored = {}
    for name in ("ruler_llama31", "longbench_llama31", "math500_r1qwen15"):
        recs = read_records(os.path.join(ROOT, "results", "bench", name + ".jsonl"))
        assert len({r["config"] for r in recs}) == 1, name
        stored[name] = recs[0]["config"]
    llama, eos = "--model meta-llama/Llama-3.1-8B-Instruct", [128001, 128008, 128009]
    for name, cmd, e in (
            ("ruler_llama31", f"--bench ruler {llama} --lengths 8192 --limit 10", eos),
            ("longbench_llama31", f"--bench longbench {llama} --max-len 16000 --limit 20 --tasks qasper lcc", eos),
            ("longbench_llama31", f"--bench longbench {llama} --max-len 16000 --limit 10 --tasks gov_report", eos),
            ("math500_r1qwen15", "--bench math --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --levels 1 2 "
                                 "--limit 20 --budgets 256 512 1024 2048 --max-new 4096", [151643])):
        assert result_settings(parse(f"{cmd} --out o.jsonl"), e)[1] == stored[name], name
    print("resume: settings hash, other configs not reused, cut-off last line dropped, stored runs resumable")


if __name__ == "__main__":
    check_gather_rows()
    check_row_views()
    check_pool()
    check_pick()
    check_scoring()
    check_resume()
