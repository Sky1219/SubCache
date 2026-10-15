#!/bin/bash
# Benchmark runs on one GPU. Two queues that can share a 48 GB card: `long` (RULER-style, then
# LongBench, Llama-3.1-8B-Instruct) and `math` (MATH-500, DeepSeek-R1-Distill-Qwen-1.5B). Every
# run_bench.py call resumes from its --out file, so rerunning continues where it stopped.
# Defaults are the quick configuration; raise the sizes with the environment variables.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/bench_all.sh long|math
# (on a multi-GPU box also export CUDA_DEVICE_ORDER=PCI_BUS_ID so the index means what nvidia-smi says)
set -u
cd "$(dirname "$0")/.."
PY=${SUBCACHE_PYTHON:-python}
LLAMA=meta-llama/Llama-3.1-8B-Instruct
R1=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
mkdir -p results/bench logs/bench
export PYTHONUNBUFFERED=1

# Before benchmarking a model, the engine must match HF generate on it exactly in fp32 (a code
# bug shows up there). bf16, the dtype the benchmarks run in, is gated too, but only non-finite
# logits (exit 2) stop the queue: two correct bf16 paths can land a few rounding steps apart and
# pick different tokens where the top two logits are close (seen on R1-Distill-Qwen-1.5B).
gate() {
    $PY scripts/gate.py --model "$1" --dtype float32 --lengths 128 512 1900 3500 > "logs/bench/gate_$2_fp32.log" 2>&1 \
        || { echo "fp32 gate failed for $1, see logs/bench/gate_$2_fp32.log"; exit 1; }
    $PY scripts/gate.py --model "$1" --dtype bfloat16 --lengths 128 512 1900 3500 > "logs/bench/gate_$2_bf16.log" 2>&1
    [ $? -ne 2 ] || { echo "non-finite bf16 logits for $1, see logs/bench/gate_$2_bf16.log"; exit 1; }
}

if [ "$1" = long ]; then
    gate $LLAMA llama31
    $PY scripts/run_bench.py --bench ruler --model $LLAMA --lengths ${RULER_LENGTHS:-8192} \
        --limit ${RULER_N:-10} --out results/bench/ruler_llama31.jsonl >> logs/bench/ruler_llama31.log 2>&1
    $PY scripts/run_bench.py --bench longbench --model $LLAMA --max-len ${LB_MAXLEN:-16000} --limit ${LB_N:-20} \
        --tasks ${LB_TASKS:-qasper multifieldqa_en hotpotqa lcc} \
        --out results/bench/longbench_llama31.jsonl >> logs/bench/longbench_llama31.log 2>&1
    $PY scripts/run_bench.py --bench longbench --model $LLAMA --max-len ${LB_MAXLEN:-16000} --limit ${LB_SUM_N:-10} \
        --tasks gov_report --out results/bench/longbench_llama31.jsonl >> logs/bench/longbench_llama31.log 2>&1
fi
if [ "$1" = math ]; then
    gate $R1 r1qwen15
    $PY scripts/run_bench.py --bench math --model $R1 --levels ${MATH_LEVELS:-1 2} --limit ${MATH_N:-20} \
        --budgets 256 512 1024 2048 --max-new ${MATH_MAX_NEW:-4096} \
        --out results/bench/math500_r1qwen15.jsonl >> logs/bench/math500_r1qwen15.log 2>&1
fi
