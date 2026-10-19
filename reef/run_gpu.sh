#!/bin/bash
# GPU reproduction on a multi-GPU Linux box (used on edmonds: 2x RTX 2080 Ti + RTX A6000).
#
#   REEF_PYTHON=/path/to/python bash reef/run_gpu.sh [out_dir] [gpu_a] [gpu_b] [gpu_c]
#
# From the directory that contains reef/. Runs the policy tests and the correctness gate on
# CUDA (stops if either fails), then the three sweeps in parallel: fp32 halves on gpu_a and
# gpu_b, int8 then global-only on gpu_c. Output JSONL and logs go to out_dir
# (default reef/results_gpu); sweeps resume if re-run.

set -u
OUT=${1:-reef/results_gpu}
GA=${2:-0}; GB=${3:-2}; GC=${4:-1}
PY=${REEF_PYTHON:-python}
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

"$PY" reef/tests/test_policies.py > "$OUT/test_policies.log" 2>&1 || { echo "policy tests failed"; exit 1; }
CUDA_VISIBLE_DEVICES=$GA "$PY" reef/tests/test_gate.py --device cuda > "$OUT/gate_cuda.log" 2>&1
grep -q "gate passed" "$OUT/gate_cuda.log" || { echo "GATE FAILED: see $OUT/gate_cuda.log"; exit 1; }
echo "gate passed on $(CUDA_VISIBLE_DEVICES=$GA nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"

sweep() {  # gpu, log name, run.py arguments...
    local gpu=$1 name=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m reef.run --device cuda --batch 32 "$@" > "$OUT/$name.log" 2>&1
}
sweep "$GA" main.part0 --grid main --part 0/2 --out "$OUT/main.part0.jsonl" &
sweep "$GB" main.part1 --grid main --part 1/2 --out "$OUT/main.part1.jsonl" &
( sweep "$GC" int8 --grid int8 --quant int8 --out "$OUT/int8.jsonl" &&
  sweep "$GC" global --grid global --evict-layers global --out "$OUT/global.jsonl" ) &
wait
echo "sweeps done"
