#!/bin/bash
# GPU reproduction on a multi-GPU Linux box (used on edmonds: 2x RTX 2080 Ti + RTX A6000).
#
#   REEF_PYTHON=/path/to/python bash reef/run_gpu.sh [out_dir] [gpu_a] [gpu_b] [gpu_c]
#
# From the directory that contains reef/. Runs the policy tests and the correctness gate on
# CUDA (stops if either fails), then the sweeps: fp32 halves on gpu_a and gpu_b; int8,
# global-only and the gold config on gpu_c; then the TIU-team simulation and both KV-codec
# sweeps. Output JSONL and logs go to out_dir (default reef/results_gpu); sweeps resume if re-run.

set -u
OUT=${1:-reef/results_gpu}
GA=${2:-0}; GB=${3:-2}; GC=${4:-1}
PY=${REEF_PYTHON:-python}
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

"$PY" reef/tests/test_policies.py > "$OUT/test_policies.log" 2>&1 || { echo "policy tests failed"; exit 1; }
CUDA_VISIBLE_DEVICES=$GA "$PY" reef/tests/test_gate.py --device cuda > "$OUT/gate_cuda.log" 2>&1
grep -q "gate passed" "$OUT/gate_cuda.log" || { echo "GATE FAILED: see $OUT/gate_cuda.log"; exit 1; }
for t in test_sim test_kvquant; do
    CUDA_VISIBLE_DEVICES=$GA "$PY" reef/tests/$t.py --device cuda > "$OUT/${t}_cuda.log" 2>&1 || { echo "$t FAILED"; exit 1; }
done
echo "gate passed on $(CUDA_VISIBLE_DEVICES=$GA nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"

sweep() {  # gpu, log name, run.py arguments...
    local gpu=$1 name=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m reef.run --device cuda --batch 32 "$@" > "$OUT/$name.log" 2>&1
}
sweep "$GA" main.part0 --grid main --part 0/2 --out "$OUT/main.part0.jsonl" &
sweep "$GB" main.part1 --grid main --part 1/2 --out "$OUT/main.part1.jsonl" &
( sweep "$GC" int8 --grid int8 --quant int8 --out "$OUT/int8.jsonl" &&
  sweep "$GC" global --grid global --evict-layers global --out "$OUT/global.jsonl" &&
  sweep "$GC" gold --grid gold --no-full --out "$OUT/gold.jsonl" ) &
wait
# The TIU team's simulation and the KV Cache Engine codecs (each has its own full-cache reference).
CUDA_VISIBLE_DEVICES=$GC "$PY" -m reef.sim --device cuda --out "$OUT/sim.jsonl" > "$OUT/sim.log" 2>&1 &
sweep "$GA" kvq_G128 --grid kvq --kv-quant cq4p_wht3 --key-group 128 --out "$OUT/kvq_G128.jsonl" &
sweep "$GB" kvq_G32 --grid kvq --kv-quant cq4p_wht3 --key-group 32 --out "$OUT/kvq_G32.jsonl" &
wait
echo "sweeps done"
