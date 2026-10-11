#!/bin/bash
# One worker of the unattended batch: takes the next unclaimed job from a job file until none
# are left. Run one worker per GPU, each in its own tmux session.
#
#   scripts/batch_worker.sh <cuda index> [job file] [wait-for-file ...]
#
# A job is claimed by creating state/claim/<id> (mkdir is atomic, so two workers never take the
# same job) and marked finished with state/done/<id>. A job that fails keeps its claim and is
# not retried; delete state/claim/<id> and start a worker again to rerun it. run_eviction.py is
# called with --resume, so a rerun continues after the rows it already wrote.
# Extra arguments are files to wait for before starting (used to run the speed jobs only after
# the accuracy workers have written their DONE markers).
#
# Runs from the repository root, wherever it is checked out. Machine-specific choices come from
# the environment: SUBCACHE_PYTHON (the interpreter, default `python`), and the usual
# HF_HUB_CACHE / HF_DATASETS_CACHE if the Hugging Face caches should live somewhere else.

GPU=$1
JOBS=${2:-jobs.txt}
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/batch results/batch state/claim state/done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
PY=${SUBCACHE_PYTHON:-python}
LOG=logs/batch/worker_${GPU}_$(basename "$JOBS" .txt).log

for marker in "$@"; do
  echo "$(date '+%F %T') waiting for $marker" >> "$LOG"
  while [ ! -e "$marker" ]; do sleep 60; done
done

while IFS='|' read -r id cmd; do
  [ -z "$id" ] && continue
  [ -e "state/done/$id" ] && continue
  mkdir "state/claim/$id" 2>/dev/null || continue
  echo "$(date '+%F %T') start $id  load $(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
  timeout 16h $PY $cmd > "logs/batch/$id.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date '+%F %T') end   $id  rc=$rc" >> "$LOG"
  [ $rc -eq 0 ] && touch "state/done/$id" && rmdir "state/claim/$id"
done < "$JOBS"

echo "$(date '+%F %T') WORKER_DONE" >> "$LOG"
touch "state/worker_${GPU}_$(basename "$JOBS" .txt).finished"
