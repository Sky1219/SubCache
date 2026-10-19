"""Reproduce every number, table and figure in reef/WRITEUP.md, from the SubCache root:

    python -m reef.reproduce

Steps: correctness gate and policy tests (stop on failure) -> fp32 sweep (two parallel processes)
-> int8-weight sweep -> global-layers-only sensitivity sweep -> the TIU team's non-permanent
simulation (reef/sim.py) -> bootstrap analysis, figures and the hardware-cost table, plus the
CPU-vs-GPU check if reef/results/gpu holds a run of reef/run_gpu.sh.
On a 16-thread laptop CPU the sweeps take roughly 3-4 hours; they resume where they stopped if
re-run. --skip-sweeps re-does only the analysis from the JSONL already in reef/results/.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join("reef", "results")


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="torch threads per sweep process")
    args = ap.parse_args()
    py = sys.executable

    run([py, "reef/tests/test_policies.py"])
    run([py, "reef/tests/test_gate.py"])
    run([py, "reef/tests/test_sim.py"])

    if not args.skip_sweeps:
        os.makedirs(os.path.join(ROOT, RESULTS), exist_ok=True)
        procs = []
        for k in range(2):
            cmd = [py, "-m", "reef.run", "--grid", "main", "--part", f"{k}/2", "--threads", str(args.threads),
                   "--out", os.path.join(RESULTS, f"main.part{k}.jsonl")]
            print("$", " ".join(cmd), flush=True)
            procs.append(subprocess.Popen(cmd, cwd=ROOT))
        if any(p.wait() for p in procs):
            sys.exit("fp32 sweep failed")
        run([py, "-m", "reef.run", "--grid", "int8", "--quant", "int8", "--threads", str(2 * args.threads),
             "--out", os.path.join(RESULTS, "int8.jsonl")])
        run([py, "-m", "reef.run", "--grid", "global", "--evict-layers", "global", "--threads", str(2 * args.threads),
             "--out", os.path.join(RESULTS, "global.jsonl")])

        run([py, "-m", "reef.sim", "--out", os.path.join(RESULTS, "sim.jsonl")])

    run([py, "-m", "reef.hw_cost"])
    run([py, "-m", "reef.analyze"])
    if os.path.isdir(os.path.join(ROOT, RESULTS, "gpu")):  # written by reef/run_gpu.sh, copied back
        run([py, "-m", "reef.compare_devices"])


if __name__ == "__main__":
    main()
