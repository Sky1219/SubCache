"""Write the job list for the unattended batch: python scripts/make_batch.py > jobs.txt

One job per line, `id|arguments to python`. Jobs are listed most important first; the two
workers (scripts/batch_worker.sh) each take the next unclaimed job, so if the batch is cut
short the core comparisons are the ones that finished. Speed jobs go to speed_jobs.txt and run
afterwards on one GPU alone, because timings are not clean while another job shares the CPU.
"""

import sys

DS = "deepseek-ai/deepseek-coder-1.3b-instruct"
TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
jobs, speed = [], []


def ev(job_id, out, rest, model=DS):
    jobs.append(f"{job_id}|scripts/run_eviction.py --model {model} --compact --needle-trials 0 --resume "
                f"--out results/batch/{out}.jsonl --tag {job_id} {rest}")


def sp(job_id, rest):
    speed.append(f"{job_id}|scripts/bench_speed.py --model {DS} --out results/batch/speed.jsonl --tag {job_id} {rest}")


BASE = "full recency random h2o gcmi rkv"
DATA = {
    "late": "--data data/agent_late_8k.pt",   # agent sessions, 24k tokens in: most repetitive
    "start": "--data data/agent_8k.pt",       # agent sessions from the first token
    "wiki": "--text wikitext-train",          # prose control, almost no repetition
}
B8 = "--budgets 0.25 0.125 0.0625 0.03125"
B4 = "--budgets 0.5 0.25 0.125 0.0625"
WFL = "--policies wfl --wfl-mode prune"

# 0. Gate on both cards, recorded with the batch.
jobs.append(f"gate_ds|scripts/gate.py --model {DS} --lengths 128 512 1900 3500")
jobs.append(f"gate_tiny|scripts/gate.py --model {TINY}")

# 1. Core comparison at 8,192 tokens, 64 sequences: late agent window first, then controls.
for name in ("late", "start", "wiki"):
    d = f"{DATA[name]} --seq-len 8192 --n-seqs 64 {B8}"
    ev(f"core8k_{name}_base", f"core8k_{name}", f"{d} --policies {BASE}")
    ev(f"core8k_{name}_wfl25", f"core8k_{name}", f"{d} {WFL} --mix 0.25")
    ev(f"core8k_{name}_gcmi128r", f"core8k_{name}", f"{d} --policies gcmi --window 128 --window-mode reservoir")

# 2. Same at 4,096 tokens (cheaper, one more budget step).
for name in ("late", "wiki", "start"):
    d = f"{DATA[name]} --seq-len 4096 --n-seqs 64 {B4}"
    ev(f"core4k_{name}_base", f"core4k_{name}", f"{d} --policies {BASE}")
    ev(f"core4k_{name}_wfl25", f"core4k_{name}", f"{d} {WFL} --mix 0.25")

# 3. Second redundancy weight and R-KV settings on the late agent window.
late8 = f"{DATA['late']} --seq-len 8192 --n-seqs 64 {B8}"
ev("core8k_late_wfl50", "core8k_late", f"{late8} {WFL} --mix 0.5")
ev("core8k_late_rkv_lam001", "core8k_late", f"{late8} --policies rkv --lam 0.01")
ev("core8k_late_rkv_lam05", "core8k_late", f"{late8} --policies rkv --lam 0.5")
ev("core8k_late_rkv_win8", "core8k_late", f"{late8} --policies rkv --window 8")

# 4. Sweeps on the late agent window at 4,096 tokens, two budgets.
late4 = f"{DATA['late']} --seq-len 4096 --n-seqs 64 --budgets 0.25 0.0625"
for rf in ("0.1", "0.25", "0.75"):
    ev(f"rf{rf}_base", "sweep_recent", f"{late4} --recent-frac {rf} --policies full h2o gcmi rkv recency")
    ev(f"rf{rf}_wfl25", "sweep_recent", f"{late4} --recent-frac {rf} {WFL} --mix 0.25")
for mix in ("0.1", "1.0"):
    ev(f"wfl_mix{mix}", "sweep_wfl", f"{late4} {WFL} --mix {mix}")
for power in ("2", "4"):
    ev(f"wfl_pow{power}", "sweep_wfl", f"{late4} {WFL} --mix 0.25 --sim-power {power}")
for step in ("1", "32"):
    ev(f"wfl_step{step}", "sweep_wfl", f"{late4} {WFL} --mix 0.25 --prune-step {step}")
ev("wfl_ref", "sweep_wfl", f"{late4} --policies full h2o")
for interval in ("16", "256"):
    ev(f"int{interval}_base", "sweep_interval", f"{late4} --interval {interval} --policies full h2o gcmi rkv recency")
    ev(f"int{interval}_wfl25", "sweep_interval", f"{late4} --interval {interval} {WFL} --mix 0.25")
for sink in ("0", "1", "16"):
    ev(f"sink{sink}", "sweep_sink", f"{late4} --sink {sink} --policies full h2o recency")
ev("lam_sweep", "sweep_rkv", f"{late4} --policies full h2o rkv --lam 0.05")
ev("lam_sweep2", "sweep_rkv", f"{late4} --policies rkv --lam 0.2")

# 5. Long sessions: 16,000 tokens from the start, 32 sequences.
long16 = "--data data/agent_16k.pt --seq-len 16000 --n-seqs 32 --budgets 0.125 0.0625 0.03125"
ev("long16k_base", "long16k", f"{long16} --policies {BASE}")
ev("long16k_wfl25", "long16k", f"{long16} {WFL} --mix 0.25")

# 6. TinyLlama on WikiText: R-KV, and how the cheaper wfl algorithms compare with exact greedy.
tiny = "--seq-len 1024 --n-seqs 32 --budgets 0.5 0.25 0.125 0.0625"
ev("tiny_base", "tiny", f"{tiny} --policies full recency h2o gcmi rkv", TINY)
ev("tiny_wfl_select", "tiny", f"{tiny} --policies wfl --mix 0.25", TINY)
ev("tiny_wfl_prune", "tiny", f"{tiny} {WFL} --mix 0.25", TINY)
ev("tiny_wfl_prune1", "tiny", f"{tiny} {WFL} --mix 0.25 --prune-step 1", TINY)
for eps in ("0.5", "0.1", "0.01"):
    ev(f"tiny_wfl_eps{eps}", "tiny", f"{tiny} --policies wfl --mix 0.25 --eps {eps}", TINY)

# 7. More sequences for the headline comparison, if time remains.
ev("core8k_late_base_n96", "core8k_late_n96", f"{DATA['late']} --seq-len 8192 --n-seqs 96 {B8} --policies full h2o rkv gcmi")
ev("core8k_late_wfl25_n96", "core8k_late_n96", f"{DATA['late']} --seq-len 8192 --n-seqs 96 {B8} {WFL} --mix 0.25")

# Speed, run alone afterwards. Three repeats of the main table to see run-to-run variation.
late = "--data data/agent_late_8k.pt"
for rep in ("1", "2", "3"):
    sp(f"speed8k_rep{rep}", f"{late} --context 8192 --new 256 --interval 128 --batches 1 2 3 4 6 8 12 16 24 32 "
                            f"--budgets 0.5 0.25 0.125 0.0625 --policies full h2o rkv")
sp("speed4k", f"{late} --context 4096 --new 256 --interval 64 --batches 1 2 4 8 12 16 24 32 48 "
              f"--budgets 0.5 0.25 0.125 0.0625 --policies full h2o rkv")
sp("speed8k_wfl", f"{late} --context 8192 --new 256 --interval 128 --batches 1 2 4 8 "
                  f"--budgets 0.125 0.0625 --policies wfl --wfl-mode prune --mix 0.25")
sp("speed4k_wfl", f"{late} --context 4096 --new 256 --interval 64 --batches 1 2 4 8 16 "
                  f"--budgets 0.25 0.125 0.0625 --policies wfl --wfl-mode prune --mix 0.25")
sp("speed8k_int64", f"{late} --context 8192 --new 256 --interval 64 --batches 1 2 3 4 6 8 12 16 24 32 "
                    f"--budgets 0.25 0.125 0.0625 0.03125 --policies full h2o rkv")
sp("speed16k", "--data data/agent_16k.pt --context 16000 --new 256 --interval 128 --batches 1 2 3 4 6 8 12 "
               "--budgets 0.25 0.125 0.0625 --policies full h2o")

ids = [line.split("|")[0] for line in jobs + speed]
assert len(ids) == len(set(ids)), "duplicate job ids"
with open(sys.argv[1] if len(sys.argv) > 1 else "jobs.txt", "w", newline="\n") as f:
    f.write("\n".join(jobs) + "\n")
with open(sys.argv[2] if len(sys.argv) > 2 else "speed_jobs.txt", "w", newline="\n") as f:
    f.write("\n".join(speed) + "\n")
print(f"{len(jobs)} accuracy jobs, {len(speed)} speed jobs")
