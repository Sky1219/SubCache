"""Bundle reef/ into one self-contained Colab notebook: python reef/build_colab.py

Same pattern as scripts/build_colab.py: every source file is written with %%writefile, so
uploading reef/reef_colab.ipynb is all Colab needs. The notebook runs the tests, the correctness
gate on the GPU, and the int8-weight and global-layers-only sweeps, then downloads their JSONL
(plus the gate log) for reef/results/. Re-run this after changing any source file.
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = [
    "reef/__init__.py",
    "reef/engine.py",
    "reef/policies.py",
    "reef/data.py",
    "reef/quant.py",
    "reef/run.py",
    "reef/tests/test_policies.py",
    "reef/tests/test_gate.py",
]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


md("# Reef: streaming KV eviction on TinyStories-28M (int8 and global-only sweeps)\n\n"
   "Runtime → Change runtime type → **T4 GPU** (any GPU works), then Runtime → Run all.\n\n"
   "Order: write the source files, run the policy unit tests, run the correctness gate **on the GPU**, "
   "then the two sweeps. The gate cell raises if anything fails; no results from a failed gate.")

code("import torch, transformers\n"
     "assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'\n"
     "print(torch.cuda.get_device_name(0), '| torch', torch.__version__, '| transformers', transformers.__version__)\n"
     "!mkdir -p reef/tests reef/results")

md("## 1. Source files")
for path in FILES:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        code(f"%%writefile {path}\n" + f.read())

md("## 2. Policy unit tests (no model, about a minute)")
code("!python reef/tests/test_policies.py")

md("## 3. Correctness gate on the GPU (a few minutes)\n\n"
   "Engine vs Hugging Face's GPT-Neo forward: full cache, every policy under streaming eviction "
   "(HF with per-head masks), the sweep's metric path, and the int8 fake-quant. "
   "Logit differences should be ~1e-5; the threshold is 1e-3.")
code("!python reef/tests/test_gate.py --device cuda 2>&1 | grep -v -i 'warn' | tee reef/results/gate_cuda.log\n"
     "log = open('reef/results/gate_cuda.log').read()\n"
     "assert 'gate passed' in log and 'Traceback' not in log, 'GATE FAILED: stop here and send gate_cuda.log'")

md("## 4. Smoke run (one batch, about a minute)\n\nChecks the sweep runs on the GPU. Its output is not used.")
code("!python -m reef.run --grid int8 --quant int8 --device cuda --batch 32 --limit-batches 1 --out smoke.jsonl")

md("## 5. Sweeps\n\nBoth resume where they stopped if the runtime restarts (re-run the cell). "
   "`int8`: simulated int8 weights, 5 policies, sinks + recent window. "
   "`global`: the budget applies to global layers only, local layers keep their 256-token window.")
code("!python -m reef.run --grid int8 --quant int8 --device cuda --batch 32 --out reef/results/int8.jsonl")
code("!python -m reef.run --grid global --evict-layers global --device cuda --batch 32 --out reef/results/global.jsonl")

md("## 6. Download\n\nColab deletes files when the runtime ends. Put the three files in `SubCache/reef/results/`, "
   "then run `python -m reef.reproduce --skip-sweeps` locally.")
code("!wc -l reef/results/*.jsonl\n"
     "!cd reef/results && zip -q ../../reef_colab_results.zip int8.jsonl global.jsonl gate_cuda.log\n"
     "from google.colab import files\n"
     "files.download('reef_colab_results.zip')")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for cell in nb["cells"]:
    cell["source"] = cell["source"].splitlines(keepends=True)
out = os.path.join(ROOT, "reef", "reef_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), f"({len(cells)} cells)")
