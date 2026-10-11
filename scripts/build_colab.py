"""Bundle the repo into one self-contained Colab notebook: python scripts/build_colab.py

The notebook writes every source file with %%writefile, so uploading the single .ipynb is all
Colab needs. Re-run this after changing any source file.
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = [
    "subcache/__init__.py",
    "subcache/engine.py",
    "subcache/policies.py",
    "subcache/evals.py",
    "scripts/gate.py",
    "scripts/run_eviction.py",
    "scripts/analyze.py",
    "tests/test_policies.py",
]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


md("# SubCache: submodular KV-cache eviction\n\n"
   "Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all.\n\n"
   "Order: write the source files, unit-test the selection code, gate the engine against "
   "Hugging Face `generate`, then run the eviction comparison. Stop if the gate reports a real mismatch.")

code("import torch, transformers\n"
     "assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'\n"
     "print(torch.cuda.get_device_name(0), '| torch', torch.__version__, '| transformers', transformers.__version__)\n"
     "!mkdir -p subcache scripts tests results")

md("## 1. Source files")
for path in FILES:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        code(f"%%writefile {path}\n" + f.read())

md("## 2. Unit tests for the selection code (no model, a few seconds)")
code("!python tests/test_policies.py")

md("## 3. Correctness gate: engine vs `generate`, fp16, 128 new tokens, prompts up to 1,900 tokens\n\n"
   "Pass = every line says `MATCH`, or `near-tie` with a top-2 gap under 0.05. "
   "A line saying `MISMATCH` is a real bug: stop and send me the output.")
code("!python scripts/gate.py 2>&1 | grep -v -i 'max_new_tokens\\|Token indices'")

md("## 4. Quick eviction run (about 2–3 minutes)\n\nTwo sequences, one budget, one needle trial per depth. "
   "This checks that everything runs on the GPU before the full sweep.")
code("!python scripts/run_eviction.py --n-seqs 2 --budgets 0.25 --needle-trials 1 --tag quick --out results/quick.jsonl")

md("## 5. Full sweep\n\n7 policies × 3 budgets, 8 sequences of 1,024 tokens of WikiText-2, 4 needle trials per depth.\n\n"
   "Columns: `ppl_post` is perplexity on the positions where the cache is over budget (compare policies on this, "
   "against `full` at the same budget). `needle` is passcode-retrieval accuracy. `retained` is the share of recent "
   "queries' attention mass that survived eviction. `sel s/evict` is selection time per eviction.")
code("!python scripts/run_eviction.py --tag main --out results/eviction.jsonl")

md("## 6. Variants\n\nEach appends to its own file. Uncomment the ones you want.")
code("# Value-aware kernel (attention x value-vector norm), submodular policies only\n"
     "# !python scripts/run_eviction.py --policies full gcmi com flqmi --value-aware --tag value_aware --out results/value_aware.jsonl\n\n"
     "# Stochastic greedy instead of exact greedy (compare ppl_post and 'sel s/evict' with the main run)\n"
     "# !python scripts/run_eviction.py --policies com flqmi --eps 0.1 --tag stochastic --out results/stochastic.jsonl\n\n"
     "# Less of the budget reserved for recent tokens, so the policy chooses more\n"
     "# !python scripts/run_eviction.py --recent-frac 0.25 --tag recent25 --out results/recent25.jsonl\n\n"
     "# COM with a weaker / stronger modular (per-token relevance) term\n"
     "# !python scripts/run_eviction.py --policies com --eta 0.0 --tag eta0 --out results/eta.jsonl\n"
     "# !python scripts/run_eviction.py --policies com --eta 4.0 --tag eta4 --out results/eta.jsonl")

md("## 7. Results table")
code("import glob, json, pandas as pd\n"
     "rows = [json.loads(line) for f in sorted(glob.glob('results/*.jsonl')) for line in open(f)]\n"
     "df = pd.DataFrame([{**{k: v for k, v in r.items() if k != 'settings'}, 'tag': r['settings']['tag'],\n"
     "                    'gpu': r['settings']['gpu']} for r in rows])\n"
     "cols = [c for c in ['tag', 'policy', 'budget', 'ppl', 'ppl_post', 'needle_acc', 'retained_mean', 'retained_std',\n"
     "                    'select_s_total', 'evictions', 'wall_s'] if c in df]\n"
     "display(df[cols].round(3))\n"
     "main = df[df.tag == 'main'].drop_duplicates(['policy', 'budget'], keep='last')\n"
     "if len(main):\n"
     "    print('ppl_post by policy and budget (main run)')\n"
     "    display(main.pivot(index='policy', columns='budget', values='ppl_post').round(3))\n"
     "    if 'needle_acc' in main:\n"
     "        print('needle accuracy by policy and budget (main run)')\n"
     "        display(main.pivot(index='policy', columns='budget', values='needle_acc').round(2))")

md("## 8. Save the results\n\nColab deletes files when the runtime ends. Download the results and send them over.")
code("!zip -q -r results.zip results\n"
     "from google.colab import files\n"
     "files.download('results.zip')")

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
out = os.path.join(ROOT, "subcache_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), f"({len(cells)} cells)")
