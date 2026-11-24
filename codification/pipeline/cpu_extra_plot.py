"""Plot item 1 of cpu_extra.py: majority accuracy against denoising steps per question (n candidates x K).

    python3 cpu_extra_plot.py [extra_1_equal_compute.json] [out.png]

Needs matplotlib (not available on Blackwell; run locally on the JSON). Top row: first n candidates by index; bottom row: mean over
200 random n-subsets (seed 0). Dashed vertical line = 256 steps per question, where first-sample K = 256 and majority-of-4 at
K = 64 sit on the same x.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results" / "iter2" / "cpu" / "extra_1_equal_compute.json"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("extra_1_equal_compute.png")
data = json.loads(src.read_text(encoding="utf-8"))

STYLE = {"K16 (500)": ("#9aa5b1", "o", "-"), "K32 (500)": ("#0072B2", "s", "-"), "K64 (500)": ("#D55E00", "^", "-"),
         "K64 (subset)": ("#D55E00", "^", "--"), "K256 (subset)": ("#009E73", "D", "-")}
TITLE = {"dream-base": "Dream 7B", "llada-base": "LLaDA 8B"}

fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
for col, m in enumerate(("dream-base", "llada-base")):
    for row, (key, name) in enumerate((("acc_first_n", "first n candidates"), ("acc_random_subsets", "mean of 200 random n-subsets"))):
        ax = axes[row][col]
        for label, c in data[m]["curves"].items():
            colr, mk, ls = STYLE[label]
            ax.plot(c["steps_per_question"], [100 * v for v in c[key]], color=colr, marker=mk, ms=4, ls=ls, lw=1.6,
                    label=label)
        ax.axvline(256, color="#555555", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 2560]); ax.set_xticklabels(["16", "64", "256", "1024", "2560"])
        ax.set_title(f"{TITLE[m]}: majority vote, {name}", fontsize=10)
        ax.grid(alpha=0.25)
        if row == 1:
            ax.set_xlabel("denoising steps per question (n candidates x K)")
        if col == 0:
            ax.set_ylabel("GSM8K accuracy (%)")
axes[0][0].legend(fontsize=8, loc="upper left")
fig.suptitle("Equal compute: accuracy against denoising steps per question (exploratory; T = 0.7, N = 10).\n"
             "Dotted line = 256 steps. Steps are not wall-clock time. Subsets: 185 questions (Dream), 192 (LLaDA).", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(dst, dpi=150)
print("wrote", dst)
