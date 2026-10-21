"""Re-plot Fig. 1 (figA_convergence) from the shipped data/EA4_convergence.csv.

2026-09-24 correction (state-of-the-art review): Chen--Shao--Wu (arXiv:2604.11063,
Sec. 5) report the accuracy 1e-10 for "N ~ 60-120", where N is the number of basis
functions of the UPPER spinor component (the lower one has K = N in the balanced
configuration they recommend). In the 2N real-degrees-of-freedom convention of this
figure that operating region is 2N ~ 120-240, not 60-120 as previously drawn.
Only the shaded band, its label and the x-range change; every plotted data point is
read unchanged from data/EA4_convergence.csv (no recomputation).
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TEXTWIDTH_IN = 390.0 / 72.27
MARK = ["o", "s", "D", "^"]

conv = {}
with (PAPER / "data" / "EA4_convergence.csv").open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        x, y = conv.setdefault(row["case"], ([], []))
        x.append(int(row["dof"]))
        y.append(max(float(row["rel_err"]), 1e-16))

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.9))
for (name, (x, y)), mk in zip(conv.items(), MARK):
    ax.semilogy(np.array(x), np.array(y), marker=mk, ms=4.5, lw=1.1, label=name)
# ALLSM operating region (Chen-Shao-Wu 2026): ~1e-10 relative with N ~ 60-120
# basis functions per spinor component, i.e. 2N ~ 120-240 real DOF.
ax.axhspan(3e-11, 3e-10, xmin=0.0, xmax=1.0, color="0.75", alpha=0.35, zorder=0)
ax.axvspan(120, 240, color="0.85", alpha=0.4, zorder=0)
ax.text(180, 1.2e-10, "ALLSM (2026)\n$10^{-10}$, $2N\\approx120$–$240$",
        ha="center", va="center", fontsize=8.2, color="0.25")
ax.set_xlabel("degrees of freedom  $2N$")
ax.set_ylabel(r"relative error in $E-c^2$")
ax.set_ylim(3e-16, 1e-1)
ax.set_xlim(12, 250)
ax.legend(handlelength=1.7, loc="upper right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(PAPER / f"figA_convergence.{ext}")
plt.close(fig)
print("figA_convergence re-plotted")
