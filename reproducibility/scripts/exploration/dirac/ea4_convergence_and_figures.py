"""E-A4 + Paper-A figures.

(1) E-A4: honest error-vs-DOF convergence of the exact-structure method
    on the SAME cases the 2026 competitor (Chen-Shao-Wu, ALLSM) reports,
    with their published operating points overlaid.
(2) Figure: relativistic critical-screening map s_c(Zalpha) from E-A3
    (reads ea3_critical_screening.csv), showing the 2p fine-structure
    splitting.

DOF convention: total real degrees of freedom = 2*nb (both components).
ALLSM reports ~1e-10 relative with 60-120 DOF for point-nucleus /
Yukawa cases; we quote our relative error vs the 40-digit Sommerfeld
reference (point nucleus) at matched cases.

Outputs: figures/figA_convergence.(pdf|png), figures/figA_screening.(pdf|png)
and results/EA4_convergence.csv.
"""

import csv
import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dirac import C_AU, RadialDiracBasis  # noqa: E402

getcontext().prec = 40
C = C_AU
t0 = time.perf_counter()
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TEXTWIDTH_IN = 390.0 / 72.27
MARK = ["o", "s", "D", "^"]


def exact_dec(nr, kappa, Z):
    cD, ZD = Decimal(repr(C)), Decimal(repr(Z))
    nu = ZD / cD
    g = (Decimal(kappa * kappa) - nu * nu).sqrt()
    d = nu / (Decimal(nr) + g)
    return cD * cD / (Decimal(1) + d * d).sqrt() - cD * cD


# ---------------- E-A4 convergence data --------------------------------
CASES = [("Z=92,  1s$_{1/2}$", 92.0, -1, 0),
         ("Z=92,  2p$_{3/2}$", 92.0, -2, 0),
         ("Z=110, 1s$_{1/2}$", 110.0, -1, 0)]
NBS = [8, 10, 12, 15, 20, 25, 30, 40]

rows = [["case", "nb", "dof", "rel_err"]]
conv = {}
for name, Z, kappa, nr in CASES:
    nu = Z / C
    gam = float(np.sqrt(kappa * kappa - nu * nu))
    exD = exact_dec(nr, kappa, Z)
    xs, ys = [], []
    for nb in NBS:
        B = RadialDiracBasis(nb, kappa, Z, 0.9 * Z, mu=gam, quad="cusp",
                             nquad=nb + 150, cert_tol=1e-5)
        ev = B.solve_naive(shifted=True)
        E = float(ev[0])
        rel = abs(float((Decimal(repr(E)) - exD) / exD))
        xs.append(2 * nb)
        ys.append(max(rel, 1e-16))
        rows.append([name, nb, 2 * nb, f"{rel:.3e}"])
    conv[name] = (np.array(xs), np.array(ys))

with (RES / "EA4_convergence.csv").open("w", newline="") as fh:
    csv.writer(fh).writerows(rows)

# ---------------- Figure A1: convergence vs ALLSM ----------------------
fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.9))
for (name, (x, y)), mk in zip(conv.items(), MARK):
    ax.semilogy(x, y, marker=mk, ms=4.5, lw=1.1, label=name)
# ALLSM published operating band (Chen-Shao-Wu 2026): ~1e-10 rel at
# 60-120 DOF across point-nucleus/Yukawa cases.
ax.axhspan(3e-11, 3e-10, xmin=0.0, xmax=1.0, color="0.75", alpha=0.35, zorder=0)
ax.axvspan(60, 120, color="0.85", alpha=0.4, zorder=0)
ax.text(90, 1.2e-10, "ALLSM (2026)\n$10^{-10}$, 60–120 DOF",
        ha="center", va="center", fontsize=8.2, color="0.25")
ax.set_xlabel("degrees of freedom  $2N$")
ax.set_ylabel(r"relative error in $E-c^2$")
ax.set_ylim(3e-16, 1e-1)
ax.set_xlim(12, 130)
ax.legend(handlelength=1.7, loc="upper right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"figA_convergence.{ext}")
plt.close(fig)
print("figA_convergence done")

# ---------------- Figure A2: critical-screening map --------------------
# E-A3 writes ea3_critical_screening.csv next to this script.  When E-A3 is
# skipped (--skip-slow) that file is absent, so fall back to the copy shipped
# with the manuscript, which carries the same kappa/Zalpha/s_c columns.
csv_path = Path(__file__).with_name("ea3_critical_screening.csv")
if not csv_path.exists():
    shipped = ROOT.parents[1] / "data" / "critical_screening_map.csv"
    if not shipped.exists():
        raise SystemExit(
            "ea3_critical_screening.csv not found and the shipped fallback "
            f"{shipped} is missing; run ea3_critical_screening.py first.")
    csv_path = shipped
    print(f"figA_screening: using shipped map {csv_path.name} (E-A3 skipped)")
data = {}
with csv_path.open() as fh:
    for r in csv.DictReader(fh):
        data.setdefault(int(r["kappa"]), []).append(
            (float(r["Zalpha"]), float(r["s_c"])))

label = {-1: r"$1s_{1/2}$  ($\kappa=-1$)",
         1: r"$2p_{1/2}$  ($\kappa=+1$)",
         -2: r"$2p_{3/2}$  ($\kappa=-2$)"}
fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.0))
for kap, mk in zip((-1, 1, -2), MARK):        # 1s, 2p1/2, 2p3/2 order
    pts = sorted(data[kap])
    za = np.array([p[0] for p in pts])
    sc = np.array([p[1] for p in pts])
    ax.plot(za, sc, marker=mk, ms=4.5, lw=1.2, label=label[kap])
ax.set_xlabel(r"$Z\alpha$")
ax.set_ylabel(r"critical screening  $\lambda_c/Z$")
ax.legend(handlelength=1.7, loc="upper left")
ax.annotate("$2p$ fine-structure\nsplitting", xy=(0.94, 0.272),
            xytext=(0.55, 0.46), fontsize=8.2, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
ax.set_ylim(0.18, 1.55)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"figA_screening.{ext}")
plt.close(fig)
print("figA_screening done")
print(f"\ntotal: {time.perf_counter()-t0:.1f} s")
