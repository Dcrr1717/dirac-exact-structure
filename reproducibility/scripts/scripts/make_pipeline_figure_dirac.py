"""Paper A methodology figure: the EXACT-STRUCTURE radial DIRAC pipeline.

Vector output (PDF + PNG), Okabe-Ito colorblind-safe palette, Agg backend.

This is the relativistic counterpart of ``make_pipeline_figure.py``.  That
script draws the NON-RELATIVISTIC Schrodinger/FEAST pipeline (Laguerre
basis + Gauss-Legendre elements + FEAST contour integration) and is
**not** the chain of Paper A; do not reuse it there.  The chain drawn
here is the one implemented in ``src/dirac.py`` and described in
Sec. II of the Paper A manuscript:

    radial Dirac operator h_kappa
      -> cusp-adapted gamma-Laguerre basis (mu = gamma)
      -> exact generalized Gauss-Laguerre assembly
      -> run-time certification C1-C3
      -> mass-shifted pencil H - c^2 S
      -> exactly summed Rayleigh refinement
      -> certified spectrum (point-nucleus table, screened validation,
         critical-screening map)

Every label is traceable to the manuscript and the code: Eqs. (1), (2)
and (4) of Paper A, and the ``RadialDiracBasis`` constructor (basis,
quadrature, C1-C4 certifications), ``_ingap_shifted`` (mass-shifted
pencil) and ``_rayleigh`` (exactly summed refinement) of ``src/dirac.py``.

Geometry: 7.0 x 4.0 inches, laid out in two columns of four stages, so
that it is included at ``\\textwidth`` in a two-column ``figure*`` (the
labels then render at their native ~7 pt; at single-column
``\\linewidth`` they would fall below 4 pt and be unreadable).

Usage
-----
    python scripts/make_pipeline_figure_dirac.py
    python scripts/make_pipeline_figure_dirac.py --out /path/figA_pipeline.pdf

Default outputs: ``figures/pipeline_dirac.pdf`` and ``.png``.  ``--out``
writes an additional copy of the PDF at the given path (used to place the
figure next to the manuscript as ``figA_pipeline.pdf``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Okabe-Ito colorblind-safe palette (same five hues as the Schrodinger
# figure); drawn at reduced alpha so that black label text stays legible.
BLUE, ORANGE, GREEN, VERM, PURPLE = ("#56B4E9", "#E69F00", "#009E73",
                                     "#D55E00", "#CC79A7")
GREY = "#EEEEEE"
ALPHA = 0.55
EN = "–"                            # en dash (matplotlib keeps "--")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

X0, X1, Y0, Y1 = -1.0, 101.0, -1.5, 57.5          # data canvas
FIGW = 7.0
fig, ax = plt.subplots(figsize=(FIGW, FIGW * (Y1 - Y0) / (X1 - X0)))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.axis("off")

ASPECT = (X1 - X0) / (Y1 - Y0)


def box(x, y, w, h, text, fc, fs=7.2, alpha=ALPHA):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.7",
                                fc=fc, ec="k", lw=0.9, alpha=alpha,
                                mutation_aspect=ASPECT))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            linespacing=1.5, zorder=5)


def varrow(x, y_from, y_to, text=""):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.0, color="k"))
    if text:
        ax.text(x + 1.4, (y_from + y_to) / 2, text, fontsize=6.3, ha="left",
                va="center", style="italic", color="0.25")


# ---- geometry: two columns of four stages --------------------------------
BW, BH = 46.0, 11.0
LX, RX = 1.0, 53.0                       # left / right column origins
LC, RC = LX + BW / 2, RX + BW / 2
ROWS = (45.0, 30.0, 15.0, 0.0)           # bottom edge of each row

# ==== left column: operator -> basis -> quadrature -> certification =======
box(LX, ROWS[0], BW, BH,
    r"Radial Dirac operator $h_{\kappa}$   (large $P$, small $Q$)" "\n"
    r"$(V+c^{2})P+c\,\mathcal{O}_{+}Q=EP$,"
    r"$\quad c\,\mathcal{O}_{-}P+(V-c^{2})Q=EQ$" "\n"
    r"$\mathcal{O}_{\pm}=\mp\,\mathrm{d}/\mathrm{d}r+\kappa/r$,"
    r"$\quad V=-Z/r$  (or $-Ze^{-\lambda r}/r$)",
    GREY, alpha=1.0)
varrow(LC, ROWS[0], ROWS[1] + BH + 0.6, "expand both components")

box(LX, ROWS[1], BW, BH,
    r"Cusp-adapted $\gamma$-Laguerre basis   ($\mu=\gamma$)" "\n"
    r"$q_{j}\propto x^{\gamma}e^{-x/2}L_{j}^{(2\gamma)}(x)$,"
    r"$\;\;x=2\beta r$,$\;\;\gamma=\sqrt{\kappa^{2}-(Z\alpha)^{2}}$" "\n"
    r"point-nucleus eigenfunctions are $\mathit{finite}$"
    "\n" r"combinations of the $q_{j}$",
    BLUE)
varrow(LC, ROWS[1], ROWS[2] + BH + 0.6, r"weight $x^{2\gamma-1}e^{-x}$")

box(LX, ROWS[2], BW, BH,
    f"Exact generalized Gauss{EN}Laguerre assembly\n"
    r"overlap, Coulomb and $\mathcal{O}_{\pm}$-coupling integrals" "\n"
    r"are polynomial $\times$ weight: integrated $\mathit{exactly}$,"
    "\nno resolution of the cusp required",
    BLUE)
varrow(LC, ROWS[2], ROWS[3] + BH + 0.6, r"assembled $H$, $S$")

box(LX, ROWS[3], BW, BH,
    f"Run-time certification  C1{EN}C3   (every build)\n"
    r"C1  $\langle f_{i}|f_{j}\rangle=\delta_{ij}$"
    r"$\qquad$C2  $\langle f_{i}|\mathcal{O}_{+}f_{j}\rangle="
    r"\langle\mathcal{O}_{-}f_{i}|f_{j}\rangle$" "\n"
    r"C3  $\langle f|\mathcal{O}_{\pm}\mathcal{O}_{\mp}f\rangle="
    r"\|\mathcal{O}_{\mp}f\|^{2}$   (centrifugal $\kappa(\kappa\mp1)$)" "\n"
    r"order-unity violation $\Rightarrow$ build rejected",
    ORANGE)

# ---- wrap connector: bottom of the left column to top of the right -------
CH = (LX + BW + RX) / 2                  # channel between the columns
for seg in (Line2D([LX + BW, CH], [ROWS[3] + BH / 2, ROWS[3] + BH / 2],
                   lw=1.0, color="k"),
            Line2D([CH, CH], [ROWS[3] + BH / 2, ROWS[0] + BH / 2],
                   lw=1.0, color="k")):
    ax.add_line(seg)
ax.add_patch(FancyArrowPatch((CH, ROWS[0] + BH / 2), (RX, ROWS[0] + BH / 2),
                             arrowstyle="-|>", mutation_scale=10, lw=1.0,
                             color="k"))
ax.text(CH - 0.8, (ROWS[3] + ROWS[0]) / 2 + BH / 2, "accepted build",
        fontsize=6.3, ha="center", va="center", rotation=90, style="italic",
        color="0.25")

# ==== right column: pencil -> refinement -> spectrum -> outputs ===========
box(RX, ROWS[0], BW, BH,
    r"Mass-shifted pencil  $H-c^{2}S$" "\n"
    r"the rest mass survives only as $-2c^{2}S$ in the" "\n"
    r"lower block, so the in-gap eigenvalues $\mathit{are}$" "\n"
    r"the binding energies $E-c^{2}$",
    ORANGE)
varrow(RC, ROWS[0], ROWS[1] + BH + 0.6, "in-gap eigenpairs")

box(RX, ROWS[1], BW, BH,
    "Exactly summed Rayleigh refinement\n"
    r"$E=\mathrm{fsum}(x^{\top}(H-c^{2}S)x)\,/\,\mathrm{fsum}(x^{\top}Sx)$"
    "\nremoves the residual cancellation:\n"
    r"floor $\sim10^{-14}$ $\mathit{relative}$ to the binding energy",
    ORANGE)
varrow(RC, ROWS[1], ROWS[2] + BH + 0.6, "refined eigenvalues")

box(RX, ROWS[2], BW, BH,
    r"Certified spectrum  $E-c^{2}$  (Hartree)" "\n"
    f"$N=15${EN}$45$ basis functions," r"$\quad Z\alpha\leq0.9924$" "\n"
    r"$\beta$ fixed by minimal inter-basis drift",
    GREEN)
varrow(RC, ROWS[2], ROWS[3] + BH + 0.6)

box(RX, ROWS[3], BW, BH,
    "Paper A outputs\n"
    f"point-nucleus table: 72 levels, $Z=92${EN}$136$\n"
    r"screened validation: $f$ to $2\times10^{-15}$ vs 40 digits" "\n"
    r"critical screening $\lambda_{c}(Z,\kappa)$:"
    f" $2p_{{1/2}}${EN}$2p_{{3/2}}$ split",
    PURPLE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="extra path to write the PDF to (e.g. the "
                         "manuscript's figA_pipeline.pdf)")
    args = ap.parse_args()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"pipeline_dirac.{ext}", dpi=300)
        print("saved", FIG / f"pipeline_dirac.{ext}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print("saved", out)


if __name__ == "__main__":
    main()
