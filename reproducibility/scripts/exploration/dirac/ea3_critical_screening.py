"""E-A3: relativistic critical screening map lambda_c(Z, kappa).

Yukawa (exponentially screened Coulomb) V(r) = -Z e^{-lam r}/r.  As the
screening lam grows the ground state of channel kappa loses binding;
lambda_c is the threshold where the binding energy E - c^2 reaches 0.

Physics: the short-range behaviour is still -Z/r, so the cusp exponent
gamma = sqrt(kappa^2 - (Z/c)^2) is unchanged and the gamma-basis + exact
cusp quadrature stay exact at r -> 0.  Near threshold the state is
diffuse (small beta) and the binding -> 0, so the SHIFTED pencil
(rest-mass cancellation removed) is essential.

Threshold extrapolation: for the ell=0 ground state (kappa=-1) the
short-range threshold law gives E_b ~ -(lam_c - lam)^2, i.e. sqrt(-E_b)
is LINEAR in lam near threshold; we fit sqrt(-E_b) vs lam on the
near-threshold window and take the zero crossing.  For each channel we
ALSO fit a free power E_b ~ -A (lam_c - lam)^p to report the exponent and
cross-check.  The reduced critical screening s_c = lam_c / Z is the
dimensionless observable; s_c(Z->0) recovers the nonrelativistic constant
(1s: 1.190612..., the Yukawa 1s value), and its rise with Zalpha is the
relativistic enhancement.

Convergence control: lam_c is computed for two basis sizes and two beta
scales; the spread is the honest error bar.  A run-time consistency check
compares against the known nonrelativistic s_c at the smallest Z.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import C_AU, RadialDiracBasis  # noqa: E402

C = C_AU
t0 = time.perf_counter()

# Screening grid scanned by sample_curve().  Module-level so that an
# opt-in CLI run can widen or refine it for channels whose threshold
# sits outside the default window; the literal below is EXACTLY the
# expression that used to be inlined in sample_curve, so the default
# run is unchanged.
S_SCAN = np.arange(0.05, 1.60, 0.01)

# nonrelativistic reduced critical screening s_c = lam_c/Z (Yukawa), for
# the Z->0 consistency anchor only (NOT used to drive sampling):
# 1s (kappa=-1): 1.190612; 2p_{1/2}/2p_{3/2} (kappa=+1,-2): 0.220216
# (nonrel 2p is degenerate; del Valle-Nader 2018).
S_NONREL = {-1: 1.190612, 1: 0.220216, -2: 0.220216}


def ground_binding(Z, kappa, lam, nb, beta, nq=None):
    """Physical ground-state binding of channel kappa.  For kappa>0 the
    naive equal-basis discretization carries ONE spurious state (the
    deepest gap eigenvalue; degenerate with the -kappa 1s at lam=0, see
    E-A2/FINDINGS #8, and shifting under screening but staying far deeper
    than the 2p threshold) -> the physical ground state is ev[1].  For
    kappa<0 there is no spurious state -> ev[0]."""
    nu = Z / C
    gam = float(np.sqrt(kappa * kappa - nu * nu))
    B = RadialDiracBasis(nb, kappa, Z, beta, mu=gam, quad="cusp",
                         nquad=(nq or nb + 150), cert_tol=1e-6)
    B.V = -Z * np.exp(-lam * B.r) / B.r
    ev = B.solve_naive(shifted=True)
    idx = 1 if kappa > 0 else 0
    return float(ev[idx]) if ev.size > idx else np.nan


def sample_curve(Z, kappa, nb, beta_fac):
    """Self-bracketing: scan the reduced screening s upward until the
    physical ground state is lost, collecting (lam, E_b).  beta adapts to
    the shrinking binding so the diffuse tail stays resolved:
    beta ~ max(beta_fac*Z, sqrt(2|E_b|)).  Fine step near threshold."""
    lams, Ebs = [], []
    lost = 0
    for s in S_SCAN:
        lam = s * Z
        Eb_prev = Ebs[-1] if Ebs else -0.05 * Z * Z / (kappa * kappa)
        beta = max(beta_fac * Z, np.sqrt(2 * abs(Eb_prev)))
        Eb = ground_binding(Z, kappa, lam, nb, beta)
        if not np.isfinite(Eb) or Eb >= -1e-9:
            if lams:                       # only stop after we had a state
                lost += 1
                if lost >= 2:
                    break
            continue
        lost = 0
        lams.append(lam)
        Ebs.append(Eb)
    return np.array(lams), np.array(Ebs)


def _lin_zero(x, y):
    a, b = np.polyfit(x, y, 1)
    return -b / a


def extrapolate_lamc(lams, Ebs, ell):
    """Extrapolate E_b -> 0 with the KNOWN short-range threshold law
    E_b ~ -(lam_c - lam)^p, p fixed by angular momentum: p = 2 for
    ell = 0 and p = 1 for ell >= 1.  The zero crossing is taken from a
    line through the points CLOSEST to threshold (the smallest |E_b|),
    where the asymptotic law holds; using the closest window rather than a
    free-power fit removes the crossover bias that inflated s_c for the 2p
    states.  Returns (lam_c, p)."""
    if len(lams) < 5:
        return np.nan, np.nan
    k = min(len(lams), 8)
    lw, ew = lams[-k:], Ebs[-k:]
    p = 2.0 if ell == 0 else 1.0
    y = (-ew) ** (1.0 / p)                # linear in lam near threshold
    return _lin_zero(lw, y), p


ZS = [1.0, 40.0, 70.0, 90.0, 110.0, 120.0, 130.0]
KAPPAS = [-1, 1, -2]
OUT = Path(__file__).with_name("ea3_critical_screening.csv")

# ---------------------------------------------------------------------
# OPT-IN overrides.  With NO command-line flags every name above keeps
# the literal value it has always had, S_SCAN is the original
# np.arange(0.05, 1.60, 0.01), and the run is bit-for-bit the default
# E-A3 sweep.  The flags exist so that channels whose threshold lies
# outside the default screening window (e.g. the d doublet kappa = +2
# / -3, whose s_c is several times smaller than the 2p one) can be
# mapped with the SAME ground_binding / extrapolate_lamc physics on a
# grid that actually brackets them.
# ---------------------------------------------------------------------
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--kappas", default=None, metavar="K1,K2,...",
                 help="opt-in channel list, replaces the default -1,1,-2")
_ap.add_argument("--zs", default=None, metavar="Z1,Z2,...",
                 help="opt-in nuclear-charge list")
_ap.add_argument("--s-scan", default=None, metavar="LO,HI,STEP",
                 help="opt-in screening scan, replaces 0.05,1.60,0.01")
_ap.add_argument("--out", default=None, metavar="PATH",
                 help="opt-in output CSV (default ea3_critical_screening.csv)")
_args, _rest = _ap.parse_known_args()
if _args.kappas:
    KAPPAS = [int(k) for k in _args.kappas.split(",")]
if _args.zs:
    ZS = [float(z) for z in _args.zs.split(",")]
if _args.s_scan:
    _lo, _hi, _st = (float(v) for v in _args.s_scan.split(","))
    S_SCAN = np.arange(_lo, _hi, _st)
if _args.out:
    OUT = Path(_args.out)
if _args.kappas or _args.zs or _args.s_scan or _args.out:
    print(f"OPT-IN RUN  kappas={KAPPAS}  ZS={ZS}"
          f"  s in [{S_SCAN[0]:.4f}, {S_SCAN[-1]:.4f}] step"
          f" {S_SCAN[1]-S_SCAN[0]:.4f} ({S_SCAN.size} pts)  -> {OUT.name}")

rows = []
print(f"c = {C}, V = -Z e^(-lam r)/r, ground state of each channel")
print(f"{'Z':>5} {'kap':>4} {'Zalpha':>7} {'lam_c':>10} {'s_c=lam_c/Z':>11}"
      f" {'p_thresh':>8} {'spread':>9} {'s_c/s_nr':>8}")
for kappa in KAPPAS:
    for Z in ZS:
        nu = Z / C
        if kappa * kappa - nu * nu <= 0.01:      # channel dived / too deep
            continue
        ell = kappa if kappa > 0 else -kappa - 1
        ests = []
        for nb in (90, 120):
            for bf in (0.20, 0.32):
                lams, Ebs = sample_curve(Z, kappa, nb, bf)
                lc, p = extrapolate_lamc(lams, Ebs, ell)
                if np.isfinite(lc):
                    ests.append((lc, p))
        if not ests:
            print(f"{Z:>5.0f} {kappa:>4}  (no bracket)")
            continue
        lcs = np.array([e[0] for e in ests])
        lamc = float(np.median(lcs))
        spread = float(lcs.max() - lcs.min())
        pmed = float(np.nanmedian([e[1] for e in ests]))
        sc = lamc / Z
        # channels reached only through the opt-in --kappas flag have no
        # tabulated nonrelativistic anchor -> report nan rather than
        # inventing one.
        ratio = sc / S_NONREL[kappa] if kappa in S_NONREL else float("nan")
        rows.append(dict(Z=Z, kappa=kappa, Zalpha=nu, lam_c=lamc, s_c=sc,
                         p_threshold=pmed, spread=spread, ratio=ratio))
        print(f"{Z:>5.0f} {kappa:>4} {nu:>7.4f} {lamc:>10.4f} {sc:>11.5f}"
              f" {pmed:>8.3f} {spread:>9.1e} {ratio:>8.4f}")

# consistency: smallest-Z s_c must match the nonrelativistic constant
for kappa in KAPPAS:
    r = [x for x in rows if x['kappa'] == kappa]
    if r:
        r0 = min(r, key=lambda x: x['Z'])
        if kappa not in S_NONREL:
            print(f"anchor kappa={kappa}: s_c(Z={r0['Z']:.0f})={r0['s_c']:.6f}"
                  f"  (no tabulated nonrelativistic constant for this"
                  f" channel; the Z->0 value above IS the computed"
                  f" nonrelativistic limit)")
            continue
        dev = abs(r0['s_c'] - S_NONREL[kappa])
        print(f"anchor kappa={kappa}: s_c(Z={r0['Z']:.0f})={r0['s_c']:.5f}"
              f" vs nonrel {S_NONREL[kappa]:.5f}  dev={dev:.1e}")

out = OUT
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nCSV -> {out.name}   total: {time.perf_counter()-t0:.1f} s")
