"""E-A1 (final form): external validation and adjudication of the
published Dirac-Yukawa anchor points.

RESOLVED CONVENTIONS (deduced from residual patterns, then independently
verified against Baye Table VI and the Krauthauser-Hill 40-digit
benchmarks; see FINDINGS.md section 7):
- c = 137.0359895 (CODATA 1986) underlies ALL published values in the
  comparison chain (never stated in any of the three papers).
- Egs = E - c^2 in Hartree; V(r) = -V0 e^{-lam r}/r.
- CSW Table 4 "Case 2: V0=0.4c, lam=0.07c" is an ERRATUM: the energy
  corresponds to Baye's case V0=0.7c, lam=0.04c (digits transposed,
  consistently in prose, table and figure caption).
- Underlying c=1 benchmarks (Krauthauser-Hill 2002, 40-digit, via Baye
  Table VI): f1 = -0.00408291802884811, f2 = -0.25879891617626010.

RESULTS (this script reproduces them):
  Case 1: our f agrees with KH to ~1e-14; ALLSM (2026) off by 4.4e-11 Ha.
  Case 2: our f agrees with KH to  2e-15; Baye LM good to 3.7e-11 Ha;
          ALLSM off by 7.6e-9 Ha.  Both adjudications favor KH.
  The Legendre quadrature stalls at 1e-4 for gamma=0.714 while the exact
  cusp rule is machine-precise: live proof of the exact-quadrature claim.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import RadialDiracBasis  # noqa: E402

t0 = time.perf_counter()
C86 = 137.0359895                      # CODATA 1986, the chain's convention

CASES = [
    ("Case 1 (V0=0.1c, lam=0.01c)", 0.1, 0.01, 8.0,
     -0.00408291802884811,            # Krauthauser-Hill 40-digit (c=1)
     -76.672555928755170,             # ALLSM 2026
     -76.672555928706560),            # reference row (KH via CSW Table 4)
    ("Case 2 (V0=0.7c, lam=0.04c) [CSW erratum: printed as 0.4c/0.07c]",
     0.7, 0.04, 92.0,
     -0.25879891617626010,
     -4859.949240857070,
     -4859.949240864677),             # Baye Lagrange-mesh value
]

# Short machine-readable identifiers for the CSV export added below
# (the printed report is unchanged).
CASE_ID = {CASES[0][0]: "case1", CASES[1][0]: "case2"}

rows = []                              # CSV export (added, print-only before)

for name, nu0, mu, beta, f_kh, e_allsm, e_ref in CASES:
    gam = np.sqrt(1.0 - nu0 ** 2)
    Z, lam = nu0 * C86, mu * C86
    print(f"== {name}  (gamma={gam:.6f}) ==")
    drift = {}
    e80 = {}
    for quad in ("cusp", "legendre"):
        vals = []
        for nb in (60, 80):
            nq = (nb + 80) if quad == "cusp" else None
            B = RadialDiracBasis(nb, -1, Z, beta, cert_tol=np.inf, mu=gam,
                                 quad=quad, nquad=nq, c=C86)
            B.V = -Z * np.exp(-lam * B.r) / B.r
            vals.append(B.solve_naive()[0])
        print(f"   {quad:>9}: E(nb=80) = {vals[1]:.12f}"
              f"   drift(60->80) = {abs(vals[1]-vals[0]):.0e}")
        drift[quad] = abs(vals[1] - vals[0])
        e80[quad] = vals[1]
        if quad == "cusp":
            E, f = vals[1], vals[1] / C86 ** 2
    print(f"   f = {f:.17f}")
    print(f"   f - KH(40d) = {f - f_kh:+.2e}   E - ALLSM = {E - e_allsm:+.2e}"
          f"   E - ref = {E - e_ref:+.2e}")
    print(f"   published ALLSM-ref discrepancy: {e_allsm - e_ref:+.1e}")

    # ---- ADDED (2026-08, packaging pass; nothing above is changed) -------
    # The manuscript (Sec. IV) quotes the MASS-SHIFTED pencil values of
    # Sec. II D -- solve_naive(shifted=True) -- while the report printed
    # above uses the unshifted solve that predates the pencil (see
    # FINDINGS.md section 8, "E-A1 con pencil").  Both are computed and
    # exported so the shipped CSV carries exactly the numbers quoted in
    # the paper together with the pre-pencil ones.
    vs = []
    for nb in (60, 80):
        B = RadialDiracBasis(nb, -1, Z, beta, cert_tol=np.inf, mu=gam,
                             quad="cusp", nquad=nb + 80, c=C86)
        B.V = -Z * np.exp(-lam * B.r) / B.r
        vs.append(B.solve_naive(shifted=True)[0])
    E_sh, f_sh = vs[1], vs[1] / C86 ** 2
    print(f"   shifted pencil (Sec. IV value): E = {E_sh:.12f}"
          f"   f = {f_sh:.17f}   f - KH(40d) = {f_sh - f_kh:+.2e}"
          f"   drift(60->80) = {abs(vs[1]-vs[0]):.0e}\n")

    rows.append(dict(
        case=CASE_ID[name], label=name, V0_over_c=nu0, lambda_over_c=mu,
        c_convention=C86, Z=Z, lam=lam, gamma=float(gam), beta=beta, nb=80,
        E_shifted=E_sh, f_shifted=f_sh, dev_f_shifted=f_sh - f_kh,
        E_unshifted=E, f_unshifted=f, dev_f_unshifted=f - f_kh,
        f_KH40=f_kh, E_ALLSM2026=e_allsm, dev_E_ALLSM=E_sh - e_allsm,
        E_ref=e_ref, dev_E_ref=E_sh - e_ref,
        published_ALLSM_minus_ref=e_allsm - e_ref,
        E_legendre=e80["legendre"],
        drift_cusp=drift["cusp"], drift_legendre=drift["legendre"],
        drift_cusp_shifted=abs(vs[1] - vs[0]),
    ))

out = Path(__file__).with_name("validation_errors.csv")
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"CSV -> {out.name}")

print(f"total: {time.perf_counter()-t0:.1f} s")
