"""G3: exact (generalized Gauss-Laguerre, alpha=2mu-1) quadrature for the
cusp-adapted basis vs the Gauss-Legendre floor.

PRE-REGISTERED PREDICTIONS:
Q1  C1 (overlap identity) drops to machine epsilon (~1e-15): the rule is
    polynomially exact for it, unlike Gauss-Legendre on (0, X).
Q2  The 1e-5 (Z=80) and 1e-2 (Z=120) gamma-basis floors collapse to the
    rest-mass cancellation floor (~1e-11 absolute on eigenvalues, i.e.
    2c^2 * eps_mach), at ALL nb -- since the representation was already
    exact at nb=15.
Q3  No below-exact drift: with exact integrals the naive-gamma assembly is
    a faithful Galerkin pair, and the known kappa>0 single spurious level
    remains (it is a property of the span, not of quadrature).
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import RadialDiracBasis, exact_ladder, C_AU  # noqa: E402

t0 = time.perf_counter()
gam = lambda kappa, Z: np.sqrt(kappa**2 - (Z / C_AU) ** 2)


def run(kappa, Z, nb, quad):
    B = RadialDiracBasis(nb, kappa, Z, 0.9 * Z, cert_tol=np.inf,
                         mu=gam(kappa, Z), quad=quad)
    gap = B.solve_naive()
    lad = exact_ladder(kappa, Z, kmax=8)
    errs = np.array([gap[np.argmin(np.abs(gap - E))] - E for E in lad[:3]])
    return errs, B.certs


print("== G3: gamma basis, Gauss-Legendre vs exact cusp quadrature ==")
print(f"{'kappa':>6} {'Z':>5} {'nb':>4} {'quad':>9} {'err(1st)':>11} "
      f"{'err(2nd)':>11} {'err(3rd)':>11} {'C1':>8}")
for kappa, Z in [(-1, 80.0), (+1, 80.0), (-1, 120.0)]:
    for nb in (15, 40):
        for quad in ("legendre", "cusp"):
            errs, certs = run(kappa, Z, nb, quad)
            print(f"{kappa:>6} {Z:>5.0f} {nb:>4} {quad:>9} "
                  f"{errs[0]:>11.1e} {errs[1]:>11.1e} {errs[2]:>11.1e} {certs[0]:>8.0e}")

print("\n== G3b: does it hold to Z=130 (nu=0.949, gamma=0.317)? ==")
for nb in (15, 30, 50):
    errs, certs = run(-1, 130.0, nb, "cusp")
    print(f"  nb={nb:3d}: err={errs[0]:+.2e}, {errs[1]:+.2e}, {errs[2]:+.2e}   C1={certs[0]:.0e}")

print(f"\ntotal: {time.perf_counter()-t0:.1f} s")
