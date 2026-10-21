"""G1: cusp-adapted (r^gamma) Laguerre basis vs integer (r^(ell+1)) basis
for point-nucleus Dirac-Coulomb at strong coupling.

PRE-REGISTERED PREDICTIONS (stated before running):
P1  Integer basis, naive assembly, Z=80/120: cusp-limited slow algebraic
    convergence (errors ~0.1-100 Ha at nb<=60), as measured earlier.
P2  Gamma basis (mu = gamma = sqrt(kappa^2 - (Z/c)^2)): near-machine
    ground-state accuracy already at small nb, because the exact
    eigenfunctions are finite combinations of gamma-Laguerre functions
    (Grant L-spinors; Wu 2025).
P3  For kappa=+1 the gamma basis may exhibit the L-spinor's known single
    spurious level -- count in-gap states and compare with the ladder.
P4  Certifications: C1 (overlap) machine-precision in all cases; C2/C3
    degrade for mu<1/2 (Z=120, boundary r^(2mu-1) no longer vanishes) --
    reported, not fatal, via cert_tol=inf in this experiment.

If P2 fails, the gamma-basis route (and part of Lemma L3's cure) needs
rethinking; if it holds, the point-nucleus branch of Teorema 1 has its
practical basis.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import RadialDiracBasis, exact_ladder, C_AU  # noqa: E402

t0 = time.perf_counter()


def run(kappa, Z, nb, mu=None, beta=None):
    beta = beta if beta is not None else 0.9 * Z
    B = RadialDiracBasis(nb, kappa, Z, beta, cert_tol=np.inf, mu=mu)
    gap = B.solve_naive()
    lad = exact_ladder(kappa, Z, kmax=10)
    # robust match: each exact level to its nearest Galerkin value
    errs = np.array([gap[np.argmin(np.abs(gap - E))] - E for E in lad[:4]])
    # unmatched in-gap states below the 4th level = spurious candidates
    matched = {int(np.argmin(np.abs(gap - E))) for E in lad[:8]}
    spur = [e for i, e in enumerate(gap) if i not in matched and e < lad[3]]
    return errs, len(spur), B.certs, gap.size


gam = lambda kappa, Z: np.sqrt(kappa**2 - (Z / C_AU) ** 2)

print("== G1: integer vs gamma prefactor, naive assembly, nb=40 ==")
print(f"{'kappa':>6} {'Z':>5} {'basis':>8} {'err(1st)':>10} {'err(2nd)':>10} "
      f"{'spurious<E4':>11} {'C1':>8} {'C3':>8}")
for kappa, Z in [(-1, 80.0), (+1, 80.0), (-1, 120.0)]:
    for tag, mu in (("integer", None), ("gamma", gam(kappa, Z))):
        errs, nspur, certs, ngap = run(kappa, Z, 40, mu=mu)
        print(f"{kappa:>6} {Z:>5.0f} {tag:>8} {errs[0]:>10.1e} {errs[1]:>10.1e} "
              f"{nspur:>11d} {certs[0]:>8.0e} {certs[2]:>8.0e}")

print("\n== G2: convergence in nb, kappa=-1, gamma basis vs integer ==")
print(f"{'Z':>5} {'basis':>8} " + " ".join(f"nb={n:<3d}" for n in (15, 25, 40, 60)))
for Z in (80.0, 120.0):
    for tag, mufn in (("integer", lambda k, z: None), ("gamma", gam)):
        errs = []
        for nb in (15, 25, 40, 60):
            mu = mufn(-1, Z) if tag == "gamma" else None
            e, _, _, _ = run(-1, Z, nb, mu=mu)
            errs.append(abs(e[0]))
        print(f"{Z:>5.0f} {tag:>8} " + " ".join(f"{e:7.1e}" for e in errs))

print(f"\ntotal: {time.perf_counter()-t0:.1f} s")
