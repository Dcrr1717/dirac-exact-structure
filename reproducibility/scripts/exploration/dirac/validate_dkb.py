"""Validation battery for the corrected DKB radial Dirac solver.

B1  Accuracy vs exact Dirac-Coulomb across channels and Z (DKB vs RKB),
    with the VARIATIONAL flag: any Galerkin level strictly below its exact
    counterpart marks a broken (non-Galerkin) assembly.
B2  Convergence in basis size for Z=1 (near-integer cusp gamma ~ 1) and
    Z=80, 120 (strong non-integer cusp): expect spectral-like decay for
    Z=1 and cusp-limited algebraic decay for large Z -- the (L3) phenomenon.
B3  Assembly certifications C1-C3 reported for every build.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import RadialDiracBasis, exact_ladder, validate  # noqa: E402

t0 = time.perf_counter()

print("== B1: accuracy & variational flag (nb=60, best over beta grid) ==")
print(f"{'kappa':>6} {'Z':>6} {'method':>7} {'err(state1)':>12} {'err(state2)':>12} "
      f"{'viol':>5} {'certs (C1,C2,C3)':>30}")
for kappa in (-1, +1, -2, +2):
    for Z in (1.0, 80.0):
        for method in ("dkb", "rkb"):
            r = validate(kappa, Z, nb=60, method=method, kcheck=3)
            if r is None:
                print(f"{kappa:>6} {Z:>6.0f} {method:>7}   -- insufficient in-gap states --")
                continue
            print(f"{kappa:>6} {Z:>6.0f} {method:>7} {r['err'][0]:>12.2e} {r['err'][1]:>12.2e} "
                  f"{str(r['viol']):>5}  ({r['certs'][0]:.0e},{r['certs'][1]:.0e},{r['certs'][2]:.0e})")

print("\n== B2: convergence in nb (kappa=-1, DKB, best over beta) ==")
print(f"{'Z':>6} " + " ".join(f"nb={n:<4d}" for n in (10, 20, 30, 40, 60, 80)))
for Z in (1.0, 80.0, 120.0):
    errs = []
    for nb in (10, 20, 30, 40, 60, 80):
        r = validate(-1, Z, nb=nb, method="dkb", kcheck=1)
        errs.append(abs(r["err"][0]) if r else np.nan)
    print(f"{Z:>6.0f} " + " ".join(f"{e:8.1e}" for e in errs))

print(f"\ntotal wall time: {time.perf_counter()-t0:.1f} s")
