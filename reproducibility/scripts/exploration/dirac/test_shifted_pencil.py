"""Shifted-pencil validation: does H - c^2 S + exactly-summed Rayleigh
refinement kill the rest-mass cancellation floor (2 c^2 eps ~ 8e-12 Ha)?

Reference: Sommerfeld formula evaluated in 40-digit decimal arithmetic
(so the reference itself is correctly rounded, not double-truncated).

PRE-REGISTERED PREDICTIONS:
P1  unshifted floor at Z=1: |rel err| ~ 1e-11 (abs ~ c^2*eps, rel to 0.5 Ha)
P2  shifted+refined at Z=1: |rel err| ~ 1e-15 or better (3-4 orders gained)
P3  shifted+refined at Z=92 (gamma-basis, cusp rule): ~1e-15 relative
P4  shifted and unshifted agree to ~5e-12 absolute always (same physics)
"""

import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import C_AU, RadialDiracBasis  # noqa: E402

getcontext().prec = 40
t0 = time.perf_counter()


def exact_dec(nr, kappa, Z, c):
    """Binding energy E - c^2, Sommerfeld formula, 40-digit decimal."""
    cD, ZD = Decimal(repr(c)), Decimal(repr(Z))
    nu = ZD / cD
    g = (Decimal(kappa * kappa) - nu * nu).sqrt()
    d = nu / (Decimal(nr) + g)
    return cD * cD / (Decimal(1) + d * d).sqrt() - cD * cD


def relerr(E, nr, kappa, Z, c):
    ex = exact_dec(nr, kappa, Z, c)
    return float((Decimal(repr(float(E))) - ex) / ex)


print("== Z=1, integer basis, Legendre quadrature, nb=40 ==")
print(f"{'kappa':>6} {'state':>6} {'rel unshifted':>14} {'rel shifted':>12} {'|diff| abs':>11}")
for kappa in (-1, 1, -2, 2):
    B = RadialDiracBasis(40, kappa, 1.0, 1.0)
    e_u = B.solve_naive()
    e_s = B.solve_naive(shifted=True)
    nr0 = 0 if kappa < 0 else 1
    for i in range(3):
        print(f"{kappa:>6} {'nr=' + str(nr0 + i):>6}"
              f" {relerr(e_u[i], nr0 + i, kappa, 1.0, C_AU):>14.1e}"
              f" {relerr(e_s[i], nr0 + i, kappa, 1.0, C_AU):>12.1e}"
              f" {abs(e_u[i] - e_s[i]):>11.1e}")

print("\n== Z=1, RKB, nb=40 ==")
for kappa in (-1, 1):
    B = RadialDiracBasis(40, kappa, 1.0, 1.0)
    e_u, e_s = B.solve_rkb(), B.solve_rkb(shifted=True)
    nr0 = 0 if kappa < 0 else 1
    for i in range(3):
        print(f"{kappa:>6} {'nr=' + str(nr0 + i):>6}"
              f" {relerr(e_u[i], nr0 + i, kappa, 1.0, C_AU):>14.1e}"
              f" {relerr(e_s[i], nr0 + i, kappa, 1.0, C_AU):>12.1e}"
              f" {abs(e_u[i] - e_s[i]):>11.1e}")

print("\n== Z=92, gamma-basis + cusp rule, nb=40, beta=0.8Z ==")
for kappa in (-1, 1, -2, 2):
    gam = np.sqrt(kappa * kappa - (92.0 / C_AU) ** 2)
    B = RadialDiracBasis(40, kappa, 92.0, 0.8 * 92.0, mu=gam, quad="cusp",
                         cert_tol=1e-6)
    e_u = B.solve_naive()
    e_s = B.solve_naive(shifted=True)
    nr0 = 0 if kappa < 0 else 1
    for i in range(3):
        print(f"{kappa:>6} {'nr=' + str(nr0 + i):>6}"
              f" {relerr(e_u[i], nr0 + i, kappa, 92.0, C_AU):>14.1e}"
              f" {relerr(e_s[i], nr0 + i, kappa, 92.0, C_AU):>12.1e}"
              f" {abs(e_u[i] - e_s[i]):>11.1e}")

print(f"\ntotal: {time.perf_counter() - t0:.1f} s")
