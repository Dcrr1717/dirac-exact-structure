"""E-A2: point-nucleus Dirac-Coulomb reference table, Z = 92..136,
kappa = -1, +1, -2, +2, lowest three physical states per channel.

Instrument: gamma-basis (mu = gamma = sqrt(kappa^2 - (Z/c)^2)) + exact
cusp quadrature + SHIFTED pencil H - c^2 S with exactly-summed Rayleigh
refinement (floor ~eps relative to the binding energy).

Conventions (declared, unlike the literature we adjudicated in E-A1):
  c = 137.035999084 (CODATA 2018), V = -Z/r, binding = E - c^2 in Ha.
Reference: Sommerfeld formula in 40-digit decimal arithmetic.

Protocol (pre-registered):
- beta grid {0.55, 0.80, 1.05} x Z, nb in {30, 45}; per state, beta is
  chosen by MINIMAL DRIFT |E(45) - E(30)| (never by distance to exact).
- State matching: each computed gap state is assigned to the nearest
  exact level; rel. distance > 1e-3 => unmatched. Unmatched states deeper
  than the 4th exact level are reported as SPURIOUS (the naive kappa>0
  discretization is expected to carry one, pinned near the ground state
  of the kappa<0 partner channel).
- Cross-check: shifted RKB at beta = 0.8 Z where it is defined. NOTE:
  for gamma < 1/2 (|kappa| = 1, Z >~ 119) the RKB small components
  chi ~ r^(gamma-1) are NOT normalizable at a point nucleus -- RKB
  leaves L2/the form domain and is skipped (na). The naive gamma-basis
  needs no such caveat.
"""

import csv
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

Z_LIST = [92.0, 100.0, 110.0, 120.0, 130.0, 136.0]
KAPPAS = [-1, 1, -2, 2]
BETA_FACS = [0.55, 0.80, 1.05]
NB_LO, NB_HI = 30, 45
NSTATE = 3


def exact_dec(nr, kappa, Z):
    cD, ZD = Decimal(repr(C_AU)), Decimal(repr(Z))
    nu = ZD / cD
    g = (Decimal(kappa * kappa) - nu * nu).sqrt()
    d = nu / (Decimal(nr) + g)
    return cD * cD / (Decimal(1) + d * d).sqrt() - cD * cD


def match(levels, exact):
    """Assign computed gap levels to nearest exact level (rel tol 1e-3).
    Returns (dict level_index -> computed value, list of spurious)."""
    matched, spurious = {}, []
    for e in levels:
        j = int(np.argmin(np.abs(exact - e)))
        if abs((e - exact[j]) / exact[j]) < 1e-3:
            matched.setdefault(j, e)          # keep first (deepest) match
        elif e < exact[3]:                    # deep impostor
            spurious.append(e)
    return matched, spurious


def solve(nb, kappa, Z, beta, gam, method):
    B = RadialDiracBasis(nb, kappa, Z, beta, mu=gam, quad="cusp",
                         nquad=nb + 150, cert_tol=1e-6)
    ev = (B.solve_naive(shifted=True) if method == "naive"
          else B.solve_rkb(shifted=True))
    return ev, B.certs


rows = []
worst_cert = 0.0
for Z in Z_LIST:
    nu = Z / C_AU
    for kappa in KAPPAS:
        gam = float(np.sqrt(kappa * kappa - nu * nu))
        nr0 = 0 if kappa < 0 else 1
        exact = np.array([float(exact_dec(nr0 + k, kappa, Z))
                          for k in range(12)])
        exactD = [exact_dec(nr0 + k, kappa, Z) for k in range(NSTATE)]

        runs = {}                             # (beta_fac, nb) -> matched dict
        spur = []
        for bf in BETA_FACS:
            for nb in (NB_LO, NB_HI):
                ev, certs = solve(nb, kappa, Z, bf * Z, gam, "naive")
                worst_cert = max(worst_cert, certs[0], certs[3])
                m, s = match(ev, exact)
                runs[(bf, nb)] = m
                if nb == NB_HI and bf == 0.80:
                    spur = s

        # RKB cross-check where the small components are normalizable
        rkb_rel = [float("nan")] * NSTATE
        if gam > 0.55:
            ev, _ = solve(NB_HI, kappa, Z, 0.80 * Z, gam, "rkb")
            m, _ = match(ev, exact)
            for k in range(NSTATE):
                if k in m:
                    rkb_rel[k] = float((Decimal(repr(float(m[k]))) - exactD[k])
                                       / exactD[k])

        for k in range(NSTATE):
            best = None
            for bf in BETA_FACS:
                lo, hi = runs[(bf, NB_LO)], runs[(bf, NB_HI)]
                if k in lo and k in hi:
                    drift = abs(hi[k] - lo[k])
                    if best is None or drift < best[0]:
                        best = (drift, bf, hi[k])
            drift, bf, E = best
            E = float(E)
            rel = float((Decimal(repr(E)) - exactD[k]) / exactD[k])
            rows.append(dict(Z=Z, kappa=kappa, nu=nu, gamma=gam,
                             nr=nr0 + k, n=nr0 + k + abs(kappa),
                             E=E, rel=rel, drift=drift, beta_fac=bf,
                             nspur=len(spur),
                             spur0=(spur[0] if spur else float("nan")),
                             rkb_rel=rkb_rel[k]))

# ---- report ---------------------------------------------------------------
print(f"c = {C_AU} (CODATA 2018), binding E - c^2 in Ha, nb = {NB_HI}")
print(f"worst enforced cert (C1/C4) over all builds: {worst_cert:.1e}\n")
print(f"{'Z':>4} {'kap':>4} {'gamma':>7} {'nr':>3} {'E_binding':>22}"
      f" {'rel_err':>9} {'drift':>8} {'bF':>5} {'#sp':>4} {'rkb_rel':>9}")
for r in rows:
    print(f"{r['Z']:>4.0f} {r['kappa']:>4} {r['gamma']:>7.4f} {r['nr']:>3}"
          f" {r['E']:>22.13f} {r['rel']:>9.1e} {r['drift']:>8.1e}"
          f" {r['beta_fac']:>5.2f} {r['nspur']:>4} {r['rkb_rel']:>9.1e}")

sp = [r for r in rows if r['nspur'] and r['nr'] == (0 if r['kappa'] < 0 else 1)]
if sp:
    print("\nspurious states (naive, beta=0.8Z, nb=45) vs partner-channel 1s:")
    for r in sp:
        e1s_partner = float(exact_dec(0, -abs(r['kappa']), r['Z']))
        print(f"  Z={r['Z']:>3.0f} kappa={r['kappa']:>2}: {r['nspur']} spurious,"
              f" deepest at {r['spur0']:.6f}  (partner 1s: {e1s_partner:.6f},"
              f" rel offset {abs((r['spur0'] - e1s_partner) / e1s_partner):.1e})")

out = Path(__file__).with_name("ea2_point_table.csv")
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nCSV -> {out.name}   total: {time.perf_counter() - t0:.1f} s")
