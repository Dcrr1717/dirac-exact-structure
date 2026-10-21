"""Discriminator: is the Z=80 DKB variational violation an assembly bug or
genuine (non-form-bounded partition / spurious-state) pathology?

D1  Direct-application cross-check: rebuild every H/S block by applying the
    Dirac operator NUMERICALLY to the DKB basis columns (no integration-by-
    parts shortcuts) and compare with the assembled blocks. Any mismatch
    beyond quadrature noise = algebra bug. Agreement = faithful Galerkin.

D2  Quadrature-drift test: if <O+phi| Z/r |O+phi> is log-divergent for
    ell=0 (O+phi does not vanish at r=0, point nucleus), the offending
    matrix elements and the spurious eigenvalues must DRIFT as nquad grows
    (deeper nodes probe the singularity). A converged-and-stable wrong
    answer would instead point back to algebra.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dirac import RadialDiracBasis, exact_ladder  # noqa: E402

t0 = time.perf_counter()


def direct_blocks(B: "RadialDiracBasis"):
    """H, S blocks via direct numerical application of h to basis columns."""
    c, kappa, r, W = B.c, B.kappa, B.r, B.W
    f, g, h, df, d2f = B.f, B.g, B.h, B.df, B.d2f
    V = B.V
    # derivatives of the balanced components (from f'' chain rule)
    gp = (d2f + kappa * (df / r - f / r ** 2)) / (2 * c)     # g'
    hp = (-d2f + kappa * (df / r - f / r ** 2)) / (2 * c)    # h'
    Opg = -gp + kappa / r * g
    Omh = hp + kappa / r * h

    def dot(u1, u2, v1, v2):
        return np.einsum('ik,jk,k->ij', u1, v1, W) + np.einsum('ik,jk,k->ij', u2, v2, W)

    # h applied to Psi+ = (f, g) and Psi- = (-h, f)
    hP1, hP2 = (V + c * c) * f + 2 * c * c * h + c * (Opg - 2 * c * h) + 0 * f, None
    # careful: cO+ g computed directly:
    hP1 = (V + c * c) * f + c * Opg
    hP2 = c * (2 * c * g) + (V - c * c) * g          # cO- f = 2c^2 g
    hM1 = -(V + c * c) * h + c * (2 * c * h)         # cO+ f = 2c^2 h
    hM2 = -c * Omh + (V - c * c) * f

    Hpp = dot(f, g, hP1, hP2)
    Hpm = dot(f, g, hM1, hM2)
    Hmp = dot(-h, f, hP1, hP2)
    Hmm = dot(-h, f, hM1, hM2)
    Spp = dot(f, g, f, g)
    Spm = dot(f, g, -h, f)
    Smm = dot(-h, f, -h, f)
    return dict(Hpp=Hpp, Hpm=Hpm, Hmp=Hmp, Hmm=Hmm, Spp=Spp, Spm=Spm, Smm=Smm)


def assembled_blocks(B: "RadialDiracBasis"):
    c, V = B.c, B.V
    f, g, h = B.f, B.g, B.h
    ME, MEv = B._me, B._mev
    Sff = ME(f, f)
    out = dict(
        Spp=Sff + ME(g, g),
        Spm=-ME(f, h) + ME(g, f),
        Smm=ME(h, h) + Sff,
        Hpp=(MEv(f, V, f) + c * c * Sff + 4 * c * c * ME(g, g)
             + MEv(g, V, g) - c * c * ME(g, g)),
        Hmm=(MEv(h, V, h) + c * c * ME(h, h) - 4 * c * c * ME(h, h)
             + MEv(f, V, f) - c * c * Sff),
        Hpm=(-(MEv(f, V, h) + c * c * ME(f, h)) + 2 * c * c * ME(f, h)
             - 0.5 * ME(g, B.OmOp) + MEv(g, V, f) - c * c * ME(g, f)),
    )
    return out


print("== D1: assembled vs direct-application blocks (max rel deviation) ==")
for kappa, Z in [(-1, 1.0), (-1, 80.0), (+1, 80.0)]:
    beta = 0.9 * max(Z, 1.0)
    B = RadialDiracBasis(40, kappa, Z, beta)
    A = assembled_blocks(B)
    D = direct_blocks(B)
    devs = {}
    for k in ("Spp", "Spm", "Smm", "Hpp", "Hmm", "Hpm"):
        scale = max(1.0, np.max(np.abs(D[k])))
        devs[k] = np.max(np.abs(A[k] - D[k])) / scale
    # also symmetry of the DIRECT H (boundary-term detector):
    sym = np.max(np.abs(D["Hpm"] - D["Hmp"].T)) / max(1.0, np.max(np.abs(D["Hpm"])))
    print(f" kappa={kappa:+d} Z={Z:4.0f}: " +
          " ".join(f"{k}={devs[k]:.1e}" for k in devs) + f"  Hpm-vs-Hmp^T={sym:.1e}")

print("\n== D2: quadrature drift of the kappa=+1, Z=80 DKB ground error ==")
lad = exact_ladder(+1, 80.0)
for nq in (2000, 4000, 8000):
    B = RadialDiracBasis(40, +1, 80.0, 72.0, nquad=nq)
    gap = B.solve_dkb()
    err = gap[0] - lad[0] if gap.size else np.nan
    # the suspicious matrix element for the log story, ell=0 channel analogue:
    print(f"  nq={nq:5d}: err(state1)={err:+.4e}   certs={tuple(f'{x:.0e}' for x in B.certs)}")

print("\n== D2b: same drift test for kappa=-1 (the O+phi(0)!=0 channel) ==")
lad = exact_ladder(-1, 80.0)
for nq in (2000, 4000, 8000):
    B = RadialDiracBasis(40, -1, 80.0, 72.0, nquad=nq)
    gap = B.solve_dkb()
    err = gap[0] - lad[0] if gap.size else np.nan
    hVh = -np.einsum('k,k->', B.h[0] ** 2 * B.Z / B.r, B.W)   # <h0|V|h0>
    print(f"  nq={nq:5d}: err(state1)={err:+.4e}   <h_0|V|h_0>={hVh:+.6f}")

print("\n== D3: S-pencil conditioning, s=-1 (module) vs s=+1 span, nb=40 vs 60 ==")
print("   (hypothesis: the s=+1 catastrophe at nb=60 was S near-singularity)")


def build_S(B, s):
    ME = B._me
    f, g, h = B.f, B.g, B.h
    Sff = ME(f, f)
    Spp = Sff + ME(g, g)
    Spm = s * ME(f, h) + ME(g, f)
    Smm = ME(h, h) + Sff
    S = np.block([[Spp, Spm], [Spm.T, Smm]])
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    return ev.min(), ev.max()


for nb in (40, 60):
    for kappa, Z, beta in [(-1, 1.0, 0.5), (-1, 1.0, 0.9), (+1, 80.0, 72.0)]:
        B = RadialDiracBasis(nb, kappa, Z, beta)
        for s in (+1, -1):
            mn, mx = build_S(B, s)
            print(f"  nb={nb} kappa={kappa:+d} Z={Z:4.0f} beta={beta:5.1f} s={s:+d}: "
                  f"min(S)={mn:.2e}  cond={mx/mn if mn > 0 else float('inf'):.1e}")

print(f"\ntotal: {time.perf_counter()-t0:.1f} s")
