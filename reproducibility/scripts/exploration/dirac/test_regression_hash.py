"""A-N2: hashed regression test for the certified Dirac spectrum.

One fast, fully deterministic canonical case is pushed through
``RadialDiracBasis`` exactly as the production driver does it, the
resulting in-gap eigenvalues are formatted at a fixed precision, and the
concatenation is hashed with SHA-256.  The digest is compared against the
value stored in ``EXPECTED_SHA256`` below.

Canonical case (the headline case of Paper A, Table I)
------------------------------------------------------
    Z = 92, kappa = -1, point nucleus  V = -Z/r
    cusp-adapted gamma-Laguerre basis, mu = gamma = sqrt(1 - (Z/c)^2)
    N  = 45 basis functions,  beta = 0.8 Z,  nquad = N + 150
    generalized Gauss-Laguerre ("cusp") rule, cert_tol = 1e-6
    mass-shifted pencil  H - c^2 S  + exactly summed Rayleigh refinement
    -> the 6 deepest in-gap eigenvalues (binding energies E - c^2, Ha)

Nothing here is sampled and no seed is used: the basis, the quadrature
nodes and the LAPACK call are all deterministic, so two runs of the same
interpreter on the same machine give bit-identical eigenvalues.

Two independent checks, on purpose
----------------------------------
``test_canonical_digest``  is the bit-level regression guard.  It is the
strictest possible statement that *nothing at all* changed, and it is the
one that catches a silent perturbation of the assembly.

``test_canonical_values``  is the PORTABLE check.  ``{:.12e}`` keeps
thirteen significant digits, but the eigenvalues are only accurate to
~1e-14 *relative*, so the last digit is at the roundoff floor and depends
on the numpy/LAPACK(BLAS) build.  On a different LAPACK the digest may
legitimately differ while the physics is untouched; the numeric
tolerance checks are what a third party should trust.  A digest mismatch
is therefore a signal to *look*, not automatically a failure of the
method -- run ``python exploration/dirac/test_regression_hash.py`` to see
which digit moved.

Usage
-----
    pytest exploration/dirac/test_regression_hash.py -q       # check
    python exploration/dirac/test_regression_hash.py          # print digest
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dirac import C_AU, RadialDiracBasis, exact_dirac_coulomb  # noqa: E402

# ---- canonical case: frozen parameters -----------------------------------
Z_CANON = 92.0
KAPPA_CANON = -1
NB_CANON = 45
BETA_FAC = 0.8
NQUAD_EXTRA = 150
CERT_TOL = 1e-6
N_LEVELS = 6                 # how many in-gap eigenvalues enter the hash
PRECISION = "{:.12e}"        # fixed formatting -> stable byte string

# ---- stored digest -------------------------------------------------------
# Obtained by running this file as a script; see the environment recorded in
# PaperA_Dirac_PRA/reproducibility/EXPECTED_OUTPUTS.md.  ENVIRONMENT
# SENSITIVE (numpy/LAPACK build) -- see the module docstring.
EXPECTED_SHA256 = (
    "77c61fe2ece0d1c233a5c07207d64f5e0ff25e87960cd9f016725d742e24c082"
)


def canonical_eigenvalues() -> np.ndarray:
    """Run the canonical case; return the N_LEVELS deepest binding energies."""
    gamma = float(np.sqrt(KAPPA_CANON ** 2 - (Z_CANON / C_AU) ** 2))
    basis = RadialDiracBasis(
        NB_CANON, KAPPA_CANON, Z_CANON, BETA_FAC * Z_CANON,
        mu=gamma, quad="cusp", nquad=NB_CANON + NQUAD_EXTRA,
        cert_tol=CERT_TOL,
    )
    ev = basis.solve_naive(shifted=True)
    assert ev.size >= N_LEVELS, f"only {ev.size} in-gap levels found"
    return np.asarray(ev[:N_LEVELS], dtype=float)


def format_eigenvalues(ev: np.ndarray) -> str:
    """The exact byte string that gets hashed (newline separated, LF)."""
    return "\n".join(PRECISION.format(float(x)) for x in ev)


def digest(ev: np.ndarray) -> str:
    return hashlib.sha256(format_eigenvalues(ev).encode("ascii")).hexdigest()


# ---- tests ---------------------------------------------------------------
def test_canonical_digest() -> None:
    """Bit-level regression guard (environment sensitive, see docstring)."""
    got = digest(canonical_eigenvalues())
    assert got == EXPECTED_SHA256, (
        f"canonical spectrum digest changed:\n"
        f"  expected {EXPECTED_SHA256}\n"
        f"  got      {got}\n"
        f"Run `python {Path(__file__).name}` to see the eigenvalues. If only "
        f"the last of the thirteen significant digits moved this is a "
        f"LAPACK/BLAS difference, not a regression: check "
        f"test_canonical_values, which is the portable statement."
    )


def test_canonical_digest_is_reproducible() -> None:
    """Two builds in the same process must agree bit for bit."""
    assert digest(canonical_eigenvalues()) == digest(canonical_eigenvalues())


# Measured convergence of the canonical case, per principal quantum number.
# beta = 0.8 Z is the drift-minimizing scale for the DEEP states, so with
# N = 45 the ladder is at the roundoff floor up to n = 4 and still
# truncation-limited above it.  That is a property of the case, not a
# defect: the shallow levels are deliberately kept in the hash because
# their extra sensitivity makes them the best detectors of an accidental
# change in the assembly.
N_CONVERGED = 4          # n = 1..4 sit at the ~1e-14 roundoff floor
REL_FLOOR = 1e-12        # ceiling for those, as quoted in the manuscript
REL_TRUNCATION = 1e-3    # loose bound for n = 5..6 (measured 1e-9, 2e-5)


def test_canonical_values() -> None:
    """Portable check: the physics, at a tolerance, not the bytes."""
    ev = canonical_eigenvalues()

    # 1s_{1/2} against the closed-form Sommerfeld level.  This is the
    # manuscript's headline number, Table I: -4861.1979043696 Ha.
    exact_1s = exact_dirac_coulomb(1, KAPPA_CANON, Z_CANON)
    rel = abs(ev[0] - exact_1s) / abs(exact_1s)
    assert rel < REL_FLOOR, f"1s relative deviation {rel:.3e} exceeds 1e-12"
    assert abs(ev[0] - (-4861.1979043696)) / 4861.1979043696 < 1e-12

    # The ladder against Sommerfeld, at the tolerance each level supports.
    for n in range(1, N_LEVELS + 1):
        exact_n = exact_dirac_coulomb(n, KAPPA_CANON, Z_CANON)
        rel_n = abs(ev[n - 1] - exact_n) / abs(exact_n)
        tol = REL_FLOOR if n <= N_CONVERGED else REL_TRUNCATION
        assert rel_n < tol, f"n={n} relative deviation {rel_n:.3e} > {tol:.0e}"

    # Strictly increasing (deepest first) and all inside the mass gap.
    assert np.all(np.diff(ev) > 0.0)
    assert ev[0] > -2.0 * C_AU ** 2 and ev[-1] < 0.0


def test_canonical_build_is_certified() -> None:
    """C1 and C4, the two identities the cusp rule integrates exactly."""
    gamma = float(np.sqrt(KAPPA_CANON ** 2 - (Z_CANON / C_AU) ** 2))
    basis = RadialDiracBasis(
        NB_CANON, KAPPA_CANON, Z_CANON, BETA_FAC * Z_CANON,
        mu=gamma, quad="cusp", nquad=NB_CANON + NQUAD_EXTRA,
        cert_tol=CERT_TOL,
    )
    c1, _c2, _c3, c4 = basis.certs
    assert c1 < 1e-10, f"C1 overlap identity {c1:.3e}"
    assert c4 < 1e-10, f"C4 integration-by-parts identity {c4:.3e}"


if __name__ == "__main__":
    import platform
    import scipy

    ev = canonical_eigenvalues()
    got = digest(ev)
    print(f"python {platform.python_version()}  numpy {np.__version__}  "
          f"scipy {scipy.__version__}  ({platform.system()} "
          f"{platform.machine()})")
    print(f"canonical case: Z={Z_CANON:.0f} kappa={KAPPA_CANON:+d} "
          f"N={NB_CANON} beta={BETA_FAC}Z cusp rule "
          f"nquad={NB_CANON + NQUAD_EXTRA}")
    print(f"lowest {N_LEVELS} in-gap eigenvalues (binding E - c^2, Ha), "
          f"formatted {PRECISION!r}:")
    for line in format_eigenvalues(ev).splitlines():
        print(f"  {line}")
    print(f"sha256 = {got}")
    print("stored = " + EXPECTED_SHA256)
    print("MATCH" if got == EXPECTED_SHA256 else "MISMATCH (update "
          "EXPECTED_SHA256 only after confirming the numbers are right)")
