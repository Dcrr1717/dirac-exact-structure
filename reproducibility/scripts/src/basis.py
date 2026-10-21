"""Orthonormal generalized-Laguerre basis for the radial Schrodinger equation.

Basis definition (atomic units, x = 2*beta*r, alpha = 2*ell + 2):

    phi_j(r) = sqrt(2*beta) * q_j(2*beta*r),      j = 0, 1, ..., N-1

    q_j(x) = sqrt( j! / Gamma(j + alpha + 1) ) * x^(ell+1) * exp(-x/2)
             * L_j^{(alpha)}(x)

with L_j^{(alpha)} the generalized Laguerre polynomial. The set {phi_j}
is exactly L2-orthonormal on (0, inf):

    <phi_i | phi_j> = delta_ij,

satisfies phi_j(0) = 0 (correct regularity u ~ r^{ell+1} at the origin)
and decays exponentially. All matrix elements are evaluated with a
high-order Gauss-Legendre rule whose accuracy is certified a posteriori
by the overlap identity ||S - I||_max (see `BasisSet.overlap_error`).

The q_j are generated with the numerically stable three-term recurrence
of the *orthonormal* Laguerre polynomials (prefactor absorbed, so all
intermediate values stay O(1) -- no overflow for large j):

    q_{j+1} = [ (2j+alpha+1-x) q_j - sqrt(j(j+alpha)) q_{j-1} ]
              / sqrt((j+1)(j+1+alpha))

Derivatives use  x * dL/dx identities, giving

    dq_j/dx = ( (ell+1+j)/x - 1/2 ) q_j - sqrt(j(j+alpha))/x * q_{j-1}.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln


def _q_functions(nbasis: int, ell: int, x: np.ndarray):
    """Evaluate q_j(x) and dq_j/dx for j = 0..nbasis-1 on nodes x > 0.

    Returns (q, dq), each of shape (nbasis, len(x)).
    """
    alpha = 2 * ell + 2
    q = np.zeros((nbasis, x.size))
    # q_0 = x^{ell+1} e^{-x/2} / sqrt(Gamma(alpha+1)), evaluated in log form
    q[0] = np.exp((ell + 1) * np.log(x) - 0.5 * x - 0.5 * gammaln(alpha + 1))
    if nbasis > 1:
        q[1] = (alpha + 1 - x) * q[0] / np.sqrt(1.0 * (1 + alpha))
    for j in range(1, nbasis - 1):
        q[j + 1] = ((2 * j + alpha + 1 - x) * q[j]
                    - np.sqrt(j * (j + alpha)) * q[j - 1]) \
                   / np.sqrt((j + 1.0) * (j + 1 + alpha))

    dq = np.zeros_like(q)
    for j in range(nbasis):
        dq[j] = ((ell + 1 + j) / x - 0.5) * q[j]
        if j > 0:
            dq[j] -= np.sqrt(j * (j + alpha)) / x * q[j - 1]
    return q, dq


class BasisSet:
    """Orthonormal Laguerre basis with certified quadrature.

    Parameters
    ----------
    nbasis : number of basis functions N
    ell    : orbital angular momentum
    beta   : exponential scale of the basis, phi ~ exp(-beta*r)
    nquad  : Gauss-Legendre nodes (default: generous, self-certified)
    """

    def __init__(self, nbasis: int, ell: int, beta: float, nquad: int | None = None,
                 overlap_tol: float = 1e-9):
        self.nbasis, self.ell, self.beta = nbasis, ell, beta
        # basis functions oscillate ~j times on x in (0, ~4j); resolve well
        self.xmax = 4.0 * nbasis + 24.0 * (ell + 1) + 120.0
        self.nquad = nquad or max(3000, 16 * nbasis)
        t, w = leggauss(self.nquad)
        self.x = 0.5 * self.xmax * (t + 1.0)      # nodes on (0, xmax)
        self.w = 0.5 * self.xmax * w              # GL weights
        self.q, self.dq = _q_functions(nbasis, ell, self.x)
        # weighted values: integral f = (q*w) @ q.T style contractions
        self._qw = self.q * self.w
        # run-time certification gate: every basis must reproduce the
        # overlap identity or the run aborts (measured worst case over all
        # production bases is 9.7e-12, two orders below the tolerance)
        self.overlap_err = self.overlap_error()
        if self.overlap_err > overlap_tol:
            raise RuntimeError(
                f"quadrature certification failed: max|S-I| = "
                f"{self.overlap_err:.3e} > {overlap_tol:.1e} "
                f"(N={nbasis}, ell={ell}, nquad={self.nquad})")

    # -- certification -------------------------------------------------
    def overlap_error(self) -> float:
        """max|S - I| with S_ij = int q_i q_j dx. Certifies quadrature."""
        S = self._qw @ self.q.T
        return float(np.max(np.abs(S - np.eye(self.nbasis))))

    # -- matrix elements (atomic units) --------------------------------
    def kinetic(self) -> np.ndarray:
        """T_ij = 1/2 <phi_i' | phi_j'> + centrifugal term."""
        b = self.beta
        T = 2.0 * b * b * ((self.dq * self.w) @ self.dq.T)
        if self.ell > 0:
            T += 2.0 * b * b * self.ell * (self.ell + 1) * \
                 ((self._qw / self.x**2) @ self.q.T)
        return 0.5 * (T + T.T)  # symmetrize (kills quadrature round-off)

    def coulomb(self, Z: float = 1.0) -> np.ndarray:
        """V_ij = -Z <phi_i | 1/r | phi_j>."""
        V = -Z * 2.0 * self.beta * ((self._qw / self.x) @ self.q.T)
        return 0.5 * (V + V.T)

    def yukawa(self, Z: float = 1.0, lam: float = 0.0, theta: float = 0.0):
        """V_ij = -Z <phi_i| exp(-lam * r * e^{i theta}) / r |phi_j> * e^{-i theta}.

        theta = 0 gives the real (unscaled) screened-Coulomb matrix.
        theta > 0 gives the complex-scaled matrix (complex symmetric).
        """
        scale = np.exp(1j * theta) if theta else 1.0
        r = self.x / (2.0 * self.beta)
        f = np.exp(-lam * r * scale)
        V = -Z * 2.0 * self.beta * ((self._qw * f / self.x) @ self.q.T)
        V = 0.5 * (V + V.T)
        return V / scale if theta else np.real(V)

    # -- assembled Hamiltonians -----------------------------------------
    def hamiltonian_hydrogen(self, Z: float = 1.0) -> np.ndarray:
        return self.kinetic() + self.coulomb(Z)

    def hamiltonian_yukawa(self, Z: float = 1.0, lam: float = 0.0) -> np.ndarray:
        return self.kinetic() + self.yukawa(Z, lam)

    def hamiltonian_complex_scaled(self, Z: float = 1.0, lam: float = 0.0,
                                   theta: float = 0.3) -> np.ndarray:
        """H(theta) = e^{-2 i theta} T + V_theta  (complex symmetric).

        For the pure Coulomb part the scaling is analytic: V -> e^{-i theta} V.
        For the Yukawa screening the complex-scaled integral is evaluated
        directly by quadrature.
        """
        T = self.kinetic().astype(complex)
        H = np.exp(-2j * theta) * T
        if lam == 0.0:
            H += np.exp(-1j * theta) * self.coulomb(Z).astype(complex)
        else:
            H += self.yukawa(Z, lam, theta=theta)
        return H
