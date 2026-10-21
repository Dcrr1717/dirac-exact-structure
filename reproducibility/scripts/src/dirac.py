"""Radial Dirac-Coulomb solver on orthonormal generalized-Laguerre bases.

Three discretizations of the radial Dirac operator

    h_kappa = [[ V + c^2,  c*O+ ],          O+/- = -/+ d/dr + kappa/r,
               [ c*O-,     V - c^2 ]]       V = -Z/r,

are provided: NAIVE (independent Laguerre sets for both components), RKB
(restricted kinetic balance) and DKB (dual kinetic balance, Shabaev et al.,
PRL 93, 130405 (2004)), all built on the L2-orthonormal Laguerre functions
phi_j ~ r^(ell+1) e^(-beta r) with ell(ell+1) = kappa(kappa+1).

Operator identities used (order matters!):

    O- O+ u = -u'' + kappa(kappa-1) u / r^2
    O+ O- u = -u'' + kappa(kappa+1) u / r^2

CERTIFIED ASSEMBLY: every build checks, at run time, the internal
identities that the discrete matrices must satisfy exactly (up to
quadrature/roundoff),

    C1:  <phi_i | phi_j>                  = delta_ij
    C2:  <phi | O+O- phi>  (via d2f)      = <O- phi | O- phi>   (= 4c^2 <g|g>)
    C3:  <phi | O-O+ phi>  (via d2f)      = <O+ phi | O+ phi>   (= 4c^2 <h|h>)

C2/C3 certify simultaneously the second-derivative recurrence, the
quadrature near the r->0 singularity, the integration-by-parts boundary
terms, AND the centrifugal coefficients kappa(kappa+-1) -- i.e. exactly the
class of bug that a wrong operator-product expansion introduces.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, roots_genlaguerre
from scipy.linalg import eigh, eigvalsh_tridiagonal

C_AU = 137.035999084          # 1/alpha, CODATA 2018

_LEGGAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def exact_dirac_coulomb(n: int, kappa: int, Z: float, c: float = C_AU) -> float:
    """Point-nucleus Dirac-Coulomb binding energy E - c^2 (Hartree)."""
    g = np.sqrt(kappa * kappa - (Z / c) ** 2)
    nr = n - abs(kappa)
    return c * c / np.sqrt(1.0 + (Z / c / (nr + g)) ** 2) - c * c


def exact_ladder(kappa: int, Z: float, kmax: int = 12) -> np.ndarray:
    """First kmax exact binding energies of channel kappa (ascending)."""
    n0 = abs(kappa) if kappa < 0 else kappa + 1
    return np.array([exact_dirac_coulomb(n, kappa, Z) for n in range(n0, n0 + kmax)])


def laguerre_funcs(nb: int, mu: float, x: np.ndarray):
    """Orthonormal Laguerre functions with REAL prefactor power mu > 0,

        q_j(x) = sqrt(j!/Gamma(j+2mu+1)) x^mu e^{-x/2} L_j^{(2mu)}(x),

    plus first and second derivatives (stable orthonormal recurrence).
    Integer case mu = ell+1 recovers the nonrelativistic basis; the
    cusp-adapted relativistic case uses mu = gamma = sqrt(kappa^2-(Z/c)^2)
    (Grant's L-spinor prefactor), for which the point-nucleus Dirac
    eigenfunctions are finite linear combinations of the q_j.
    """
    a = 2.0 * mu
    q = np.zeros((nb, x.size))
    q[0] = np.exp(mu * np.log(x) - 0.5 * x - 0.5 * gammaln(a + 1))
    if nb > 1:
        q[1] = (a + 1 - x) * q[0] / np.sqrt(1.0 * (1 + a))
    for j in range(1, nb - 1):
        q[j + 1] = ((2 * j + a + 1 - x) * q[j]
                    - np.sqrt(j * (j + a)) * q[j - 1]) / np.sqrt((j + 1.0) * (j + 1 + a))
    # first derivatives (complete them BEFORE any second derivative uses them)
    dq = np.zeros_like(q)
    for j in range(nb):
        dq[j] = ((mu + j) / x - 0.5) * q[j]
        if j > 0:
            dq[j] -= np.sqrt(j * (j + a)) / x * q[j - 1]
    # second derivatives from the *complete* dq
    d2q = np.zeros_like(q)
    for j in range(nb):
        A = (mu + j) / x - 0.5
        Ap = -(mu + j) / x ** 2
        d2q[j] = Ap * q[j] + A * dq[j]
        if j > 0:
            B = np.sqrt(j * (j + a)) / x
            Bp = -np.sqrt(j * (j + a)) / x ** 2
            d2q[j] -= Bp * q[j - 1] + B * dq[j - 1]
    return q, dq, d2q


def laguerre_funcs_scaled(nb: int, mu: float, x: np.ndarray,
                          lognorm: np.ndarray):
    """Same functions as laguerre_funcs, but each value is returned
    multiplied by exp(lognorm) (a per-node log-factor, e.g. log sqrt(W)),
    with the recurrence run under a per-node running exponent so that
    NOTHING under- or overflows even where q_0 alone is far below the
    double-precision minimum.  Intended for the fused basis*sqrt(weight)
    representation: products q_j*sqrt(W) are O(1) wherever the weight is
    nonzero, so the outputs are always representable.  Nodes with
    lognorm = -inf (dropped quadrature nodes) yield exactly 0.
    """
    a = 2.0 * mu
    dead = ~np.isfinite(lognorm)
    ln = np.where(dead, 0.0, lognorm)
    S = mu * np.log(x) - 0.5 * x - 0.5 * gammaln(a + 1)     # log q_0
    q = np.zeros((nb, x.size))

    def emit(j, qj):
        f = np.where(dead, 0.0, np.exp(np.clip(S + ln, -745.0, 700.0)))
        q[j] = qj * f

    q0 = np.ones_like(x)                                    # scaled q_0
    emit(0, q0)
    if nb > 1:
        q1 = (a + 1 - x) / np.sqrt(1.0 + a) * q0
        emit(1, q1)
        for j in range(1, nb - 1):
            q2 = ((2 * j + a + 1 - x) * q1 - np.sqrt(j * (j + a)) * q0) \
                / np.sqrt((j + 1.0) * (j + 1 + a))
            big = np.abs(q2) > 1e100
            if big.any():
                f = np.where(big, np.abs(q2), 1.0)
                q0, q1, q2 = q0 / f, q1 / f, q2 / f
                S = S + np.log(f)
            emit(j + 1, q2)
            q0, q1 = q1, q2
    # first and second derivatives from the EMITTED (true-scale) values,
    # exactly as in laguerre_funcs -- the relations are linear in the
    # q's with j-independent per-node coefficients, so the fused
    # sqrt(weight) factor passes through them unchanged.
    dq = np.zeros_like(q)
    for j in range(nb):
        dq[j] = ((mu + j) / x - 0.5) * q[j]
        if j > 0:
            dq[j] -= np.sqrt(j * (j + a)) / x * q[j - 1]
    d2q = np.zeros_like(q)
    for j in range(nb):
        A = (mu + j) / x - 0.5
        Ap = -(mu + j) / x ** 2
        d2q[j] = Ap * q[j] + A * dq[j]
        if j > 0:
            B = np.sqrt(j * (j + a)) / x
            Bp = -np.sqrt(j * (j + a)) / x ** 2
            d2q[j] -= Bp * q[j - 1] + B * dq[j - 1]
    return q, dq, d2q


def cusp_rule(nq: int, mu: float):
    """Generalized Gauss-Laguerre rule (weight x^(2mu-1) e^{-x}) in the
    dx-measure convention used by RadialDiracBasis, computed in
    LOG-SCALED arithmetic so it works at any nq.

    Nodes: eigenvalues of the Jacobi matrix of L^(alpha), alpha = 2mu-1
    (symmetric tridiagonal; diag 2k+alpha+1, offdiag sqrt(k(k+alpha))),
    via a dedicated tridiagonal eigensolver -- no NaN breakdown.

    Weights: with p_k the orthonormal polynomials w.r.t. x^alpha e^{-x},
    the Gauss weight is 1/sum_k p_k(x)^2, and the dx-measure weight for
    integrands carrying e^{-x/2} per factor is
        W(x) = 1 / sum_k phi_k(x)^2,
    phi_k(x) = p_k(x) x^{alpha/2} e^{-x/2} the orthonormal Laguerre
    FUNCTIONS with prefactor power mu - 1/2.  At far-tail nodes phi_0
    starts below double-precision minimum, so the recurrence is run with
    a per-node running exponent (renormalize when the scaled values
    exceed 1e100, accumulate the log): the summed square is obtained as
    log(sum) exactly, with no under/overflow.  Nodes whose true weight
    exceeds e^600 are dropped (W = 0): any product of two basis
    functions with index <= nb is smaller than e^{-4(nq-nb)} relative
    there, so with the standard margin nq >= nb + 50 the dropped
    contribution is far below machine precision -- and the C1/C4
    certifications verify every build regardless.  This removes both
    scipy failure modes (NaN for nq >~ 350; genuine tail-weight
    underflow past x ~ 709) and lifts the old cusp-basis ceiling
    nb <~ 170.
    """
    a2 = 2.0 * mu - 1.0                     # Laguerre alpha
    k = np.arange(nq, dtype=float)
    x = eigvalsh_tridiagonal(2 * k + a2 + 1, np.sqrt(k[1:] * (k[1:] + a2)))
    # scaled recurrence for phi_k (prefactor power mup = mu - 1/2):
    # q0 folded into the running exponent S, so the scaled q0 is 1.
    mup = mu - 0.5
    a = 2.0 * mup
    S = mup * np.log(x) - 0.5 * x - 0.5 * gammaln(a + 1)   # log phi_0
    q0 = np.ones_like(x)
    acc = q0 * q0
    q1 = (a + 1 - x) / np.sqrt(1.0 + a) * q0
    acc += q1 * q1
    for j in range(1, nq - 1):
        q2 = ((2 * j + a + 1 - x) * q1
              - np.sqrt(j * (j + a)) * q0) / np.sqrt((j + 1.0) * (j + 1 + a))
        big = np.abs(q2) > 1e100
        if big.any():
            f = np.where(big, np.abs(q2), 1.0)
            q1, q2, acc = q1 / f, q2 / f, acc / (f * f)
            S = S + np.log(f)
        acc += q2 * q2
        q0, q1 = q1, q2
    logsum = 2.0 * S + np.log(acc)          # log sum_k phi_k^2
    with np.errstate(over="ignore"):
        w = np.where(logsum > -600.0,
                     np.exp(-np.maximum(logsum, -600.0)), 0.0)
    logw = np.where(logsum > -600.0, -logsum, -np.inf)
    return x, w, logw


class RadialDiracBasis:
    """Laguerre machinery for one (kappa, Z, beta) channel, with certified
    quadrature and the DKB/RKB/naive Hamiltonians."""

    def __init__(self, nb: int, kappa: int, Z: float, beta: float,
                 nquad: int | None = None, xmax: float | None = None,
                 c: float = C_AU, cert_tol: float = 1e-7,
                 mu: float | None = None, quad: str = "legendre"):
        # cert_tol calibration: structural assembly bugs show up as O(0.1-1)
        # relative violations; converged-quadrature noise sits at <~1e-9
        # (the O+ channel of ell=0 has a non-vanishing integrand at r=0 and
        # is the slowest to converge). 1e-7 separates the two cleanly.
        # mu: prefactor power. Default ell+1 (nonrelativistic regularity);
        # pass mu = sqrt(kappa^2-(Z/c)^2) for the cusp-adapted basis. NOTE:
        # the C2/C3 boundary terms scale as r^(2mu-1), so for mu < 1/2
        # (Z ~> 118 at |kappa|=1) those certifications legitimately degrade.
        assert kappa != 0
        self.nb, self.kappa, self.Z, self.beta, self.c = nb, kappa, Z, beta, c
        self.ell = kappa if kappa > 0 else -kappa - 1
        self.mu = float(mu) if mu is not None else float(self.ell + 1)

        if quad == "cusp":
            # Generalized Gauss-Laguerre rule with weight x^(2mu-1) e^{-x}.
            # Every product of two basis-family functions carries exactly
            # x^(2mu) e^{-x} (or x^(2mu-1) for the 1/r elements), so the
            # overlap, Coulomb and O+- coupling integrands are POLYNOMIAL
            # times this weight: the rule is EXACT for nq >= nb + a few.
            # (1/r^2-type elements, i.e. C2/C3 and the DKB cross block,
            # remain inexact and require mu > 1/2 for convergence at all.)
            nq = nquad or (nb + 30)
            # own Golub-Welsch rule in function space (see cusp_rule):
            # removes scipy's NaN breakdown (nq >~ 350) and the genuine
            # tail-weight underflow past x ~ 709 that capped nb at ~170.
            x, w, logw = cusp_rule(nq, self.mu)
            if not np.isfinite(x).all():
                raise RuntimeError(
                    f"cusp quadrature breakdown at nq={nq}, "
                    f"mu={self.mu:.3f}: non-finite nodes")
        else:
            nq = nquad or max(4000, 16 * nb)
            if nq not in _LEGGAUSS_CACHE:
                _LEGGAUSS_CACHE[nq] = leggauss(nq)
            t, w0 = _LEGGAUSS_CACHE[nq]
            X = xmax or (4.0 * nb + 24.0 * (self.ell + 1) + 200.0)
            x = 0.5 * X * (t + 1.0)
            w = 0.5 * X * w0

        b = beta
        if quad == "cusp":
            # fused representation: sqrt(W) folded into the functions via
            # the log-scaled recurrence (laguerre_funcs_scaled) -- every
            # stored value is O(1) at every node, so arbitrarily large nb
            # works; the effective quadrature weight becomes constant.
            q, dq, d2q = laguerre_funcs_scaled(nb, self.mu, x, 0.5 * logw)
            w = np.ones_like(x)
        else:
            q, dq, d2q = laguerre_funcs(nb, self.mu, x)
        self.r = x / (2 * b)
        self.W = w / (2 * b)                       # dr-measure weights
        self.f = np.sqrt(2 * b) * q
        self.df = (2 * b) ** 1.5 * dq
        self.d2f = (2 * b) ** 2.5 * d2q
        r = self.r
        self.g = (self.df + (kappa / r) * self.f) / (2 * c)     # (1/2c) O- f
        self.h = (-self.df + (kappa / r) * self.f) / (2 * c)    # (1/2c) O+ f
        # operator products -- ORDER MATTERS:
        self.OmOp = -self.d2f + kappa * (kappa - 1) / r ** 2 * self.f  # O-O+ f
        self.OpOm = -self.d2f + kappa * (kappa + 1) / r ** 2 * self.f  # O+O- f
        self.V = -Z / r

        # ---- certified assembly ------------------------------------
        ME = self._me
        Kg = 4 * c * c * ME(self.g, self.g)      # <O- f | O- f>
        Kh = 4 * c * c * ME(self.h, self.h)      # <O+ f | O+ f>
        c1 = np.max(np.abs(ME(self.f, self.f) - np.eye(nb)))
        c2 = np.max(np.abs(ME(self.f, self.OpOm) - Kg)) / max(1.0, np.max(np.abs(Kg)))
        c3 = np.max(np.abs(ME(self.f, self.OmOp) - Kh)) / max(1.0, np.max(np.abs(Kh)))
        # I1 (IBP identity <f_i|O+ f_j> = <O- f_i|f_j>): catches O+/O- sign
        # errors and boundary-term violations in one shot.
        c4 = np.max(np.abs(ME(self.f, self.h) - ME(self.g, self.f)))
        self.certs = (c1, c2, c3, c4)            # relative to natural scale
        # cusp rule: C1/C4 integrands are exact (polynomial x weight);
        # C2/C3 involve 1/r^2 and are legitimately inexact -> report only.
        enforced = (c1, c4) if quad == "cusp" else self.certs
        # NaN-proof: 'not (x <= tol)' fails on NaN, unlike 'x > tol'
        if not (max(enforced) <= cert_tol):
            raise RuntimeError(f"assembly certification failed: {self.certs}")

    # -- integrals ----------------------------------------------------
    def _me(self, u, v):
        return np.einsum('ik,jk,k->ij', u, v, self.W)

    def _mev(self, u, p, v):
        return np.einsum('ik,jk,k->ij', u, p * v, self.W)

    # -- solvers --------------------------------------------------------
    def _ingap(self, H, S):
        ev = eigh(0.5 * (H + H.T), 0.5 * (S + S.T), eigvals_only=True)
        ev = np.sort(ev) - self.c ** 2
        return ev[(ev > -2 * self.c ** 2 + 1e-6) & (ev < -1e-9)]

    def _rayleigh(self, Hs, S, x):
        num = math.fsum((x[:, None] * Hs * x[None, :]).ravel().tolist())
        den = math.fsum((x[:, None] * S * x[None, :]).ravel().tolist())
        return num / den

    def _ingap_shifted(self, Hs, S):
        """Gap eigenvalues of the SHIFTED pencil (H - c^2 S, S): these are
        binding energies directly, so the rest-mass cancellation floor
        (2 c^2 eps_mach ~ 8e-12 Ha) never enters a subtraction.  The dense
        eigensolver still carries an absolute backward error ~ ||H||*eps
        ~ 2c^2*eps, so each gap eigenvalue is refined by a Rayleigh
        quotient of its eigenvector evaluated with exactly-rounded
        summation (math.fsum).  This works because in the shifted pencil
        the only O(c^2) block, -2c^2 S in the lower-lower corner, is
        contracted against the SMALL components of an electronic state:
        every term of the quadratic form is O(|E_binding|) and fsum
        removes the summation cancellation exactly.  Net accuracy
        (adversarially verified): ~1e-14 RELATIVE to the binding energy,
        i.e. 50-100 eps -- the floor is set by the pre-rounding of the
        individual products x_i*Hs_ij*x_j and by assembly rounding of the
        O(c^2) entries, which no refinement of the assembled pencil can
        remove.  Caveats: the second-order eigenvector term
        (~(n eps 2c^2/gap)^2) is negligible for level gaps >~ 1e-4 Ha
        (verified) but untested below; for RKB the ill-conditioned
        small-component Gram (cond(Sss) ~ 1e6-1e7) inflates the
        eigensolver floor to ~1e-9 absolute.
        """
        Hs = 0.5 * (Hs + Hs.T)
        S = 0.5 * (S + S.T)
        ev, U = eigh(Hs, S)
        sel = np.nonzero((ev > -2 * self.c ** 2 + 1e-6) & (ev < -1e-9))[0]
        return np.sort([self._rayleigh(Hs, S, U[:, k]) for k in sel])

    def solve_dkb(self, s: int = -1) -> np.ndarray:
        """DKB, Shabaev sign convention (PRL 93, 130405):

            Psi+_j = (  f_j              , +(1/2c) O- f_j )
            Psi-_j = ( s*(1/2c) O+ f_j   ,  f_j           ),   s = -1.

        The MINUS sign is load-bearing: it flips one component only, which
        changes the span (not a basis-sign congruence). It follows from the
        positronic limit P = c O+ Q / (E - c^2 - V) -> -(1/2c) O+ Q at
        E ~ -c^2, and it is what gives the H-- block the correct
        charge-conjugate nonrelativistic limit (kinetic +(1/2)<O+f|O+f>).
        With s=+1 the positronic limit has the wrong-sign kinetic term and
        spurious states appear deep inside the gap (observed empirically;
        the anti-balanced mechanism is proven for bounded V in FINDINGS
        #12). The s parameter exists ONLY for pollution experiments;
        physical calculations must use the default s = -1.
        """
        assert s in (-1, +1)
        c, V = self.c, self.V
        f, g, h = self.f, self.g, self.h
        ME, MEv = self._me, self._mev
        Sff = ME(f, f)
        Spp = Sff + ME(g, g)
        Spm = s * ME(f, h) + ME(g, f)
        Smm = ME(h, h) + Sff
        Hpp = (MEv(f, V, f) + c * c * Sff
               + 4 * c * c * ME(g, g)                   # <f|cO+|g> + <g|cO-|f>
               + MEv(g, V, g) - c * c * ME(g, g))
        Hmm = (MEv(h, V, h) + c * c * ME(h, h)
               + s * 4 * c * c * ME(h, h)               # <s h|cO+|f>+<f|cO-|s h>
               + MEv(f, V, f) - c * c * Sff)
        Hpm = (s * (MEv(f, V, h) + c * c * ME(f, h))    # <f|(V+c^2)|s h>
               + 2 * c * c * ME(f, h)                   # <f|cO+|f> = 2c^2<f|h>
               + s * 0.5 * ME(g, self.OmOp)             # <g|cO-|s h>
               + MEv(g, V, f) - c * c * ME(g, f))
        H = np.block([[Hpp, Hpm], [Hpm.T, Hmm]])
        S = np.block([[Spp, Spm], [Spm.T, Smm]])
        return self._ingap(H, S)

    def solve_rkb(self, shifted: bool = False) -> np.ndarray:
        """RKB: large in {f_j}, small in {O- f_j} with its Gram metric.

        shifted=True: analytic H - c^2 S pencil + Rayleigh refinement
        (see _ingap_shifted)."""
        c, V = self.c, self.V
        f = self.f
        chi = 2 * c * self.g                          # O- f
        ME, MEv = self._me, self._mev
        Sll = ME(f, f)
        Sss = ME(chi, chi)
        # <f_i | c O+ | chi_j> = c <O- f_i | chi_j> = c (Sss)_ij  (by parts)
        S = np.block([[Sll, 0 * Sll], [0 * Sll, Sss]])
        if shifted:
            Hs = np.block([[MEv(f, V, f), c * Sss],
                           [c * Sss.T, MEv(chi, V, chi) - 2 * c * c * Sss]])
            return self._ingap_shifted(Hs, S)
        Hll = MEv(f, V, f) + c * c * Sll
        Hss = MEv(chi, V, chi) - c * c * Sss
        H = np.block([[Hll, c * Sss], [c * Sss.T, Hss]])
        return self._ingap(H, S)

    def solve_naive(self, shifted: bool = False) -> np.ndarray:
        """Both components in the SAME Laguerre set (pollution showcase).

        shifted=True assembles the pencil H - c^2 S analytically (the mass
        term survives only as -2c^2 in the lower-lower block) and refines
        every gap eigenvalue with the exactly-summed Rayleigh quotient;
        see _ingap_shifted."""
        c, V = self.c, self.V
        f, df = self.f, self.df
        kappa, r = self.kappa, self.r
        ME, MEv = self._me, self._mev
        Opf = -df + kappa / r * f
        Hvv = MEv(f, V, f)
        Hoff = c * ME(f, Opf)
        Sff = ME(f, f)
        S = np.block([[Sff, 0 * Sff], [0 * Sff, Sff]])
        if shifted:
            Hs = np.block([[Hvv, Hoff],
                           [Hoff.T, Hvv - 2 * c * c * Sff]])
            return self._ingap_shifted(Hs, S)
        H = np.block([[Hvv + c * c * Sff, Hoff],
                      [Hoff.T, Hvv - c * c * Sff]])
        return self._ingap(H, S)


def validate(kappa: int, Z: float, nb: int = 60, betas=None,
             method: str = "dkb", kcheck: int = 4):
    """Best-over-beta validation of the lowest kcheck states vs exact.

    Returns dict with per-state errors, the variational-violation flag
    (any Galerkin level strictly BELOW its exact counterpart), and certs.
    """
    betas = betas if betas is not None else [0.5 * max(Z, 1.0), 0.9 * max(Z, 1.0), 1.4 * max(Z, 1.0)]
    ladder = exact_ladder(kappa, Z)
    best = None
    for b in betas:
        B = RadialDiracBasis(nb, kappa, Z, b)
        gap = getattr(B, f"solve_{method}")()
        if gap.size < kcheck:
            continue
        errs = gap[:kcheck] - ladder[:kcheck]
        score = abs(errs[0])
        if best is None or score < best["err"][0].__abs__():
            best = dict(beta=b, err=errs, levels=gap[:kcheck], certs=B.certs,
                        viol=bool(np.any(errs < -1e-8 * np.maximum(1.0, np.abs(ladder[:kcheck])))))
    return best
