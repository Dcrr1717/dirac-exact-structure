# Numerical conventions (Paper A)

- Hartree atomic units: ħ = m_e = 1, c = 1/α.
- **α convention:** 1/α = 137.035999084 (CODATA 2018) for all spectra of
  this paper (`C_AU` in `src/dirac.py`).
- **Comparison-chain caveat:** the forty-digit screened-Coulomb chain
  (Krauthauser–Hill 2002 → Baye et al. 2014 → Chen–Shao–Wu 2026) uses the
  older, undeclared value c = 137.0359895; reproducing their last digits
  requires adopting it. The convention-safe quantity is the dimensionless
  invariant f = E/c².
- Binding energies are E − c² (mass-shifted pencil; the rest-mass
  cancellation is removed analytically, not by subtraction).
- Basis: orthonormal generalized Laguerre functions with prefactor power
  μ = γ = √(κ² − (Zα)²), scale β chosen by minimal inter-basis drift
  |E(N=45) − E(N=30)|, never by proximity to the reference value.
- Quadrature: generalized Gauss–Laguerre with weight x^{2μ−1}e^{−x}
  (exact for overlap, Coulomb and coupling integrals).
- Certification tolerances: run-time identities C1 (overlap), C2/C3
  (operator products / integration by parts / centrifugal coefficients);
  structural errors give order-unity violations, converged-quadrature
  noise sits far below — a single tolerance separates them.
- Sommerfeld references evaluated in 40-digit decimal arithmetic
  (`decimal` module, prec = 40).
