# Expected outputs (smoke tests)

All target values below are quoted from the manuscript/tables — nothing
new. A third party running the repo generators should observe:

1. **Certification (every build):** overlap identity, integration-by-parts
   identity and centrifugal identities pass; a deliberately injected sign
   error produces order-unity violations (see `crosscheck_assembly.py`).
2. **Point-nucleus table** (`ea2_point_table.py`): Z=92, κ=−1 ground
   binding −4861.197 904 3696 Ha with relative deviation 1.7×10⁻¹⁴ from
   the 40-digit Sommerfeld value; all 72 levels between 2×10⁻¹⁷ and
   1.3×10⁻¹³ (measured range over the `rel` column of
   `data/point_nucleus_table.csv`: 1.9323×10⁻¹⁷ to 1.2644×10⁻¹³);
   Z=136, κ=−1 (Zα=0.9924) converges cleanly with the
   distinguished-extension γ.
3. **Degeneracy theorem check:** the naive κ=+1 spurious root coincides
   with the κ=−1 ground state to 4×10⁻¹⁵–8×10⁻¹³ relative across
   Z=92–136 (`spur0` column of the table).
4. **Screened validation** (`ea1_yukawa_anchors.py`): with
   c = 137.0359895 (chain convention), the dimensionless invariant f
   matches Krauthauser–Hill to 4.8×10⁻¹⁷ (case 1) and 3×10⁻¹⁵ (case 2)
   (`dev_f_shifted` in `data/validation_errors.csv`: 4.7704895589362195×10⁻¹⁷
   and −2.9976021664879227×10⁻¹⁵).
5. **Critical screening** (`ea3_critical_screening.py`): s_c(Z→1, 1s)
   → 1.1905 (one part in 10⁴ of the known constant); 2p₁/₂–2p₃/₂
   splitting ≈ 40% at Zα=0.949 (Z=130), reproduced within a few percent
   by an independent near-threshold protocol.

## These five are now scripted asserts

Two pytest modules, both fast, cover the list above; no expectation is
left as prose only.

```
pytest tests/test_paper_A.py -q                        # 17 tests, ~11 s
pytest exploration/dirac/test_regression_hash.py -q    #  4 tests,  ~2 s
```

`tests/test_paper_A.py` maps one test class to each of the five
expectations. It recomputes in process everything that costs under a
second — all certification identities including the injected sign error,
the Z=92 and Z=136 headline levels, the spurious-root degeneracy, and
both Krauthauser–Hill cases — and validates the committed CSVs for the
72-level statistics and for expectation 5, whose generator takes minutes.
The full end-to-end version of all five is
`python scripts/run_paper_A.py` (≈ 10 min, 34 checks).

## Canonical regression digest (nice-to-have N2)

`exploration/dirac/test_regression_hash.py` pins one fast, fully
deterministic case bit for bit:

* Z = 92, κ = −1, point nucleus V = −Z/r
* cusp-adapted γ-Laguerre basis, μ = γ = √(1 − (Z/c)²), N = 45,
  β = 0.8 Z, generalized Gauss–Laguerre rule with nquad = 195,
  cert_tol = 1e−6
* mass-shifted pencil H − c²S with exactly summed Rayleigh refinement
* the six deepest in-gap eigenvalues (binding energies E − c², Ha),
  each formatted `{:.12e}`, joined with `\n` (LF), ASCII, no trailing
  newline, hashed with SHA-256

Exact command:

```
python exploration/dirac/test_regression_hash.py      # prints the digest
pytest exploration/dirac/test_regression_hash.py -q   # asserts it
```

Observed output, 2026-08-15:

```
-4.861197904370e+03
-1.257395852129e+03
-5.390933289938e+02
-2.952578385133e+02
-1.854851885913e+02
-1.270915634608e+02
sha256 = 77c61fe2ece0d1c233a5c07207d64f5e0ff25e87960cd9f016725d742e24c082
```

Environment in which that digest was obtained: Python 3.14.3 (CPython,
Windows 11, AMD64), numpy 2.4.3, scipy 1.18.0, LAPACK via
`scipy.linalg`. The full verification stack is in `README.md`.

### The digest is environment-sensitive — the numbers are the portable check

`{:.12e}` keeps thirteen significant digits, and these eigenvalues are
accurate to ~10⁻¹⁴ *relative*: the last digit sits at the roundoff floor
and depends on the numpy/LAPACK(BLAS) build, the CPU's FMA and vector
width, and the threading of the dense eigensolver. **On a different
stack the digest may legitimately differ while the physics is
untouched.** A mismatch is a signal to look, not by itself a failure.

Accordingly the same module also carries `test_canonical_values`, which
asserts the physics at a tolerance — the 1s level against the closed-form
Sommerfeld value at < 10⁻¹² relative, the ladder n = 1…4 likewise, and
the in-gap/ordering structure. **Those tolerance asserts, not the hash,
are what a third party should treat as the reproducibility criterion.**
The hash is the strictest possible statement that nothing at all changed
on *one* machine, which is what makes it useful as a development guard.

The same caveat applies to every quoted expectation above: they are all
roundoff-floor quantities, and each check in `run_paper_A.py` is a
bounded-interval check rather than an equality. Two measured examples of
that spread on the 2026-08-15 verification stack, both benign:

* expectation 2, worst relative deviation over the 72 levels: **1.6×10⁻¹³**
  here against the 1.3×10⁻¹³ of the production run — same order, and the
  scripted ceiling is set one decade above at 10⁻¹²;
* levels n = 5, 6 of the canonical case are truncation-limited at N = 45
  with β tuned for the deep states (1×10⁻⁹ and 1.6×10⁻⁵ relative). They
  are kept in the hash precisely because that extra sensitivity makes
  them the best detectors of an accidental change in the assembly.
