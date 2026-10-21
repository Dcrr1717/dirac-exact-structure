# Paper A — reproducibility

The solver and all generators exist as a working repository:
`https://github.com/Dcrr1717/feast-spectral`
(`NEEDS_HUMAN`: confirm public visibility / mint DOI before citing it in
the submitted manuscript; upon acceptance, deposit in the CPC Program
Library as the journal requires).

Frozen copies of the files listed below are vendored in `scripts/` — see
`scripts/README.md` for what is and is not included and how to run them.

## Two commands

From the root of the repository (or of `scripts/`, which mirrors its
layout):

```
pip install -r requirements.txt
python scripts/run_paper_A.py
```

That runs the four generators end to end and asserts their output against
the five expectations of `EXPECTED_OUTPUTS.md`. ≈ 10 min
(2026-08-15: 600 s wall, of which `ea3_critical_screening.py` is 572 s);
`--skip-slow` drops that stage and finishes in ≈ 30 s, reporting
expectation 5 as SKIP. Exit status is 0 only if every executed check
passed.

The fast gate, ≈ 15 s:

```
pytest tests/test_paper_A.py exploration/dirac/test_regression_hash.py -q
```

## What regenerates what

| Paper element | Generator (repo path) |
|---|---|
| Fig. 1 (methodology pipeline) | `scripts/make_pipeline_figure_dirac.py` → `figures/pipeline_dirac.pdf` (copy next to the manuscript as `figA_pipeline.pdf`) |
| Table I (point-nucleus binding energies) | `exploration/dirac/ea2_point_table.py` → `ea2_point_table.csv` (copy in `../data/point_nucleus_table.csv`) |
| Fig. 2 (convergence) | `exploration/dirac/ea4_convergence_and_figures.py` (+ `results/EA4_convergence.csv`, copy in `../data/EA4_convergence.csv`) |
| Screened-Coulomb validation / adjudication (Sec. IV) | `exploration/dirac/ea1_yukawa_anchors.py` → `validation_errors.csv` (copy in `../data/validation_errors.csv`) |
| Fig. 3 / critical-screening map (Sec. VI) | `exploration/dirac/ea3_critical_screening.py` → `ea3_critical_screening.csv` (copy in `../data/critical_screening_map.csv`) |
| Pollution anatomy (Sec. V): certification, DKB form-domain violation, degeneracy theorem check | `exploration/dirac/crosscheck_assembly.py`, `validate_dkb.py` (both vendored); `atomic_balance_dkb.py`, `pollution_demo.*` (**not vendored**) |
| Leakage exponents / conformity defects | `exploration/dirac/l1prime_fiber_lemma.py`, `l2_global_defect.py`, `l3_cusp_rate.py` — **not vendored** |

> **Scope of this vendored package.** The rows above marked *not
> vendored* name generators that live in the upstream repository and are
> **not** included in `scripts/`. Concretely, the leakage exponents and
> conformity defects of Sec. V cannot be regenerated from this package
> alone; the certification and DKB checks of that section can.
> `NEEDS_HUMAN`: vendor those five generators, or state the limitation
> explicitly in the Data-and-code-availability section, before the CPC
> deposit — CPC evaluates whether the deposited program reproduces the
> published results.
| Core engine | `src/dirac.py` (certified assembly C1–C3, shifted pencil, Rayleigh refinement), `src/basis.py` |
| Unit tests (script-style, pre-registered demonstrations) | `exploration/dirac/test_gamma_basis.py`, `test_cusp_quadrature.py`, `test_shifted_pencil.py` |
| Scripted asserts (pytest) | `tests/test_paper_A.py`, `exploration/dirac/test_regression_hash.py` |

The generators overwrite their own CSV outputs in place. A fresh run
reproduces the committed versions to roundoff (~1e-13 relative), not bit
for bit.

## Conventions

See `CONVENTIONS.md`. Key values: c = 1/α = 137.035999084 (CODATA 2018;
`C_AU` in `src/dirac.py`); binding = E − c² in Hartree; Sommerfeld
references in 40-digit decimal arithmetic. The single exception is
`ea1_yukawa_anchors.py` / `validation_errors.csv`, which uses
c = 137.0359895 (CODATA 1986) because that is the convention of the
published chain it adjudicates; it is carried in the CSV's
`c_convention` column.

## Environment

### Declared floors (`requirements.txt` / `pyproject.toml`)

Python ≥ 3.10, numpy ≥ 1.26, scipy ≥ 1.11, matplotlib ≥ 3.8.
Testing: pytest ≥ 7.0. LAPACK via `scipy.linalg`. The forty-digit
Sommerfeld reference values are computed with the standard library
`decimal` module, so no arbitrary-precision third-party package is
required. There are no other dependencies and no compiled extensions of
our own.

### Verification environment (checks re-run 2026-08-15)

The whole reproduction chain in this directory was re-executed on:

| Component | Version |
|---|---|
| Python | 3.14.3 (CPython, Windows 11, AMD64) |
| numpy | 2.4.3 |
| scipy | 1.18.0 |
| matplotlib | 3.11.0 |
| sympy | 1.14.0 |
| pytest | 9.1.1 |

Result on that stack: `python scripts/run_paper_A.py` → 34 passed,
0 failed, 0 skipped, 600 s wall, exit status 0; `pytest
tests/test_paper_A.py` → 17 tests, 11 s; `pytest
exploration/dirac/test_regression_hash.py` → 4 passed, 2 s.

This is a *verification* environment, well above every declared floor,
not the environment the production numbers were first computed in.

Note on the 17: three of them validate the critical-screening map and
need `exploration/dirac/ea3_critical_screening.csv`, which only the slow
E-A3 stage produces. On a fresh clone the suite therefore reports
**14 passed, 3 skipped** with an explicit skip reason, and reaches
17 passed once `python scripts/run_paper_A.py` (full, no `--skip-slow`)
has been run. Both outcomes are exit status 0.

`python scripts/run_paper_A.py --skip-slow` is the fast gate: it skips
the two slow stages and reports **31 passed, 0 failed, 2 skipped** in
about 28 s, exit status 0 (measured 2026-08-16).

`NEEDS_HUMAN`: pin the exact interpreter and library versions of the
original 2026-07 **production** runs (e.g. an archived `pip freeze`)
alongside the table above before CPC submission. The two are expected to
differ in the last digits of roundoff-floor quantities only — see the
note in `EXPECTED_OUTPUTS.md` on LAPACK/BLAS sensitivity.

## Expected outputs

See `EXPECTED_OUTPUTS.md` for the smoke-test checks a third party should
observe, and for the SHA-256 digest of the canonical regression case.

## Licence

MIT (`LICENSE`, vendored in `scripts/`), the licence of the reference
implementation.
