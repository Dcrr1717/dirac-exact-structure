# Frozen generators for Paper A

This folder is a **verbatim copy** of the files of the `feast-spectral`
repository that produce every number and every figure of the manuscript,
taken on **2026-08-15**. Nothing here was written, rewritten or stubbed
for the editorial pass: each file is byte-identical to its counterpart in
the repository, and the layout below reproduces the repository's own
layout so that the relative imports resolve unchanged.

The living version remains
`https://github.com/Dcrr1717/feast-spectral`; this copy exists so that
the manuscript is self-contained if the repository moves, and so that a
referee can read the solver without cloning anything. On divergence, the
repository is authoritative — check the date above.

## Layout

```
LICENSE                 MIT, the licence of the whole package
requirements.txt        dependency floors
pyproject.toml          project metadata + pytest configuration
conftest.py             puts src/ and exploration/dirac/ on sys.path

src/dirac.py            the engine: cusp-adapted gamma-Laguerre basis,
                        exact generalized Gauss-Laguerre assembly, the
                        C1-C4 run-time certifications, the mass-shifted
                        pencil H - c^2 S, and the exactly summed
                        Rayleigh refinement
src/basis.py            the orthonormal Laguerre basis it builds on

scripts/run_paper_A.py  the single reproduction driver (see below)
scripts/make_pipeline_figure_dirac.py
                        Fig. 1, the methodology pipeline

exploration/dirac/
  ea1_yukawa_anchors.py         Sec. IV screened-Coulomb validation and
                                adjudication -> validation_errors.csv
  ea2_point_table.py            Table I, 72 point-nucleus levels
                                -> ea2_point_table.csv
  ea3_critical_screening.py     Fig. 3, the map lambda_c(Z, kappa)
                                -> ea3_critical_screening.csv
  ea4_convergence_and_figures.py  Fig. 2 + its source CSV
  crosscheck_assembly.py        Sec. V: assembled vs directly applied
                                blocks, quadrature-drift discriminator,
                                S-pencil conditioning
  validate_dkb.py               Sec. V: the DKB form-domain violation
  test_gamma_basis.py           script-style numerical demonstrations;
  test_cusp_quadrature.py       they run their computation at import
  test_shifted_pencil.py        time and hold no `def test_*` functions,
                                so conftest.py excludes them from pytest
                                collection -- run them with
                                `python exploration/dirac/<name>.py`
  test_regression_hash.py       a real pytest module: the hashed
                                canonical-spectrum regression guard
tests/test_paper_A.py           a real pytest module: scripted asserts
                                for the five expectations of
                                ../EXPECTED_OUTPUTS.md
```

## Running this copy

The two commands are the ones in `../README.md`, issued from *this*
directory:

```
pip install -r requirements.txt
python scripts/run_paper_A.py
```

≈ 10 min; `--skip-slow` omits `ea3_critical_screening.py` (521 s of it)
and finishes in ≈ 1 min. Then the fast gate, ≈ 15 s:

```
pytest tests/test_paper_A.py exploration/dirac/test_regression_hash.py -q
```

Run them in that order. Generated artifacts are not vendored here (see
below), so on a pristine copy the handful of tests that read a generator
CSV report SKIP until `run_paper_A.py` has produced it; everything that
is recomputed in process — all of expectation 1, the Z=92 and Z=136
headline levels, the degeneracy check, both Krauthauser–Hill cases, and
the whole hashed regression case — runs immediately.

Two further consequences of this being a copy rather than a clone. The
`results/` and `figures/` directories are likewise absent: the
generators create them on first run, so the figure and CSV outputs land
here rather than overwriting the ones shipped in `../../data/`. And only
the Paper A subset is present — the repository also contains the
nonrelativistic FEAST/Schrödinger study and the exploratory directories
for the companion papers, none of which Paper A depends on.

## What is deliberately not here

Generated artifacts. The CSVs this code produces are shipped, already
regenerated and documented, in `../../data/` (see `../../data/README.md`
for the column legends); the figures are the PDFs next to `main.tex`.
Keeping one copy of each output, rather than two, is what stops the
deposit from contradicting itself.

`NEEDS_HUMAN`: for the CPC deposit, replace this snapshot with a tagged
release of the repository plus its DOI, and reference the tag here.
