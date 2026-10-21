# Paper A — Exact-structure spectral discretization of the point-nucleus radial Dirac equation

Reproducibility repository for the manuscript *"Exact-structure spectral
discretization of the point-nucleus radial Dirac equation"* (C. D. Reinoso,
2026), prepared for Computer Physics Communications.

## Contents
- `main.tex`, `refs.bib` — manuscript source (LaTeX, `elsarticle` class from
  your TeX distribution).
- `figA_pipeline.pdf`, `figA_convergence.pdf`, `figA_screening.pdf` — figures.
- `data/` — the tabulated results behind the manuscript tables and figures.
- `reproducibility/` — a self-contained bundle (`scripts/`) that regenerates
  the paper's results, with its own `README.md`, `requirements.txt`, tests and
  a vendored copy of the spectral engine sources used for this paper.

## How to reproduce
1. Create a Python environment and install `reproducibility/scripts/requirements.txt`.
2. Run the drivers in `reproducibility/scripts/scripts/` (see
   `reproducibility/README.md` and `reproducibility/EXPECTED_OUTPUTS.md`).
3. Rebuild the PDF with `latexmk -pdf main.tex`.

## Spectral engine
The underlying spectral engine is maintained as a **separate repository** and
should be **cited by its archived DOI**. This Paper A bundle ships a vendored
snapshot of the engine sources inside `reproducibility/scripts/src/` solely so
that the paper reproduces offline; the canonical, citable engine lives in its
own repository.
