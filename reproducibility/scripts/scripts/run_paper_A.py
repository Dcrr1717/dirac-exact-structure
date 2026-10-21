"""Paper A reproduction driver: run the whole chain, then ASSERT.

    python scripts/run_paper_A.py

This is the single deterministic entry point for the manuscript

    "Exact-structure spectral discretization of the point-nucleus radial
     Dirac equation ..."  (Paper A)

It runs the four Paper A generators end to end, in dependency order, and
then checks the numbers they produce against the five expectations
recorded in the manuscript's reproducibility notes
(``PaperA_Dirac_PRA/reproducibility/EXPECTED_OUTPUTS.md``).  Nothing is
sampled and no seed is used: every stage is deterministic.

Stages
------
  C  certified assembly (in-process; C1-C4 identities + an injected sign
     error that must produce an order-unity violation)      Expectation 1
  1  exploration/dirac/ea1_yukawa_anchors.py                 Expectation 4
  2  exploration/dirac/ea2_point_table.py               Expectations 2 & 3
  3  exploration/dirac/ea3_critical_screening.py              Expectation 5
  4  exploration/dirac/ea4_convergence_and_figures.py    figures + Fig. 1
                                                              source CSV

Every check is a TOLERANCE check, never an exact float comparison.  The
tolerances are deliberately looser than the quoted values: the quoted
values are roundoff-floor quantities and their last digits depend on the
LAPACK/BLAS build (see the note printed at the end).  A check fails only
if the result has moved by orders of magnitude, i.e. if the physics or
the assembly broke.

Options
-------
  --skip-slow   omit stage 3 (ea3_critical_screening.py), which dominates
                the wall time; expectation 5 is then reported SKIP.
  --verbose     echo the full stdout of every generator.

Exit status is 0 only if every executed check passed.

NOTE: the generators overwrite their own CSV outputs in place
(``exploration/dirac/ea2_point_table.csv``,
``exploration/dirac/ea3_critical_screening.csv``,
``results/EA4_convergence.csv``).  The versions committed to this
repository are the production run quoted in the manuscript; a fresh run
reproduces them to roundoff (~1e-13 relative), not bit for bit.  Use
``git diff`` / ``git checkout`` if you want the frozen copies back.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirac import C_AU, RadialDiracBasis  # noqa: E402

DIRAC = ROOT / "exploration" / "dirac"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# expectations, quoted from PaperA_Dirac_PRA/reproducibility/EXPECTED_OUTPUTS.md
# and from the manuscript tables they summarize
# ---------------------------------------------------------------------------
E_92_KM1 = -4861.1979043696        # Ha, Table I, Z=92 kappa=-1 (N=45)
E_136_KM1 = -16474.112454094       # Ha, Table I, Z=136 kappa=-1
REL_CEILING = 1e-12                # paper: all 72 levels in [1e-16, 1.3e-13]
DEGEN_CEILING = 1e-11              # paper: spurious/1s degeneracy 4e-15..8e-13
F_KH_1 = -0.00408291802884811      # Krauthauser-Hill 40-digit, case 1
F_KH_2 = -0.25879891617626010      # Krauthauser-Hill 40-digit, case 2
S_C_1S_NONREL = 1.190612           # nonrelativistic Yukawa 1s constant
SPLIT_2P_Z130 = 0.40               # paper: ~40% at Zalpha=0.949 (Z=130)


class Report:
    """Collects PASS/FAIL/SKIP lines and prints the summary."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def record(self, ok: bool | None, tag: str, what: str, detail: str) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append((status, tag, what, detail))
        print(f"  [{status}] {tag}  {what}: {detail}")

    def check(self, tag: str, what: str, value: float, lo: float, hi: float,
              fmt: str = "{:.3e}") -> None:
        ok = bool(lo <= value <= hi)
        self.record(ok, tag, what,
                    f"{fmt.format(value)} in [{fmt.format(lo)}, {fmt.format(hi)}]")

    def skip(self, tag: str, what: str, why: str) -> None:
        self.record(None, tag, what, why)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r[0] == "FAIL")

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r[0] == "PASS")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.rows if r[0] == "SKIP")


def run_generator(script: Path, verbose: bool) -> str:
    """Run one generator in a fresh interpreter; return its stdout."""
    print(f"\n--- running {script.relative_to(ROOT)} ---", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT),
                          capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if verbose:
        print(proc.stdout)
    else:
        tail = [ln for ln in proc.stdout.strip().splitlines()[-3:]]
        for ln in tail:
            print(f"    | {ln}")
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"generator {script.name} FAILED (rc={proc.returncode})")
    print(f"--- {script.name} ok, {dt:.1f} s ---", flush=True)
    return proc.stdout


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# stage C -- Expectation 1: certified assembly
# ---------------------------------------------------------------------------
def stage_certification(rep: Report) -> None:
    print("\n=== stage C: run-time certification of the assembly ===")

    # (a) integer basis + Gauss-Legendre: all four identities are enforced.
    B = RadialDiracBasis(40, -1, 1.0, 1.0)
    c1, c2, c3, c4 = B.certs
    rep.check("E1a", "C1 overlap identity, Z=1 integer basis", c1, 0.0, 1e-10)
    rep.check("E1b", "C2 <f|O+O-f> = ||O-f||^2", c2, 0.0, 1e-7)
    rep.check("E1c", "C3 <f|O-O+f> = ||O+f||^2", c3, 0.0, 1e-7)
    rep.check("E1d", "C4 integration by parts", c4, 0.0, 1e-10)

    # (b) cusp-adapted gamma basis + exact Gauss-Laguerre rule: C1/C4 are
    #     the enforced pair (C2/C3 carry 1/r^2 and are monitored only).
    gam = float(np.sqrt(1.0 - (92.0 / C_AU) ** 2))
    Bc = RadialDiracBasis(45, -1, 92.0, 0.8 * 92.0, mu=gam, quad="cusp",
                          nquad=45 + 150, cert_tol=1e-6)
    rep.check("E1e", "C1 overlap, gamma basis Z=92 cusp rule", Bc.certs[0],
              0.0, 1e-10)
    rep.check("E1f", "C4 int. by parts, gamma basis Z=92 cusp rule",
              Bc.certs[3], 0.0, 1e-10)

    # (c) injected sign error: flip the sign of kappa/r inside O+ only.  The
    #     integration-by-parts identity C4 must break by order unity.  This
    #     is the scripted form of the "deliberately injected sign error"
    #     demonstration of crosscheck_assembly.py.
    #
    #     NOTE ON NORMALIZATION.  `certs[3]` is the ABSOLUTE max entry of
    #     <f_i|O+ f_j> - <O- f_i|f_j>, and the stored h, g are (1/2c)O_+-f,
    #     so the natural scale of those matrix elements is ~1/(2c) ~ 4e-3,
    #     not 1.  "Order-unity relative violation" (the manuscript's
    #     wording, Sec. II "Certified assembly") therefore means the
    #     violation divided by that scale.  Measured here: the clean build
    #     gives 6e-11 relative, the sign-flipped build 1.03 -- ten orders
    #     of magnitude apart, and the single tolerance cert_tol = 1e-7 that
    #     RadialDiracBasis uses to reject a build sits cleanly between them.
    clean = RadialDiracBasis(40, -1, 1.0, 1.0)
    scale = max(float(np.max(np.abs(clean._me(clean.f, clean.h)))),
                float(np.max(np.abs(clean._me(clean.g, clean.f)))))
    clean_c4 = float(np.max(np.abs(clean._me(clean.f, clean.h)
                                   - clean._me(clean.g, clean.f))))

    Bad = RadialDiracBasis(40, -1, 1.0, 1.0)
    Bad.h = (-Bad.df - (Bad.kappa / Bad.r) * Bad.f) / (2 * Bad.c)   # WRONG sign
    broken_c4 = float(np.max(np.abs(Bad._me(Bad.f, Bad.h) - Bad._me(Bad.g, Bad.f))))

    rep.check("E1g", "injected O+ sign error breaks C4 by O(1) (relative)",
              broken_c4 / scale, 1e-1, 1e1)
    rep.check("E1h", "clean C4 relative violation stays below cert_tol=1e-7",
              clean_c4 / scale, 0.0, 1e-7)
    rep.check("E1i", "broken / clean separation (decades)",
              float(np.log10(broken_c4 / clean_c4)), 6.0, np.inf,
              fmt="{:.1f}")


# ---------------------------------------------------------------------------
# stage 1 -- Expectation 4: screened-Coulomb external validation
# ---------------------------------------------------------------------------
def stage_ea1(rep: Report, verbose: bool) -> None:
    print("\n=== stage 1: E-A1 screened-Coulomb validation (ea1_yukawa_anchors) ===")
    run_generator(DIRAC / "ea1_yukawa_anchors.py", verbose)
    rows = {r["case"]: r for r in read_csv(DIRAC / "validation_errors.csv")}

    # The manuscript (Sec. IV) quotes the MASS-SHIFTED pencil values;
    # both pencils are exported, both are checked.
    for tag, case, f_kh, tol in (("E4a", "case1", F_KH_1, 1e-14),
                                 ("E4b", "case2", F_KH_2, 1e-13)):
        r = rows[case]
        for what, col in (("shifted pencil", "f_shifted"),
                          ("unshifted", "f_unshifted")):
            dev = abs(float(r[col]) - f_kh)
            rep.check(tag if what.startswith("shifted") else tag + "'",
                      f"{case} |f - KH(40 digit)| ({what})", dev, 0.0, tol)

    # exact-quadrature claim: at gamma = 0.714 the Gauss-Legendre rule
    # stalls near 1e-4 while the matched cusp rule drifts by ~1e-10.
    r2 = rows["case2"]
    rep.check("E4c", "case2 cusp-rule drift nb 60->80",
              float(r2["drift_cusp"]), 0.0, 1e-8)
    rep.check("E4d", "case2 Gauss-Legendre drift nb 60->80 (must stall)",
              float(r2["drift_legendre"]), 1e-6, np.inf)


# ---------------------------------------------------------------------------
# stage 2 -- Expectations 2 and 3: point-nucleus table
# ---------------------------------------------------------------------------
def stage_ea2(rep: Report, verbose: bool) -> None:
    print("\n=== stage 2: E-A2 point-nucleus table (ea2_point_table) ===")
    run_generator(DIRAC / "ea2_point_table.py", verbose)
    rows = read_csv(DIRAC / "ea2_point_table.csv")

    rep.check("E2a", "number of tabulated levels", float(len(rows)),
              72.0, 72.0, fmt="{:.0f}")

    def pick(Z: float, kappa: int, nr: int) -> dict[str, str]:
        for r in rows:
            if (float(r["Z"]) == Z and int(r["kappa"]) == kappa
                    and int(r["nr"]) == nr):
                return r
        raise SystemExit(f"missing row Z={Z} kappa={kappa} nr={nr}")

    r92 = pick(92.0, -1, 0)
    rep.check("E2b", "Z=92 kappa=-1 ground binding vs Table I (rel)",
              abs(float(r92["E"]) - E_92_KM1) / abs(E_92_KM1), 0.0, 1e-12)
    rep.check("E2c", "Z=92 kappa=-1 |rel. dev. from Sommerfeld(40 digit)|",
              abs(float(r92["rel"])), 0.0, REL_CEILING)

    r136 = pick(136.0, -1, 0)
    rep.check("E2d", "Z=136 kappa=-1 (Zalpha=0.9924) binding vs Table I (rel)",
              abs(float(r136["E"]) - E_136_KM1) / abs(E_136_KM1), 0.0, 1e-12)
    rep.check("E2e", "Z=136 kappa=-1 distinguished-extension gamma",
              float(r136["gamma"]), 0.1220, 0.1235, fmt="{:.4f}")
    rep.check("E2f", "Z=136 kappa=-1 Zalpha", float(r136["nu"]),
              0.9920, 0.9928, fmt="{:.4f}")

    worst = max(abs(float(r["rel"])) for r in rows)
    rep.check("E2g", "worst |rel. dev.| over all 72 levels", worst,
              0.0, REL_CEILING)

    # Expectation 3: the naive kappa>0 spurious root is degenerate with the
    # ground state of the -kappa partner channel (Tupitsyn-Shabaev theorem).
    ground = {(float(r["Z"]), int(r["kappa"])): float(r["E"])
              for r in rows if int(r["kappa"]) < 0 and int(r["nr"]) == 0}
    offs = []
    for r in rows:
        kap = int(r["kappa"])
        if kap > 0 and r["spur0"] not in ("", "nan") and int(r["nspur"]) > 0:
            partner = ground[(float(r["Z"]), -kap)]
            offs.append(abs((float(r["spur0"]) - partner) / partner))
    rep.check("E3a", "number of channels carrying a spurious root",
              float(len(set((float(r['Z']), int(r['kappa'])) for r in rows
                            if int(r['kappa']) > 0 and int(r['nspur']) > 0))),
              12.0, 12.0, fmt="{:.0f}")
    rep.check("E3b", "worst |spurious - partner 1s| / |1s| (degeneracy thm)",
              max(offs), 0.0, DEGEN_CEILING)


# ---------------------------------------------------------------------------
# stage 3 -- Expectation 5: critical-screening map
# ---------------------------------------------------------------------------
def stage_ea3(rep: Report, verbose: bool, skip: bool) -> None:
    print("\n=== stage 3: E-A3 critical screening (ea3_critical_screening) ===")
    if skip:
        rep.skip("E5a", "s_c(Z=1, 1s) vs nonrelativistic constant",
                 "--skip-slow requested")
        rep.skip("E5b", "2p_1/2 - 2p_3/2 critical-screening splitting at Z=130",
                 "--skip-slow requested")
        print("  (skipped)")
        return
    run_generator(DIRAC / "ea3_critical_screening.py", verbose)
    rows = read_csv(DIRAC / "ea3_critical_screening.csv")
    sc = {(float(r["Z"]), int(r["kappa"])): float(r["s_c"]) for r in rows}

    rep.check("E5a", "|s_c(Z=1, 1s) - 1.190612| (paper: 1 part in 1e4)",
              abs(sc[(1.0, -1)] - S_C_1S_NONREL), 0.0, 1e-3)
    split = sc[(130.0, 1)] / sc[(130.0, -2)] - 1.0
    rep.check("E5b", "2p_1/2 vs 2p_3/2 splitting at Zalpha=0.949 (paper ~0.40)",
              split, SPLIT_2P_Z130 - 0.06, SPLIT_2P_Z130 + 0.06, fmt="{:.4f}")
    rep.check("E5c", "s_c(1s) rises with Zalpha (relativistic enhancement)",
              sc[(130.0, -1)] / sc[(1.0, -1)] - 1.0, 0.15, 0.35, fmt="{:.4f}")


# ---------------------------------------------------------------------------
# stage 4 -- Fig. 1 source data and the two Paper A figures
# ---------------------------------------------------------------------------
def stage_ea4(rep: Report, verbose: bool) -> None:
    print("\n=== stage 4: E-A4 convergence + figures (ea4_convergence_and_figures) ===")
    run_generator(DIRAC / "ea4_convergence_and_figures.py", verbose)
    rows = read_csv(RESULTS / "EA4_convergence.csv")
    rep.check("F1a", "EA4_convergence.csv rows", float(len(rows)),
              24.0, 24.0, fmt="{:.0f}")

    def series(case_key: str) -> dict[int, float]:
        return {int(r["dof"]): float(r["rel_err"]) for r in rows
                if case_key in r["case"]}

    s92 = series("Z=92,  1s")
    rep.check("F1b", "Z=92 1s already at the floor at 2N=16", s92[16],
              0.0, 1e-12)
    s2p = series("2p")
    rep.check("F1c", "Z=92 2p_3/2 error at 2N=16 (pre-convergence)", s2p[16],
              1e-8, 1e-3)
    rep.check("F1d", "Z=92 2p_3/2 error at 2N=80 (converged)", s2p[80],
              0.0, 1e-12)
    s110 = series("Z=110")
    rep.check("F1e", "Z=110 1s worst error over the sweep",
              max(s110.values()), 0.0, 1e-12)

    for name in ("figA_convergence.pdf", "figA_screening.pdf"):
        p = FIGURES / name
        rep.check("F2", f"{name} regenerated (bytes)",
                  float(p.stat().st_size) if p.exists() else 0.0,
                  5000.0, np.inf, fmt="{:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-slow", action="store_true",
                    help="omit ea3_critical_screening.py (dominates wall time)")
    ap.add_argument("--verbose", action="store_true",
                    help="echo the full stdout of every generator")
    args = ap.parse_args()

    import scipy  # noqa: PLC0415  (banner only)
    print("=" * 74)
    print("Paper A reproduction driver")
    print(f"  python {sys.version.split()[0]}   numpy {np.__version__}"
          f"   scipy {scipy.__version__}")
    print(f"  repository root: {ROOT}")
    print(f"  c = {C_AU} (CODATA 2018); binding energies are E - c^2 in Ha")
    print("=" * 74)

    t0 = time.perf_counter()
    rep = Report()
    stage_certification(rep)
    stage_ea1(rep, args.verbose)
    stage_ea2(rep, args.verbose)
    stage_ea3(rep, args.verbose, args.skip_slow)
    stage_ea4(rep, args.verbose)
    dt = time.perf_counter() - t0

    print("\n" + "=" * 74)
    print(f"SUMMARY: {rep.passed} passed, {rep.failed} failed, "
          f"{rep.skipped} skipped   ({dt:.1f} s wall)")
    if rep.failed:
        print("FAILED CHECKS:")
        for status, tag, what, detail in rep.rows:
            if status == "FAIL":
                print(f"  {tag}  {what}: {detail}")
    print("=" * 74)
    print("Tolerance note: the quoted expectations are roundoff-floor "
          "quantities.\nTheir last digits depend on the LAPACK/BLAS build, so "
          "every check above is\na bounded-interval check, never an equality. "
          "A FAIL means the result moved\nby orders of magnitude.")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
