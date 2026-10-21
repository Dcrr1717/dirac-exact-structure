# Paper A → CPC (Elsevier) format migration — APPLIED 2026-08-15

Class **now**: Elsevier `elsarticle`, `\documentclass[preprint,12pt]{elsarticle}`.
Class **before**: `\documentclass[aps,pra,twocolumn,superscriptaddress,floatfix,longbibliography]{revtex4-2}`.

The migration was executed on the author's explicit written order (rule R3
satisfied). The pre-migration REVTeX 4-2 source is kept verbatim as
`../main_revtex_backup.tex` with a one-line header marking it as the
fallback; restoring it is a single file copy.

Nothing in the physics text changed: not a number, a claim, a table value
or a sentence. Verified by a word-level diff of `pdftotext` output before
and after — every difference is class formatting (section numbering
I–VIII → 1–8, `FIG. n.` → `Figure n:`, keyword `;` → `,`, `[5–7]` →
`[5, 6, 7]`, apsrev4-2 → elsarticle-num reference layout, line-break
hyphenation) plus the newly added Program Summary.

## Build state

| | before (revtex4-2) | after (elsarticle) |
|---|---|---|
| errors | 0 | **0** |
| undefined references | 0 | **0** |
| undefined citations | 0 | **0** |
| overfull boxes | 0 | **0** |
| underfull boxes | 34 | **1** |
| pages | 8 (two-column) | 19 (single-column, 12 pt, + Program Summary) |
| resolved references | 16 | 16 |

The single remaining underfull box (badness 1194) is one line of the
BibTeX-generated reference list (`main.bbl`, the Liu *et al.* entry), not
of the manuscript body. It is below the "obviously loose" threshold and is
not worth overriding the publisher's default justified bibliography.

## What was converted (record, item by item)

1. **Class** — `revtex4-2` (`aps,pra,twocolumn,superscriptaddress,floatfix,longbibliography`)
   → `elsarticle` (`preprint,12pt`). `preprint` is the correct CPC
   submission form and is single-column, which is why every full-width
   float and every column-width workaround below could be simplified.
   `\journal{Computer Physics Communications}` added; note the installed
   elsarticle deliberately suppresses the old "Preprint submitted to …"
   footer, so the line records the venue in the source without printing.
2. **Front matter** — everything from `\title` to the keywords is now
   wrapped in `\begin{frontmatter}…\end{frontmatter}`; `\maketitle` is
   gone (the environment does it). `\author{C.~D.~Reinoso}` →
   `\author[espoch]{C.~D.~Reinoso}`; `\email{}` → `\ead{}`;
   `\affiliation{…}` → `\affiliation[espoch]{organization={…},
   city={Riobamba}, country={Ecuador}}`. `\date{\today}` kept.
   `\begin{abstract}…\end{abstract}` moved inside the frontmatter,
   comments and all.
3. **Keywords** — the body's `\paragraph*{Keywords.}` list became
   `\begin{keyword} … \sep … \end{keyword}` in the frontmatter, semicolons
   replaced by `\sep`. Same seven keywords, same order.
4. **MSC** — the body's `\paragraph*{MSC 2020 classification.}` became
   `\MSC[2020] 65F15 (…) \sep 65L15 (…) \sep 81Q10 (…) \sep 81V45 (…)`
   inside the same `keyword` environment. `\MSC[2020]` is the elsarticle
   convention and is what compiles; it prints as "*2020 MSC:*". The four
   codes and their parenthetical descriptions are unchanged, and the A-S2
   rationale comment travelled with them.
5. **PROGRAM SUMMARY (CPC-specific, mandatory)** — **written and typeset**
   as `\section*{Program summary}` immediately after `\end{frontmatter}`,
   in the CPC field order: program title (`feast-spectral`), developer's
   repository link, licensing provisions (**MIT**), programming language
   (Python 3), external routines/libraries with the declared floors,
   nature of problem, solution method, and additional comments (the two
   reproduce commands, ≈ 10 min run time / 600 s wall of which
   `ea3_critical_screening.py` is 572 s, `--skip-slow` ≈ 30 s, fast gate
   ≈ 15 s, memory well under 1 GB with the largest dense pencil
   2N × 2N at N ≤ 80, roundoff-not-bitwise reproduction, and the
   restriction of the exact-expansion property). Every field is sourced
   from `README.md`, `scripts/README.md`, `scripts/LICENSE`,
   `scripts/requirements.txt`, `scripts/pyproject.toml` and the
   manuscript's own availability section. Nothing was invented. The **CPC
   Library link / Code Ocean capsule** is the one field that cannot be
   sourced — it is assigned by the Technical Editor — so it is a
   `%` comment carrying `NEEDS_HUMAN` and is **not typeset**.
6. **Tables** — `ruledtabular` does not exist outside REVTeX. All three
   tables now use `booktabs`: outer `\begin{ruledtabular}`/`\end{…}` →
   `\toprule` … `\bottomrule`, every `\colrule` → `\midrule`, `\centering`
   added. `\usepackage{array}` is now loaded (the revtex `\@@array`
   conflict that produced "! Extra \or." is gone with the class), which
   allowed `>{\raggedright\arraybackslash}p{…}` columns:
   - `tab:sota`: the `\small` and the two blank continuation rows that
     existed only to survive the 246 pt REVTeX column were removed; the
     "Potentials" column is now a ragged-right `p{0.30\columnwidth}` that
     wraps by itself. Removing them fixed a 1.62 pt overfull box.
   - `tab:domain`: the three `p{…}` columns kept their exact widths and
     gained `\raggedright`, which removed nine underfull boxes.
   - `tab:point`: rules only. **All printed values in all three tables are
     byte-identical to the REVTeX version.**
7. **Floats** — `figure*` → `figure` for the pipeline figure (elsarticle
   preprint is single-column, so the starred form has no meaning);
   `width=\textwidth` retained and still renders the eight stages at their
   native size. The other two figures and all three tables were already
   unstarred. Final numbering is unchanged from the REVTeX build:
   **Fig. 1 = pipeline, Fig. 2 = convergence, Fig. 3 = screening;
   Table 1 = comparison, Table 2 = point nucleus, Table 3 = domain.**
8. **Bibliography** — `\bibliography{refs}` alone (REVTeX supplied the
   style) → `\bibliographystyle{elsarticle-num}` + `\bibliography{refs}`.
   All 16 cited keys of the 19 in `refs.bib` resolve; 0 undefined
   citations. The stale REVTeX control file `mainNotes.bib` was deleted as
   a dead artifact.
9. **Sections** — `\section*{Data and code availability}`, `{Funding}`,
   `{Declaration of competing interest}` and `{Acknowledgments}` carry
   over unchanged; the Elsevier competing-interest wording was already in
   place from A-M4.
10. **Absent by inspection** — this manuscript never used `\pacs`,
    `\preprint`, `\altaffiliation`, `\onecolumngrid`, `\twocolumngrid`,
    `\squeezetable` or the REVTeX `\text…` helpers, so there was nothing
    to convert. The four custom macros `\Op`, `\Za`, `\half`, `\rd` are
    preserved verbatim; `\Op` (21 lines), `\Za` (22) and `\rd` (46) all
    still expand correctly, and `\half` remains defined-but-unused exactly
    as it was under REVTeX (it was never called there either).
11. **Verification** — `grep -n "ruledtabular\|colrule\|revtex\|\\\\maketitle\|\\\\affiliation{" main.tex`
    returns nothing but the migration comments; `pdftotext main.pdf - |
    grep -i "needs.human"` is empty.

## Still open (not a formatting matter)

- `NEEDS_HUMAN`: the **CPC Program Library deposit** (code + metadata) is
  the real gate for this journal, not the LaTeX class. The manuscript
  commits to it on acceptance; it has not been done.
- `NEEDS_HUMAN`: make `https://github.com/Dcrr1717/feast-spectral` public
  and mint a DOI for the tagged release, then insert both into the
  Program Summary and the availability statement. The Program Summary
  currently prints the repository URL, which is truthful only once the
  repository is public.
- `NEEDS_HUMAN`: the CPC Library catalogue identifier and Code Ocean
  capsule id, to be supplied by the Technical Editor.
- `NEEDS_HUMAN`: freeze the exact production pins (`python -V`,
  `pip freeze`) into `README.md`; the Program Summary quotes the declared
  floors, which is not the same thing.
