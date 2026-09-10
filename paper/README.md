# ReScience C submission — paper source

Structure only so far: section headings and comments saying what belongs in each,
with no prose. The manuscript is a separate deliverable.

## Build

```bash
cd paper && make        # -> article.pdf
```

Requires **xelatex** (TeX Live), **Python 3 + PyYAML**, and **make** — all present
on the machine that produced the results. `make` regenerates `metadata.tex` from
`metadata.yaml` via `yaml-to-latex.py`, then runs `latexmk -pdf -pdflatex=xelatex`.
`make clean` removes build artifacts.

**Figures must exist before a build that includes them.** They are not committed —
the per-seed CSVs under `results/` are the canonical artifacts. From the repository
root:

```bash
make figures            # rebuilds all 21 figures from committed CSVs, ~2 min, no GPU
```

Then reference them from `content.tex` as e.g. `../figures/d4/acc_vs_tol.png`.

## Provenance of these files

Everything except `metadata.yaml`, `content.tex`, `bibliography.bib` and this README
is the official ReScience C template, copied from
<https://github.com/rescience/template> at commit `0978d0e` (2023-02-15), licensed
GPL-3+ (see `COPYING`); the bundled fonts carry their own licences in their
subdirectories.

The template files are verbatim with **one exception**, marked inline: `article.tex`
gains `\usepackage{booktabs}`, because `rescience.cls` loads `tabularx` but not
`booktabs` and the result tables use `\toprule`/`\midrule`/`\bottomrule`.
`metadata.tex` and `article.pdf` are build outputs and are not committed.

`rescience.cls` loads its fonts by relative path (`./source-serif-pro/…`), so the
four font directories must stay next to `article.tex`. They are ~8 MB of the ~8.3 MB
in this directory. The alternative, if that bulk is unwanted in the repository, is
the ReScience Overleaf template — but then `metadata.tex` has to be maintained by
hand, which is exactly the metadata/PDF desync the template warns about.

## Writing rules for this paper

- **No number is typed by hand.** Every figure quoted comes from a report script run
  over a committed CSV; `make figures` prints them all with their pre-declared
  refutation checks.
- **Sources:** raw observations and pre-declared conditions in `../OVERNIGHT_LOG.md`;
  departures from the papers in `../DEVIATIONS.md`; copying provenance in
  `../PROVENANCE.md`; scope in `../REPLICATION_PLAN.md` and `../OUT_OF_SCOPE.md`.
- **Claim IDs** (D1–D4, D8, C1–C4) are internal to the repository. Either introduce
  them explicitly or drop them from the prose.
- **Report the partials and the negative result as such**: the MNIST NODE and
  CIFAR-10 ANODE accuracies undershoot, one pre-declared solver check was refuted,
  and Chen's backward/forward NFE claim did not reproduce.

## Open TODOs before submission

| Item | Where |
|---|---|
| **Author order** — not settled; currently the order the names were supplied in | `metadata.yaml` |
| Author contributions statement (the honest way to resolve ordering) | `content.tex` |
| Institutional emails, affiliation detail (department/school) | `metadata.yaml` |
| Abstract (write last) | `metadata.yaml` |
| Zenodo DOI for the tagged submission commit — deliberately deferred until the code is final | `metadata.yaml` |
| If the code moves to a different repository, `code: url:` must be updated to match wherever the DOI is minted from | `metadata.yaml` |
| Whether the original authors were contacted | `content.tex`, acknowledgements |
| ORCIDs — optional for the build (the template omits the icon when empty) and not mandated by the journal, but ORCID is the identifier the metadata and DOI records use. A Google Scholar profile is not a substitute; there is no field for it | `metadata.yaml` |
| Title prefix `[Re]` chosen (ordering and mechanism reproduce; two absolute numbers undershoot). Revisit only if an editor disagrees | `metadata.yaml` |
