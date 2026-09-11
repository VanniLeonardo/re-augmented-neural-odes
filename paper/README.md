# The ReScience C article

## Build

```bash
cd paper && make        # produces article.pdf
```

Needs xelatex, Python 3 with PyYAML, and make. `make` regenerates `metadata.tex` from
`metadata.yaml` through `yaml-to-latex.py`, then runs `latexmk` with xelatex. `make clean`
removes the build files.

Figures must exist before a build that includes them. They are not committed, because the
per-seed CSV files under `results/` are the artifact. From the repository root:

```bash
make figures            # rebuilds all figures from the committed data, about 2 min, no GPU
```

The article references them as `../figures/...`. If they have not been built, each figure
becomes a box naming the missing file, so the paper still compiles.

## Provenance of these files

Everything except `metadata.yaml`, `content.tex`, `bibliography.bib` and
this file comes from the official ReScience C template at
<https://github.com/rescience/template>, commit `0978d0e` (2023-02-15). The template is
GPL-3 or later, see `COPYING`. The bundled fonts carry their own licences in their
directories.

Five changes were made to the template, each marked in place:

1. `article.tex` adds `\usepackage{booktabs}`. The class loads `tabularx` but not `booktabs`,
   and the result tables need it.
2. `header.tex` has its abstract block uncommented. Upstream ships it commented out, so the
   abstract in `metadata.yaml` is defined but never printed. Published ReScience C articles do
   print an abstract, so it is enabled here. The neighbouring "A replication of" block stays
   commented, which also matches those articles.
3. `Makefile` adds `header.tex` as a prerequisite of `article.pdf`. It was missing, so editing
   the header did not trigger a rebuild.
4. `header.tex` redefines `\authorsSHORT`, `\authorsABBRV` and `\authorsFULL`. With more than
   three authors the class abbreviates the list to "<first author> et al." in the page footer,
   the copyright line and the PDF metadata. The authors are in alphabetical order and none of
   them leads the work, so all six are named in each place.
5. `header.tex` prints one full stop at the end of the "Code is available at" line. Each
   branch of that line ended in its own, so a record with a DOI and no Software Heritage
   identifier printed two.

`rescience.cls` loads its fonts by relative path, so the four font directories must stay next
to `article.tex`. They account for about 8 MB of this directory. `metadata.tex` and
`article.pdf` are build outputs and are not committed.

## Writing rules

- No number is typed by hand. Every figure quoted in the article comes from a report script
  run over a committed result file. `make figures` prints them all, each with the condition
  recorded before the run.
- Sources: raw results in `../OVERNIGHT_LOG.md`, departures from the papers in
  `../DEVIATIONS.md`, whether author code was copied in `../PROVENANCE.md`, and scope in
  `../OUT_OF_SCOPE.md`.
- The article refers to claims 1 to 8. The `d1`, `c2` and similar identifiers name `make`
  targets and belong in the repository, not in the prose.
- Report the partial results and the negative result as such. Two accuracies undershoot, one
  condition recorded before a run was violated, and the backward against forward NFE claim
  does not reproduce.

## Open items before submission

| Item | Where |
|---|---|
| A GitHub release for submission, archived by Zenodo, and its concept DOI | `metadata.yaml` |
| If the code moves to another repository, update `code: url:` to match the repository the DOI is minted from | `metadata.yaml` |
| The outcome of writing to the original authors | `content.tex`; the correspondence itself is kept privately, not in the repository |

## Archived code

The code is archived on Zenodo from a GitHub release. The earlier v1.0.0 record was withdrawn.
The submission release mints a new concept DOI, which goes in `metadata.yaml` (`code: doi:`)
and always resolves to the newest release.

Anything committed after a release is absent from that release. Publishing another GitHub
release archives the new state and mints a new version DOI under the same concept DOI, so
the article does not need editing.
