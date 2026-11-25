# LaTeX edition of the Task 5 paper

`paper.tex` loads the seven editable sections in `sections/`. Add citations to `references.bib` and cite them by key with `\citep` or `\citet`. Equations and tables are native LaTeX.

From the repository root, build with [Tectonic](https://tectonic-typesetting.github.io/):

```sh
cd final_submission/paper/latex
mkdir -p build
tectonic --keep-logs --outdir build paper.tex
pdfinfo build/paper.pdf | grep '^Pages:'
```

The first run may download TeX packages and fonts into Tectonic's cache. A submission-format build should have four body pages and one reference page. The bibliography starts on a new page. The provenance appendix remains in `../appendix_provenance.md` for the repository deliverable.

This edition is based on the Markdown manuscript at `99ed88c` and retains the 23 references used in the results-focused manuscript. Assignment workflow details are covered by the separate report and provenance appendix. The two original bibliography files retain the citation audit notes. The prose pass used the [Research Humanizer skill](https://github.com/aryanbethman/research-humanizer/blob/b755377/SKILL.md). The Markdown and its existing build remain the source for the separate submission workflow.

## Report workflow

The Task 5 report still builds with `../../report/build_report.py`. A report port can reuse the same build command with its own TeX entry point. Preserve its evidence links and 2–4 page limit and compare the output with the existing report before switching its source.
