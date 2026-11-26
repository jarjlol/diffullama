# Unified LaTeX Paper (Task 5 — Issue #77)

IEEEtran journal source in the IEEE Transactions on Parallel and Distributed Systems (TPDS) style, for the 4+1 page Task 5 research paper, combining the five-model diffusion selection study, the step-budget controls, and the decision-model reranking results (GSM8K $K=64/256$, weak-vote hybrid rule, HumanEval, HumanEval+, and controls).

## Build (deterministic 4 body + 1 reference page = 5 pages, IEEEtran journal)

With `pdflatex` and `bibtex`:

```sh
cd final_submission/paper/latex
mkdir -p build
pdflatex -interaction=nonstopmode -output-directory=build paper.tex
bibtex build/paper
pdflatex -interaction=nonstopmode -output-directory=build paper.tex
pdflatex -interaction=nonstopmode -output-directory=build paper.tex
pdfinfo build/paper.pdf | grep Pages
```

Or with `tectonic`:

```sh
cd final_submission/paper/latex
tectonic --keep-logs --outdir build paper.tex
```

The Markdown build in `final_submission/paper/build.py` and `sections/*.md` is left untouched as the submission fallback. A built copy of this LaTeX paper is committed at `final_submission/paper/paper.pdf` (5 pages: 4 body + 1 reference) so the upload does not depend on a local TeX install.
