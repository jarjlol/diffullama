# Decision-model edition of the paper

A separate edition of the paper, built only from the PR #76 results (`iter2/decision-rerank` @ `2d2fead`:
`codification/DECISION_RERANK_RESULTS.md`, `codification/results/*`). Title: "Knowing the Answer or Checking It? A Small Decision Model as an Answer Selector for Diffusion
Language Models". It has IEEE conference styling (numbered [n] citations in order of first use,
Roman-numeral small-caps headings, "Abstract—", TABLE I captions, IEEE-format references) on the same
pandoc→Chromium build and page checks as `final_submission/paper/`. It does not change that paper or its sections.

    python3 final_submission/paper_decision/build.py   # -> build/paper.pdf

The last build gives 4 body pages and 1 reference page, with 10/10 references cited and resolved.
Bootstrap CIs (seed 0, 10,000 resamples) and exact McNemar p-values were recomputed from the per-item files.
