# Decision-model edition of the paper

A separate edition of the paper, built only from the PR #76 results (`iter2/decision-rerank` @ `2d2fead`:
`codification/DECISION_RERANK_RESULTS.md`, `codification/results/*`). It uses the same layout and build checks
as `final_submission/paper/`, and does not change that paper or its sections.

    python3 final_submission/paper_decision/build.py   # -> build/paper.pdf

The last build gives 4 body pages and 1 reference page, with 10/10 references cited and resolved.
Bootstrap CIs (seed 0, 10,000 resamples) and exact McNemar p-values were recomputed from the per-item files.
