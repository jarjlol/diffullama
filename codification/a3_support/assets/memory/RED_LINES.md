# R1–R8 operator fallback (21 November 2026)

Operator-authored support for T1; use Aalhad's reviewed version when it lands.
Task rules are adapted from BORROWED_IDEAS.md §4 and BORROWED_IDEAS_REVIEW.md §5,
not a new verification of the cited papers.

| Rule | Why / evidence | Check |
|---|---|---|
| R1: Keep the per-model accuracy and macro metric fixed. | A weak macro gain can conceal a strong-model regression (ITERATION_1.md §Results). | Protected task.py/eval.py hashes; print both models' correct/n. |
| R2: Edit solution.py only; preserve evaluator, baseline, config and data. | Comparisons require a frozen harness. | Native plugin protected_paths and manifest hashes. |
| R3: Return an integer index into this item's existing candidates; never generate an answer. | This is selection from N=10, not new inference. | eval.py automatically checks int, not bool, and bounds. |
| R4: Do not promote a trade-off between Dream and LLaDA. | Node4.1 improved one model and hurt the other on gate (ITERATION_1.md). | Reviewer requires >=3 net gate questions on EACH model vs frozen tuned majority plus paired statistics and ablation. |
| R5: Explore on dev; use gate for checks; final300 stays outside Arbor and is evaluated once after freezing. | Prevent test-driven search. | Export only dev200/gate200; operator final manifest and one-shot receipt. |
| R6: Keep candidates, N and question set fixed. | More generation changes the experiment and oracle ceiling. | Data hashes and native protected data/**. |
| R7: select(item) is pure and deterministic, uses only that item's own candidates, and keeps no cross-call state. No other-model answers, gold, filesystem or network. | Cross-model cache was rejected twice (ITERATION_1.md §Arbor). | eval.py statically limits imports/access but DOES NOT enforce purity; external order_check.py uses fresh forward/reverse/isolated/repeated calls. Human code review remains required. |
| R8: Report extra inference passes/question and selector CPU time; compare increased inference at matched compute. | Idea1.1 compares reranking with more denoising steps. | Operator reads generation/scoring timings, selector wall time, and all added passes. |

Absolute prohibitions, with no exceptions. Static checks are a limited filter, not a
Python sandbox. R1/R2/R3 and candidate/data identity have automatic checks; R4/R5/R7/R8
also require operator review. A passing external order check is not a proof of purity.
No agreement to modify eval.py has been recorded; the official evaluator stays unchanged.
