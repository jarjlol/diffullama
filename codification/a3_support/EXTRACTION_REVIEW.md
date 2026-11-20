# Operator extraction review, 21 November 2026

Codex read all 20 candidates in the private `support-N4/EXTRACTION_AUDIT_PRIVATE.json`:
ten per model, the earliest majority-selected candidate from ten dev questions per model
where majority fails but the oracle succeeds. This is an agent review, not a teammate's
manual annotation or a random sample. No final questions were opened. Raw candidate text
and gold answers stay outside GitHub and outside the injected memory.

| Model / dev UID | Candidate | Verdict |
|---|---:|---|
| Dream / 0133 | 1 | Ambiguous malformed repetition; last-number fallback gives 1, while opening prose says 111. No reliable intended final answer. |
| Dream / 0334 | 2 | 22 matches the explicit final marker; reasoning arithmetic is wrong. |
| Dream / 0456 | 0 | 440 matches the explicit marker. |
| Dream / 1293 | 1 | 120 matches the explicit marker; reasoning arithmetic is wrong. |
| Dream / 1765 | 1 | 32 matches the explicit marker; reasoning omits a condition. |
| Dream / 1875 | 1 | **Mismatch:** fallback extracts trailing price-list 0, while the problem-related reasoning ends at 5 miles. The completion never clearly answers the requested time. |
| Dream / 2010 | 2 | 5 matches the explicit marker; reasoning arithmetic is wrong. |
| Dream / 2711 | 0 | **Normalization corruption:** malformed enormous final numeral changes digits after conversion to float. Intended answer remains unclear. |
| Dream / 2759 | 0 | 200 matches the explicit marker; reasoning is inconsistent. |
| Dream / 2993 | 0 | **Ambiguous malformed marker plus normalization corruption:** prose ends at 2860, marker concatenates many zeros, and float normalization changes its digits. Do not assume truncating it recovers a valid answer. |
| LLaDA / 0039 | 1 | 5 matches the explicit marker; subtraction is wrong. |
| LLaDA / 0334 | 2 | 24 matches the explicit marker; reasoning misallocates the cream cost. |
| LLaDA / 0558 | 0 | 1000 matches the explicit marker; reasoning fails the task. |
| LLaDA / 1194 | 1 | 5 matches the explicit marker; reasoning ignores tying three pieces together. |
| LLaDA / 1204 | 0 | 180 matches the explicit marker. |
| LLaDA / 2215 | 1 | 16 matches the explicit marker; reasoning misreads the population affected. |
| LLaDA / 2355 | 0 | 34 matches the explicit marker; address-label accounting is wrong. |
| LLaDA / 2728 | 1 | 36000 matches the explicit marker; prices/counts are wrong. |
| LLaDA / 2858 | 2 | 40 matches the explicit marker; reasoning fails the task. |
| LLaDA / 3018 | 0 | 108 matches the explicit marker; second-week tablet count is wrong. |

16/20 match an unambiguous stated marker; one trailing-text mismatch, one ambiguous
unmarked repetition, and two malformed-long-number cases with float digit changes.
These four flags are disjoint in this count; the last case has multiple symptoms.
The audit identifies limitations; it does not establish an accuracy gain from any repair.

`task.normalize` uses float, `extract_answer` prioritizes the first marker, and the last
numeric token is the final fallback (`rerank_task/task.py`). Do not alter that protected
grader during search. Scoring with `--cut` removes trailing junk from AR scoring; it
does not retroactively repair extracted answers. Any grader repair needs a separately
agreed experiment, fresh receipts and consistent comparisons across every selector.
