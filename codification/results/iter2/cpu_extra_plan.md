# Seven exploratory CPU analyses: plan, rules fixed before any result

Written Wed 25 Nov 2026, about 00:50 IST, before any of the analyses below was run. Requested by Main, approved by Aaditya.
Every analysis here is EXPLORATORY (post-hoc): the K = 256 and K = 64 results are already known, so none of this is
preregistered in the sense of `k256_preregistration.md`. All seven are reported, including nulls. No multiple-comparison
correction is applied; read each p-value as one of seven families of exploratory tests.

## Common rules (apply to every item)

- Execution: CPU only on `antpc@blackwell` (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=8`, `nice -n 10`), stdlib Python only,
  inputs read from `CODIFY_WORK_DIR=~/codify/diffullama/codification/pipeline/work` (read-only). No GPU job, nothing outside `~/codify`.
  Only scripts and text/JSON outputs are committed; candidates, features and weights are not.
- Selectors, grading and statistics are reused, not re-implemented: `baselines._majority` (None ignored, earliest-index tie-break,
  index 0 if every answer is None), `baselines._argbest`, `task.is_correct` (a None choice counts as wrong), and `step_arm.paired`
  (difference of per-question correctness, 10 000-draw paired bootstrap 95% CI with seed 0, exact two-sided McNemar on discordant pairs).
- Differences are always written "first named minus second named". A model is reported separately; no pooling across models.
- Questions: gold answers from `work/data/{dev,test}.jsonl`. "Test 500" means the 500 test questions with candidates at K = 16/32/64.
  "K = 256 subset" means the 185 (Dream) / 192 (LLaDA) test questions with a K = 256 row (the set analysed in `k256_results.md`).
- Dev-then-test: items 4 and 7 choose a rule on nothing; the rule is fixed below. They are run on dev (K = 64, 200 questions per model)
  first and printed; then ONCE on test (K = 64, 500 questions per model), and the test result is reported whatever the dev result was.
  Items 1, 2, 5 and 6 have no free parameters and are run once on test. A crash or bug fixed before any output is produced is not a
  second run; a bug found after numbers were seen is reported as such and both outputs are kept.
- All items are one run each. If a script must be rerun for a reason other than a crash, the first output stays in the repo
  and the reason is written next to it.

## Item 1. Equal compute (steps per question)

Question set: the K = 256 subset, per model. Each comparison below uses 256 denoising steps per question on each side.
- PRIMARY of this plan: first sample (candidate 0) at K = 256 versus `_majority` of candidates 0 to 3 at K = 64 (4 x 64 = 256 steps).
  Paired test, K = 256 first minus K = 64 majority-of-4.
- Also: `_majority` of candidates 0 to 7 at K = 32 (8 x 32 = 256) versus K = 256 first. Paired test, the same sign convention.
- Curve: majority at n candidates for n = 1 to 10, x = n x K denoising steps per question, for K = 16, 32, 64 on test 500, K = 256 on
  the subset, and K = 64 restricted to the subset. Two versions: (a) the first n candidates by index with `_majority` (earliest-index
  tie-break); (b) the mean over 200 random size-n subsets (`random.Random(0)`; for each n the 200 index subsets are drawn once and
  reused for every question, K and model; indices sorted so the earliest-index tie-break applies inside the subset). The n = 10 point
  is the same in both. Descriptive: accuracy and n questions, no test.
- Seconds per question: mean `gen_seconds` of Blackwell rows (all N = 10 candidates), over the same questions as each curve (as `k256_table` does).
  A per-n figure is only the proxy 0.1 x n x that mean, labelled as a proxy; linear scaling in n was not verified.
- Stated in the output: equal denoising steps is not equal wall-clock time (batching, the K = 256 rows came partly from Ada GPUs and
  from shared cards, per-sample cost differs from per-step cost).
- Output: table, JSON, and a PNG (accuracy against steps per question, one panel per model).

## Item 2. AR reranking versus the first sample, paired

Selector `ar.qwen7b-cut.sum` (`_argbest` of the lowest summed `ar.qwen7b-cut.nll_sum`) minus first sample, per model at K = 16, 32, 64
(test 500) and K = 256 (subset). This closes the line in `ITERATION_2.md` "What this says" that marks the AR-versus-first comparison
as unverified as a paired result.

## Item 3. Refresh the reports

Re-run `make_iter2_reports.py` on the current data (the full LLaDA K = 64 self-score on 500, the K = 256 rows). Update
`ITERATION_2.md`: the tier status table at the top, Tables 2 to 6, and the "Not verified / limits" bullet about commit `aa5d3d2`.
List every changed number as old value to new value, in `cpu_extra_results.md`. The stale text in the generated `step_arm.md`
("K = 256 covers only the first 100 test questions") is corrected in the generator. Nothing in this item is a new test.

## Item 4. Cut-text AR as a tie-breaker

Rule (iteration-1 Arbor node 2.1, on cut text, I-22). Count votes over the ten candidates' extracted answers, None ignored. Let T be
the set of answers with the maximum count.
- If |T| = 1, that answer wins (the same as `_majority`; the chosen candidate is the earliest one with that answer).
- If |T| >= 2 (a tie): the score of an answer is the lowest per-token mean NLL, `ar.qwen7b-cut.nll_sum / ar.qwen7b-cut.n_tokens`, over
  its candidates (candidates with a missing score or zero tokens are skipped). The tied answer with the lowest score wins; a tied answer with no
  scored candidate is not eligible; equal scores go to the earliest candidate index. If no tied answer has a score, `_majority` is used.
- If every answer is None, index 0 (as `_majority`).
Report the number of tie questions (|T| >= 2) and the number of questions where the choice differs from `_majority` (correctness
can differ only there), and the paired test of this rule minus `_majority` over all questions. Dev (K = 64, 200 per model) first, then test 500 (K = 64) once.

## Item 5. DLM self-score at K = 256

On the K = 256 subset (n = 185 / 192): selector `dlm.selfscore` (highest, `_argbest(lower=False)`; cut text) and `dlm.selftrim` (highest;
iteration-1 trimmed text), each against `_majority` and against the first sample, paired. Four paired tests per model.

## Item 6. Vote share as confidence (descriptive, no test)

Per model at K = 64 (test 500) and K = 256 (subset). The top-vote count is the largest number of candidates (of 10) that share one
extracted answer, None ignored (0 if every answer is None). "Majority correct" is `task.is_correct` of the `_majority` choice.
- Majority accuracy by top-vote count 0 to 10, with n per bin.
- Coverage-accuracy curve: for each threshold t = 1 to 10, the questions with top-vote count >= t, their share of all questions
  (coverage) and majority accuracy on them.
- AUROC of top-vote count (as the score) for "majority correct" (Mann-Whitney with mid-ranks for ties; reported with the numbers of correct and
  incorrect questions; undefined if either is empty).

## Item 7. Filter degenerate samples before voting

`codification/n4/` (neel's N4 work) contains no degeneracy definition (it measures answer diversity and failure classes), so the fixed rule is:
a candidate is dropped if its extracted answer is None, or if `"####"` does not occur in `task.trim(text)` (the same trimmed text the
extractor reads; the extractor otherwise falls back to "answer is N" or the last number). Vote with `_majority` among the kept candidates
(earliest-index tie-break within the kept ones, mapped back to the original index). If all ten are dropped, plain `_majority` over the
ten. Report the number of candidates dropped, the number of questions with all ten dropped, the number of questions whose choice differs
from plain majority, and the paired test rule minus plain `_majority` over all questions. Dev (K = 64, 200) first, then test 500 (K = 64) once.

## Outputs

`results/iter2/cpu/extra_<n>_*.txt` and `.json`, one plot (`extra_1_equal_compute.png`), and `results/iter2/cpu_extra_results.md` with one
table per item and the commit and command that produced each. Code: `codification/pipeline/cpu_extra.py`.
