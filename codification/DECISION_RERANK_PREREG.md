# Pre-registration — decision-model reranking (GSM8K, HumanEval)

Committed **before** any scored run, 2026-11-26. Base: `iter2/aaditya` @ `104dddf`.
Branch: `iter2/decision-rerank`. Post-submission experiment; `final_submission/` untouched.

## Question

The AR-likelihood verifier (Qwen2.5-7B NLL) loses to majority vote when reranking
N=10 diffusion-LM candidates. Does a System-1 decision model (`pplx-decider-flash-v1.3`,
4B, Apache-2.0, one pass, typed choice / yes-no, no chain of thought) do better?

## GSM8K

Data: cached candidates, 10 per question, 500 test questions for `dream-base` and
`llada-base`, exactly as generated for iteration 2. No regeneration. Gold from
`codification/rerank_task/data/test/gold.jsonl`. Dev (500) is used only to check
plumbing and the candidate-shuffle control, never to pick a prompt.

Selectors (all see the same candidates, no gold, no hidden test):

| id | rule |
|---|---|
| `first` | candidate 0 |
| `random(exp.)` | expected accuracy of a uniform pick (exact) |
| `majority` | most frequent extracted answer, ties -> earliest |
| `ar.qwen7b-cut.sum` | lowest summed NLL over cut text |
| `ar.qwen7b-cut.mean` | lowest per-token NLL over cut text |
| `decision.listwise` | one pass, 10 labelled candidates, argmax P(choice) |
| `decision.pointwise` | 10 independent yes/no passes, argmax P(yes) (exploratory) |
| `oracle` | any candidate correct (ceiling) |

Text: `cut_after_answer(task.trim(text))`, the same text the cut AR scores saw.
Prompt: fixed template (problem, candidates A–J, "pick the one whose final answer
is correct"), Nimble schema enum A–J. Pointwise: "does this candidate's final
answer agree with the correct answer?". No prompt tuning on test.

**H1 (primary):** `decision.listwise` macro-average accuracy (mean of the two model
accuracies) exceeds `majority`.
**H2 (secondary):** `decision.listwise` macro-average exceeds `ar.qwen7b-cut.sum`.
**H3 (exploratory):** `decision.pointwise` exceeds `majority`.

Inference: per-item correctness, paired exact McNemar (two-sided). H1 and H2 use
Holm correction at family-wise alpha = 0.05. Report delta with a paired bootstrap
95% CI (10,000 resamples, seed 0). Report a loss as plainly as a win.

Fallbacks, fixed now: if a listwise prompt exceeds the model's input limit (30,720 training tokens;
no item on these splits comes close), score that item pointwise and mark it; if
more than 1% of items need this, the listwise arm is reported as invalid rather
than silently mixed. No truncation.

Controls: candidate order shuffled once (seed 1) on 200 dev items — report the
change in listwise choice rate, not accuracy alone; closed-book on 40 test items
(ask the model to answer the problem, no candidates) to estimate benchmark
memorisation. These are diagnostics, not primary outcomes.

## HumanEval

164 problems (MIT), dev 20 / test 144, seed 0. Models: `dream-base`, `llada-base`.
N=10, 64 denoising steps, T=0.7, max_new_tokens 512, zero-shot template fixed in
`humaneval_gen.py`. HumanEval+ is out of scope tonight; noted as a limitation.

Execution: subprocess, isolated `-I`, 5 s CPU limit, 2 GB address space, temp cwd,
env scrubbed. Not a security boundary. Canonical solutions must pass 164/164
before any candidate is scored; otherwise the run aborts.

Selectors: `first`, `random(exp.)`, `ar.qwen7b.sum` / `.mean` (NLL over generated
code, 512-token cap), `example-filter` (first candidate passing the docstring
examples, else first), `decision.listwise` (which candidate passes the hidden
tests?), `decision.pointwise` (does this implement the docstring?), `oracle`.
Majority vote is undefined for code and is not reported.

**H4 (primary):** `decision.listwise` pass@1 macro-average exceeds `example-filter`.
**H5 (secondary):** `decision.listwise` exceeds `first` and `ar.qwen7b.mean`.

Same test and reporting as GSM8K. Closed-book on 20 dev problems.

## Reporting

Per-model and macro tables; accuracy / pass@1; paired McNemar vs first and vs the
benchmark's main baseline; headroom `(selector - first) / (oracle - first)`;
per-item predictions saved. All 500 GSM8K test items and all 144 HumanEval test
problems scored, or the exact n is stated and no comparison is made to full-set
numbers.

## Known limitations, stated up front

`pplx-decider-flash-v1.3` was trained on sequences up to 30,720 tokens; the
GSM8K listwise prompts here have median ~1,400 tokens and max ~2,800, so length
is not a constraint. The decision model is used zero-shot, outside its curated
domains. HumanEval generation uses 64 steps, below the models' published budgets.
