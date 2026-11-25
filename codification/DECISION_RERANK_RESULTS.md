# Decision-model reranking — results

Run date: 2026-11-26. Branch `iter2/decision-rerank` off `104dddf`.
Pre-registration: `codification/DECISION_RERANK_PREREG.md` (committed before any
scored run).

Decision model: `perplexity/pplx-decider-flash-v1.3`, served by OpenRouter's
`/api/alpha/decisions` endpoint (the response identifies the snapshot as
`pplx-decider-flash-v1.3-20261112`). One typed `choice` per question, no chain of
thought. 4B, Apache-2.0.

## GSM8K (500 test questions per model, N=10)

| selector | Dream | LLaDA | macro | headroom |
|---|---:|---:|---:|---:|
| first | 0.404 | 0.486 | 0.445 | 0.0% |
| majority | 0.724 | 0.732 | 0.728 | 64.3% |
| AR qwen7b-cut sum | 0.488 | 0.522 | 0.505 | 13.6% |
| AR qwen7b-cut mean | 0.276 | 0.482 | 0.379 | −15.0% |
| **decision.listwise** | **0.770** | **0.790** | **0.780** | **76.1%** |
| decision.pointwise | 0.742 | 0.808 | 0.775 | 75.0% |
| oracle | 0.878 | 0.892 | 0.885 | |

Paired exact McNemar, stacked over the two models (1000 pairs), with a paired
bootstrap 95% CI:

| comparison | Δ | 95% CI | p | Holm |
|---|---:|---|---:|---:|
| listwise vs majority | +0.052 | [+0.030, +0.074] | 4.9e-6 | 4.9e-6 |
| listwise vs AR sum | +0.275 | [+0.245, +0.306] | 2.7e-63 | 5.3e-63 |
| pointwise vs majority (exploratory) | +0.047 | [+0.025, +0.070] | 4.9e-5 | — |
| pointwise vs AR sum (exploratory) | +0.270 | [+0.239, +0.301] | 3.1e-60 | — |

Per model, vs majority: Dream +0.046 (p=0.0095), LLaDA +0.058 (p=1.1e-4).
Vs AR sum: Dream +0.282 (p=9.4e-31), LLaDA +0.268 (p=2.8e-34).

Coverage: 500/500 questions per model (cov=1.00). No truncation; the longest
listwise prompt is 2,973 tokens against the model's 30,720-token training length.

## HumanEval (144 test problems, Dream, N=10, 64 steps, 512 tokens)

| selector | pass@1 | headroom |
|---|---:|---:|
| first | 0.104 | 0.0% |
| AR qwen7b sum | 0.056 | −16.3% |
| AR qwen7b mean | 0.049 | −18.6% |
| example-test filter | 0.264 | 53.5% |
| **decision.listwise** | **0.403** | **100.0%** |
| decision.pointwise | 0.382 | 93.0% |
| oracle (pass@10) | 0.403 | |

Paired exact McNemar: listwise vs example-filter +0.139, 95% CI [+0.083, +0.201],
p=1.9e-6 (20 wins, 0 losses); vs first +0.299, CI [+0.229, +0.375], p=2.3e-13
(43 wins, 0 losses). Pointwise vs example-filter +0.118, CI [+0.063, +0.181],
p=2.2e-4; vs first +0.278, p=2.0e-11.

The decision model selected a passing candidate on **58/58** problems where one
existed, including 22 where the passer was not the first passing candidate and
25 where it was the only passer. Shuffling the candidate probabilities within
each item drops accuracy to 8.4%, so the signal is not positional. Caveat: the
HumanEval tests are short and the oracle is only 40%, so this is a clean but
narrow selection task; the result says the model can recognise correct code, not
that it can solve the problems.

LLaDA was dropped: at the registered 64 steps / 512 tokens it collapsed to
newline runs; 0/10 on `HumanEval/0` even with the prompt-completion fix. That is
a negative result about LLaDA's code generation at this budget, not a harness
failure.

## Controls

**Candidate-order shuffle (HumanEval, 50 items, seed 1).** The decision model
scored 30/50 with the original order and 30/50 with the candidates permuted —
the same accuracy. Position is not driving the result.

**Candidate-order shuffle (GSM8K dev, 200 items, seed 1).** 74.5% with the
original order and 74.5% with the candidates permuted.

**Closed-book memorisation probe (GSM8K, 40 test items).** The model was shown
the question and two numbers, the gold answer and a random wrong answer from
another question, order randomised. It picked the gold number **37/40 = 92.5%**
(chance 50%). The decision model is therefore good at the underlying math, not
only at comparing candidates. Part of the listwise gain is answer knowledge: the
model can identify the right final answer and then select the candidate that
carries it. This does not invalidate the reranking result — it still has to map
its answer onto one of ten candidates, and it does so better than majority — but
the claim should be "a decision model can pick the correct candidate", not "it
verifies the reasoning".

## Honest limitations

- One decision-model snapshot, one prompt wording, no prompt tuning.
- The decision model is used outside its curated domains; its calibration is not
  validated on these tasks.
- GSM8K listwise prompts have median 1,423 tokens; most are below the model's
  2,048-token training length, a minority above it.
- HumanEval uses 64 denoising steps, below Dream's published budget.

## One-paragraph summary

On 1,000 GSM8K test questions (Dream and LLaDA, N=10), a single typed listwise
pass from `pplx-decider-flash-v1.3` reaches **78.0%**, beating majority vote
(72.8%, +5.2 pts, p=4.9e-6) and the AR-likelihood verifier (50.5%, +27.5 pts);
the yes/no variant reaches 77.5%. On 144 HumanEval problems it reaches 40.3%
pass@1, equal to the pass@10 oracle, beating the execution-based example filter
(26.4%) and the first sample (10.4%). The candidate-order control is flat, but a
closed-book probe scores 92.5%, so the model knows much of the underlying math;
the defensible claim is that a decision model can pick the correct candidate, not
that it verifies reasoning. LLaDA code generation at 64 steps is unusable. MBPP was
queued, but Dream at 64 steps produces malformed code even with a 3-shot prompt
(0 passing candidates in the first 20), so it was stopped; the harness is
committed (`mbpp_*.py`) and no MBPP numbers are claimed.
