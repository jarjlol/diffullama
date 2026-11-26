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
- GSM8K listwise prompts have mean 1,544 tokens (median 1,504, p95 2,281);
  the model card gives a 30,720-token training length, so length is not the
  constraint. An earlier draft said 2,048 here; that was the Nimble number and
  is wrong for this model.
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


## Post-review corrections (2026-11-26, after #76 reviews)

**Statistics.** The stacked McNemar treats the same 500 questions twice and is
anti-conservative. A cluster bootstrap that resamples questions and carries both
models gives Δ = +5.2 points, 95% CI [+3.0, +7.4], p < 1e-4 (0 of 10,000
resamples at or below zero). The valid per-model exact McNemar tests are Dream
p = 0.0095 and LLaDA p = 0.00011. Report the cluster CI/p and the per-model
tests; the old macro p = 4.9e-6 should not be quoted.

**Judge-alone.** Removing the candidate reasoning and showing only the question
plus the distinct candidate answers gives 428/1000 = 42.8% (957 prompted items
plus 43 items with a single distinct answer, 42 of which are correct). The
listwise result is 78.0%, so the reasoning text is doing real work.

**Gold injection.** On the 115 items where no candidate has the gold answer, the
gold was added as an extra option. It was chosen 24/115 = 20.9% (chance ≈ 9–10%
for 11 options). Some answer knowledge, not enough to explain the 78%.

**HumanEval baseline, corrected.** The docstring parser now also reads
`f(x) == y` / `Example:` lines. The example filter rises from 26.4% to 27.8%.
Of the 58 solvable problems, 38 have parseable examples: on those, example
filter 36/38 and listwise 38/38; on the 20 without examples, 4/20 and 20/20.
The aggregate gap is therefore concentrated where the filter has no examples.
The execution-cluster baseline (run candidates on the docstring inputs, take the
largest output cluster) scores 6.2% — a weak baseline here, not a substitute.

**Unregistered arm.** `qwen.choice` (Qwen2.5-7B base, prompted for a letter)
scores 56.4% macro: above AR (50.5%) but below majority (72.8%). Exploratory; not
a pre-registered selector.

**Shuffle.** 67% of GSM8K dev items choose the same original candidate under the
seeded permutation; accuracy is 74.5% in both orders; the chosen position
distribution is spread, not first-position.

**Generation and serving rates.** Dream HumanEval: mean 45.6 s/problem
(median 50.2, p90 60.5) for N=10, 64 steps, 512 tokens. GSM8K cached: Dream
36.8 s, LLaDA 55.0 s (N=10, 256 tokens). MBPP: 46.4 s. Reranking adds no GPU
generation time. OpenRouter decisions: 1,000 listwise GSM8K requests in ~50 s
at 12 workers (~20/s); 10,000 pointwise in 213 s at 16 workers (~47/s); 1,470
HumanEval pointwise in ~45 s. Listwise prompts average 1,544 tokens, pointwise
233. The `:free` tier cost $0; at the paid flash price ($0.01/1M input) the
1,000 listwise calls are about $0.015 and the 10,000 pointwise calls about
$0.023.

**Declared deviations from the pre-registration.**
- Backend: OpenRouter `/api/alpha/decisions` (snapshot `pplx-decider-flash-v1.3-20261112`),
  not local weights; the 9.6 GB download could not finish. Same model family.
- H4 was registered as a Dream+LLaDA macro average; only Dream was run because
  LLaDA code generation collapsed at 64 steps. H4 is therefore a single-model test.
- The pointwise HumanEval wording includes “and will it pass the hidden tests?”,
  not only “does this implement the docstring?”.
- The probability-shuffle sentence is withdrawn; the candidate-order shuffle is
  the position control.
- TLS verification is disabled in the OpenRouter client because the campus
  firewall MITMs HTTPS. This is a lab workaround, not production code; a shipped
  client must use the system CA/certifi.
- The model snapshot and endpoint are recorded here and in the results JSON, not
  per request; raw API responses were not retained.

**Not measured.** A fully open-ended decider-only baseline is impossible with a
typed-choice model (it cannot write an answer). The HumanEval closed-book probe
on the 20 dev problems was not run; the recognition/closed-book question remains
open. GSM-Symbolic is the chosen next dataset (issue #79).

## K=256 step sensitivity (issue #80, exploratory, 2026-11-26)

**Status: exploratory, not a pre-registered primary test.** The registered tests
are still H1–H5 in `DECISION_RERANK_PREREG.md`; nothing here changes them.

Data: the cached 256-step candidates, 185 Dream + 192 LLaDA GSM8K test questions
(377 items, all inside the 500 used at K=64; spread evenly over uids 0–499), 10
candidates each. Same `candidate_text` cut as the AR features, same listwise
wording as the K=64 run, same API snapshot
(`pplx-decider-flash-v1.3-20261112`). 377/377 scored, coverage 1.00, 0 errors.
One listwise pass per question; the pointwise arm was not rerun.

| selector | Dream | LLaDA | macro | headroom |
|---|---:|---:|---:|---:|
| first | 0.741 | 0.724 | 0.732 | 0.0% |
| majority | 0.843 | 0.792 | 0.817 | 46.4% |
| AR qwen7b-cut sum | 0.757 | 0.766 | 0.761 | 15.8% |
| AR qwen7b-cut mean | 0.735 | 0.750 | 0.743 | 5.6% |
| **decision.listwise** | **0.876** | **0.839** | **0.857** | **68.3%** |
| oracle | 0.935 | 0.896 | 0.915 | |

Cluster bootstrap by question (203 unique questions, resampling questions and
carrying both models): listwise − majority **+0.040, 95% CI [+0.009, +0.071]**,
p = 0.010. Per-model exact McNemar: Dream +0.032 (11 wins / 5 losses, p = 0.21),
LLaDA +0.047 (16 / 7, p = 0.093) — neither individually significant at
n = 185/192. Listwise − AR sum is +0.096, CI [+0.059, +0.134], p < 1e-4
(Dream +0.119, LLaDA +0.073); AR sum still loses to majority (−0.056).

**Does the K=64 advantage shrink?** The headline numbers say +5.2 points at K=64
and +4.0 at K=256 (Dream +4.6 → +3.2, LLaDA +5.8 → +4.7). But the two runs use
different questions, so the K=64 predictions were restricted to the same 377
items: there the K=64 margin is **+3.5** (Dream +3.8, LLaDA +3.1), not +5.2. The
paired difference-in-differences is **+0.5 points, 95% CI [−4.0, +5.1]**,
p = 0.82 (Dream −0.5 [−7.6, +6.5], LLaDA +1.6 [−4.2, +7.3]). On identical
questions the decision-model advantage does **not** shrink measurably at 256
steps; the apparent 1.2-point drop is question-set composition. What does shrink
is absolute headroom: majority rises 72.8% → 81.7% and oracle 88.5% → 91.5%, so
the selector captures 68% of headroom instead of 76%. The per-model K=256 tests
are null mainly because n is 185/192 instead of 500, not because the matched
effect is smaller.

**N sweep (issue #83 cross-post).** 100 K=256 questions per model, first N
candidates in file order, same wording. 1,000 requests, 0 errors; the N = 10
rows reproduce the main run exactly on those 100 items (Dream 0.86, LLaDA 0.79).

| N options | Dream | LLaDA | macro | majority macro | Δ | 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.790 | 0.760 | 0.775 | 0.710 | +0.065 | [+0.026, +0.105] |
| 4 | 0.810 | 0.750 | 0.780 | 0.760 | +0.020 | [−0.020, +0.060] |
| 6 | 0.840 | 0.790 | 0.815 | 0.775 | +0.040 | [+0.000, +0.081] |
| 8 | 0.830 | 0.790 | 0.810 | 0.785 | +0.025 | [−0.016, +0.066] |
| 10 | 0.860 | 0.790 | 0.825 | 0.790 | +0.035 | [−0.010, +0.080] |

The gain is not a list-length artifact (it exists at N = 2 and does not grow
monotonically with N), but at 100 questions/model only the N = 2 interval
excludes zero; the sweep is descriptive.

**Cost and latency.** 377 listwise requests: 632,183 input tokens (mean 1,677;
Dream 1,444, LLaDA 1,901), median latency 0.42 s (p95 0.62 s), 14 workers,
~12 s wall. Free tier $0; at the paid flash price ($0.01/1M input) $0.0063 for
the run. The K=64 run was 1,000 requests in ~50 s at 12 workers, mean 1,544
tokens. The N sweep was 1.00M tokens for 1,000 requests (~$0.011).

Caveats: 377/500 questions, one snapshot, one wording, selection not
verification. The 256-step models often keep writing on the same line after
`#### N`, so 94% (Dream) / 76% (LLaDA) of the *cut* texts still carry junk after
the answer number; the AR features were cut identically, so the comparison is
fair, but it is noisier input than the K=64 run.
