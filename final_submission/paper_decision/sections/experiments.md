# Experimentation / Results and Discussion

<!-- Sources: codification/results/gsm8k_preds.jsonl, humaneval_preds.jsonl, gsm8k_decision.json, humaneval_decision.json, DECISION_RERANK_RESULTS.md (iter2/decision-rerank @ 2d2fead). CIs and McNemar p recomputed by neel's agent from the per-item files (bootstrap seed 0, 10,000 resamples). -->

**Setup.** GSM8K: 500 held-out test questions per model, $N = 10$ cached candidates from Dream 7B and LLaDA 8B, scored with full coverage (500/500 per model). HumanEval: 144 test problems, $N = 10$ Dream candidates. All selectors are scored on identical candidates.

**Table 1.** GSM8K test accuracy (%), $n = 500$ per model. Headroom is macro: (selector − first)/(oracle − first). The prompted Qwen2.5-7B arm is exploratory (not pre-registered).

| Selector | Dream | LLaDA | Macro | Headroom |
|---|---|---|---|---|
| First sample | 40.4 | 48.6 | 44.5 | 0% |
| Uniform pick (expected) | 41.2 | 48.8 | 45.0 | 1% |
| AR Qwen2.5-7B, per-token NLL | 27.6 | 48.2 | 37.9 | −15% |
| AR Qwen2.5-7B, summed NLL | 48.8 | 52.2 | 50.5 | 14% |
| Qwen2.5-7B prompted choice | 52.4 | 60.4 | 56.4 | 27% |
| Majority vote | 72.4 | 73.2 | 72.8 | 64% |
| Decision model, pointwise | 74.2 | **80.8** | 77.5 | 75% |
| Decision model, listwise | **77.0** | 79.0 | **78.0** | 76% |
| Oracle (hit@10) | 87.8 | 89.2 | 88.5 | 100% |

**Majority vote is the bar to beat.** Majority vote adds 32.0 points to the first sample on Dream and 24.6 on LLaDA (both $p < 10^{-29}$). AR likelihood stays far below it. The summed score beats the first sample by 6.0 points macro. The per-token score is worse than the first sample on Dream.

**H1 and H2: the decision model beats both.** The listwise selector is 5.2 points above majority vote macro (95% CI [3.0, 7.4]). It is correct on 90 questions where majority vote is wrong, and wrong on 38 where majority vote is right ($p = 4.9 \times 10^{-6}$, Holm $4.9 \times 10^{-6}$). The gain holds per model: +4.6 on Dream ([1.2, 8.0], $p = 0.0095$) and +5.8 on LLaDA ([3.0, 8.6], $p = 1.1 \times 10^{-4}$). Against summed AR likelihood the gap is +27.5 points ([24.5, 30.6], Holm $p = 5.3 \times 10^{-63}$). Both registered hypotheses hold.

**The pointwise variant is less consistent.** Yes/no scoring per candidate (H3, exploratory) is 4.7 points above majority vote macro ([2.5, 7.0], $p = 4.9 \times 10^{-5}$). The gain comes almost entirely from LLaDA: +7.6 there ([4.8, 10.6]), but +1.8 on Dream ([−1.6, +5.0], $p = 0.35$).

**It is the model, not the prompt.** Given the same listwise context, Qwen2.5-7B picks a correct candidate 56.4% of the time. That is 5.9 points above its own likelihood score ([2.4, 9.3], $p = 0.001$), but 16.4 points below majority vote and 21.6 below the decision model. Showing an AR model all candidates at once is therefore not enough; the gain needs a model that is good at this kind of choice.

**Table 2.** HumanEval pass@1 (%), Dream, 144 test problems, $N = 10$.

| Selector | pass@1 | Headroom |
|---|---|---|
| AR Qwen2.5-7B, per-token NLL | 4.9 | −19% |
| AR Qwen2.5-7B, summed NLL | 5.6 | −16% |
| Uniform pick (expected) | 8.5 | −6% |
| First sample | 10.4 | 0% |
| Example filter (docstring examples) | 26.4 | 54% |
| Decision model, pointwise | 38.2 | 93% |
| Decision model, listwise | **40.3** | **100%** |
| Oracle (pass@10) | 40.3 | 100% |

**H4: the decision model beats running the examples.** The example filter improves on the first sample by 16.0 points: 23 problems gained, none lost. The listwise selector adds a further 13.9 points ([8.3, 20.1]; 20 problems gained, 0 lost; $p = 1.9 \times 10^{-6}$). Of the 58 problems that had at least one passing candidate, it chose a passing one on all 58. This includes 25 problems with only one passer. The example filter chose a passing one on 38 of the 58. Pointwise scoring reaches 38.2% (+11.8 over the filter, $p = 2.2 \times 10^{-4}$). AR likelihood is below a uniform pick.

**Controls.** Permuting the candidates leaves listwise accuracy unchanged: 74.5% in both orders on 200 GSM8K dev questions, and 30/50 in both orders on the first 50 HumanEval test problems. Position does not drive the choice. We report accuracy only, not how often individual choices changed, which the pre-registration asked for.

The closed-book probe shows the decision model picks the gold number over a wrong one on 37 of 40 GSM8K test questions (92.5%, chance 50%). Much of the listwise gain may therefore come from knowing the answer and choosing the candidate that states it, rather than from checking each solution. It still has to map that answer onto one of ten candidates, and it does so better than majority vote.

**Limitations.**

- **HumanEval memorisation is untested.** The registered closed-book probe on 20 HumanEval dev problems is not reported. HumanEval is public, and choosing a passing candidate on 58 of 58 problems is what a model that knows the reference solutions would do. The HumanEval result therefore supports "picks the correct candidate", not "understands code".

- **H4 was tested on one model.** It was registered as a macro over Dream and LLaDA. LLaDA was dropped: at 64 steps and 512 tokens it produced runs of newlines, with 0 of 10 candidates passing on the first problem, so H4 as registered was tested on Dream alone.

- **Changes after registration.** The HumanEval pointwise prompt adds "and will it pass the hidden tests" to the registered wording. The analysis code changed between the pre-registration and results commits: it now filters non-finite scores, checks feature counts, and adds the exploratory prompted-Qwen arm.

- **MBPP was not completed.** Dream at 64 steps produced no passing MBPP candidate in the first 20 problems, even with a 3-shot prompt, so no MBPP numbers are reported.

- **Narrow evaluation.** We test one snapshot of one decision model, served through a third-party API, with one prompt wording and no tuning, at a step budget below the published one for both models.
