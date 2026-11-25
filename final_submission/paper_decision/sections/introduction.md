# Introduction

<!-- Decision-model edition. Results: codification/DECISION_RERANK_RESULTS.md and results/*.jsonl, iter2/decision-rerank @ 2d2fead. -->

Diffusion language models (DLMs) generate text by unmasking a sequence in parallel instead of left to right. DiffuLLaMA (Gong et al., 2025) showed that autoregressive (AR) models can be converted into DLMs by continual pre-training, and exposed a gap between what such models can produce and what they output. On SATMath, at least one of three samples is correct 57.7% of the time, while self-consistency (majority vote; Wang et al., 2023) reaches 27.7%. The gap is a selection problem: the right answer is often among the candidates, and a selector has to find it.

On our candidates the gap is large. With ten samples per GSM8K question (Cobbe et al., 2021), the first sample of Dream 7B (Ye et al., 2025) is correct 40.4% of the time, majority vote 72.4%, and the best of ten 87.8%. Majority vote cannot reach the remaining 15 points, because on those questions the correct answer is in the minority. Scoring candidates by their likelihood under a frozen AR model is the obvious training-free alternative, and it fails here: Qwen2.5-7B, Dream's own AR base, picks a correct candidate 48.8% of the time on Dream and 52.2% on LLaDA, more than 20 points below majority vote.

We test a different kind of selector: a small *decision model* that reads the problem and all candidates and returns a probability over them in one forward pass, without chain of thought. The question is whether it beats majority vote, and on code, whether it beats running the docstring examples.

**Contributions.**

1. **A pre-registered comparison of selectors on fixed candidates.** On 500 GSM8K test questions per model, with ten cached candidates from Dream 7B and LLaDA 8B (Nie et al., 2025), we compare the first sample, majority vote, AR likelihood, a prompted AR model and a decision model. The hypotheses, tests and fallbacks were committed before the results commit.
2. **A decision model beats majority vote on GSM8K.** One listwise pass of `pplx-decider-flash-v1.3` reaches 77.0% on Dream and 79.0% on LLaDA (macro 78.0%). That is 5.2 points above majority vote (95% CI [3.0, 7.4], McNemar $p = 4.9 \times 10^{-6}$) and 27.5 points above AR likelihood, closing 76% of the gap from the first sample to the oracle. Asking the same AR model, Qwen2.5-7B, to choose a letter gives 56.4%, so the gain is not just from prompting.
3. **On HumanEval it finds every passing candidate.** On 144 test problems (Chen et al., 2021) with Dream candidates, it reaches 40.3% pass@1. That equals the pass@10 oracle: it chose a passing candidate on all 58 problems that had one. Execution of the docstring examples reaches 26.4% (+13.9 points, 20 wins and 0 losses).
4. **Controls that bound the claim.** Reordering the candidates leaves accuracy unchanged. However, a closed-book probe shows the decision model knows 37 of 40 GSM8K answers, and the registered HumanEval memorisation probe was not run. We therefore claim selection, not verification. We also report every deviation from the pre-registration.
