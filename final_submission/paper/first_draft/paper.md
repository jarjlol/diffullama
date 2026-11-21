# Selecting Diffusion-Language-Model Answers: Representation Preservation, Majority Vote, and Evaluation Confounds

BITS F471 Group 3 | First assembled draft | 21 November 2026

## Abstract

Can an autoregressive (AR) scorer recover useful answers from a diffusion language model, and does preservation of its AR base's representations predict that recovery? We study five released diffusion models on GSM8K using ten cached candidates per question, AR likelihood, majority vote, and an agent-designed selector. On 300 held-out questions, majority vote reaches 73.3% for Dream and 73.0% for LLaDA; arithmetic-verification-weighted voting reaches 73.0% on both and shows no detectable improvement over majority. Across four adapted models, preservation and AR-recoverable headroom have positive rank correlations, but the sample is too small and dominated by floor models to support the proposed relationship. An audit finds that original AR scores included unrelated text after final answers, preventing a general conclusion about AR selection. Completed fidelity checks reach 80-81% for Dream and 73% for LLaDA on 100 questions each. We report an inconclusive preservation hypothesis, a strong majority baseline, and the unresolved evaluation controls needed for a fair test.

## 1. Introduction

Masked diffusion language models generate by repeatedly refining a sequence. Some are adapted from autoregressive checkpoints, including DiffuGPT and DiffuLLaMA [1], while LLaDA provides a model trained from scratch [2]. Dream provides another adapted model family [3]. This creates a question beyond whether sampling more answers helps: does the information retained from an AR base make AR scoring especially effective at selecting a diffusion model's answers?

A selector can improve accuracy only when the correct answer is already among the samples. We therefore separate generation quality, oracle coverage, and selection. The research hypothesis is that representation preservation predicts how much of the first-sample-to-oracle gap an AR scorer recovers. A positive relationship would motivate allocating test-time compute to selection for some adapted models. It would not, by itself, show that selection beats additional denoising.

We contribute a fixed-candidate evaluation across five models, a comparison against majority vote, and a protected selector-search task. We also document controls that change the interpretation: three models operate near the accuracy floor; the agent's merge comparator was weaker than majority; and trailing generated text confounds the original likelihood features. The completed experiments do not establish a selector better than majority vote or a predictive relationship between preservation and selection gain.

## 2. Related Work

DiffuLLaMA studies adaptation of AR models to diffusion generation [1]. LLaDA [2] and Dream [3] provide contrasting model lineages for our evaluation. We use released models rather than proposing a new training objective.

Selection over sampled reasoning paths has strong existing baselines. Self-consistency aggregates final answers [4], while trained verification ranks candidate solutions [5]. Our AR likelihood scorers are frozen and require no verifier training; we do not claim to reproduce a trained verifier or a published diffusion-specific selection method. Linear centered kernel alignment (CKA) measures representation similarity [6]. Our proposed use of CKA is predictive and empirical, rather than a claim that similarity entails answer correctness. Arbor organizes research around a hypothesis tree [7]; here its role is to implement and test selector rules within a fixed task.

<!-- pagebreak -->

## 3. Problem Formulation

For model m and question q_j with numeric answer a*_j, cache N = 10 candidate solutions y_j,1,...,y_j,N. An extractor phi maps a candidate to a number or a missing value. We make no assumption that the ten samples are statistically independent. A selector returns an index into that item's own candidates and may use their text, extracted answers, and cached features. It may not access gold, other models' answers, or state from earlier questions.

```text
c_j,i = 1 if phi(y_j,i) matches a*_j within 10^-6, else 0
Acc_m(s) = (1/n) sum_j c_j,s(q_j,y_j,1:N)
Oracle_m = (1/n) sum_j max_i c_j,i
H_m(s) = [Acc_m(s) - Acc_m(first)] / [Oracle_m - Acc_m(first)]
```

Headroom H is reported only when its denominator is positive. First selects the first cached candidate. Majority counts non-missing extracted answers, breaks ties by earliest candidate index, and returns index zero if all answers are missing. AR selectors minimize either summed negative log-likelihood or its per-token mean. The oracle is a ceiling for these fixed samples, not an executable selector.

For an adapted model, rho_m is linear CKA averaged across corresponding transformer layers, measured against its own AR base on common held-out text. For column-centered hidden-state matrices X and Y:

```text
CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
rho_m = mean_layer CKA(Z_m, Z_base(m))
Hypothesis: Spearman(rho_m, H_m(AR)) > 0
```

LLaDA has no AR base in this study and is excluded from this correlation. Macro accuracy averages per-model accuracies. A useful selector should improve on majority without hiding a regression in one of the two non-floor models. The broader compute question compares reranking gains with gains from increasing denoising steps; it remains unresolved in this draft.

## 4. Methodology

**Data and models.** GSM8K [5] supplies numeric-answer math problems. Selector development uses 200 training questions. The first 500 selected official test questions are separated into a 200-question merge gate and a 300-question final split. These roles must not be conflated. Models are DiffuGPT-S, DiffuGPT-M, DiffuLLaMA, Dream-Base 7B, and LLaDA-Base 8B. Candidates are generated once and reused by every selector.

**Generation and scoring.** Iteration 1 uses 64 denoising steps. Dream and LLaDA use temperature 0.7. DiffuGPT-S uses one shot and 192 new tokens because of its context limit; the other models use four shots and 256 new tokens. Scorers include GPT-2, Qwen2.5-0.5B, TinyLlama, Qwen2.5-7B, and Llama-2-7B. We distinguish the matched base of a model from an external scorer: Qwen2.5-7B is Dream's base; LLaDA has no matched AR base.

**Preservation.** Hidden states are pooled over held-out WikiText-103 segments. The pipeline measures plain and debiased CKA, with and without the initial token. Attention checks distinguish bidirectional diffusion representations from causal AR representations. The nominal comparison uses 2,000 segments; source receipts document the corpus-size repair.

**Agent search and statistics.** Arbor edits only the selector. Its task protects evaluation files and fixed data, with static checks on imports and valid candidate indices. Human/operator feedback rejected cross-model consensus twice. The search produced likelihood tie-breaking, likelihood-weighted voting, and arithmetic-verification-weighted voting. The last assigns weight 1 to samples with checked calculations that all pass and 0.5 otherwise. We compare correctness on matched questions using 10,000-resample bootstrap intervals and exact two-sided McNemar tests. Static checks alone do not prove absence of cross-call state; the next task adds external order-invariance regression checks.

<!-- pagebreak -->

## 5. Results

### 5.1 Fixed-candidate final evaluation

Table 1 reports iteration-1 final accuracy, in percent, on 300 questions per model. The historical AR column is retained for auditability: its scores include the trailing-text confound described below. It must not be read as the result of corrected cut-text scoring. The agent selector is arithmetic-verification-weighted voting.

| Model | First | Majority | AR old best | Agent | Oracle |
|---|---:|---:|---:|---:|---:|
| DiffuGPT-S | 1.7 | 2.3 | 2.0 | 2.3 | 14.7 |
| DiffuGPT-M | 0.7 | 1.3 | 2.3 | 1.3 | 8.7 |
| DiffuLLaMA | 0.3 | 2.0 | 2.3 | 2.0 | 9.7 |
| Dream-Base | 41.0 | 73.3 | 32.0 | 73.0 | 88.0 |
| LLaDA-Base | 48.7 | 73.0 | 52.0 | 73.0 | 88.3 |

The best historical AR scorers are Llama-2-7B mean for DiffuGPT-S, TinyLlama mean for DiffuGPT-M, GPT-2 mean for DiffuLLaMA, and Qwen2.5-7B sum for Dream and LLaDA. The three floor models offer little absolute accuracy to recover. On Dream and LLaDA, majority substantially improves over first and closes approximately 69% and 61% of the first-to-oracle gap, respectively.

### 5.2 Agent selector versus majority

The agent selector does not show a detectable improvement over majority. Its final differences are -0.3 percentage points for Dream and 0.0 for LLaDA. Table 2 gives paired confidence intervals and discordant counts; wins/losses refer to questions uniquely correct for the agent/majority. Both models have exact McNemar p = 1.00. These outcomes are inconclusive about small effects, not an equivalence proof.

| Model | Difference (pp) | 95% interval (pp) | Wins/losses | p |
|---|---:|---|---|---:|
| Dream-Base | -0.3 | [-2.0, +1.3] | 3/4 | 1.00 |
| LLaDA-Base | 0.0 | [-1.7, +1.7] | 3/3 | 1.00 |

The agent's gate improvements were measured against an AR-likelihood seed, rather than the stronger majority baseline. A merge therefore did not demonstrate success against our actual objective. On the gate, Dream gained 1.5 points over majority while LLaDA lost 1.5; this trade-off did not justify a general improvement claim.

### 5.3 Preservation and AR recovery

Across four adapted models, recorded CKA is 0.890 for DiffuGPT-S, 0.866 for DiffuGPT-M, 0.748 for DiffuLLaMA, and 0.509 for Dream. Across ten original AR selectors, four have Spearman correlation +1.0, five +0.8, and one +0.4 with recovered headroom. The smallest attainable exact two-sided permutation p with four models is 2/24, approximately 0.083. Three points are floor models, and corrected likelihood features are not yet available. We therefore report a descriptive pattern, not evidence that preservation predicts useful selection.

### 5.4 Completed fidelity controls

Each row below has 100 unique questions. These are greedy 256-step runs; their accuracy is not directly comparable to selecting among ten temperature-0.7 samples. Dream's two prompt settings must also remain distinct.

| Configuration | Correct / n | Accuracy (%) |
|---|---|---:|
| Dream sampler, our four-shot prompt | 81/100 | 81.0 |
| Dream published eight-shot prompt | 80/100 | 80.0 |
| LLaDA published sampler, four shots | 73/100 | 73.0 |

The controls show that the low iteration-1 first-sample accuracy is sensitive to decoding settings. They do not isolate temperature from step count or establish that extraction is error-free.

<!-- pagebreak -->

## 6. Evaluation Audit and Discussion

### 6.1 Likelihood scores were partly scores of unrelated text

The original trimming rule stopped at a new question marker. Many completions instead continued with unrelated text after their final answer. Among the first 1,000 test candidates per model, 660 Dream and 497 LLaDA candidates retained more than 20 characters after the answer line. Their AR likelihoods therefore combine reasoning quality with the likelihood and length of unrelated continuations. Corrected scoring cuts text at the answer marker and retains separate feature names. Because that comparison is unfinished, this draft does not conclude that properly scored AR selection is ineffective.

An operator-agent review of 20 development candidates found 16 unambiguous final-marker matches and four flagged cases: one trailing-text extraction mismatch, one ambiguous repetition, and two malformed long numbers whose digits change under float normalization. This is a small, selected agent audit, not independent human annotation or an estimate of dataset-wide error rate. The protected grader remains unchanged, so historical comparisons retain a common grading rule. The findings motivate a separately controlled extraction study rather than a silent repair.

### 6.2 A stronger baseline changes what counts as progress

Six protocol-specified majority variants compare earliest, shortest-text, and calculation-count tie-breaking, with missing answers counted or ignored. Selection uses development macro accuracy; the script freezes the chosen rule before loading gate labels. Earliest-index majority ignoring missing answers wins on development. On the gate it reaches 71.0% for Dream and 73.5% for LLaDA, exactly matching ordinary majority with zero discordant correctness pairs. This calibration yields no new selector gain, but sets an appropriate bar for another search.

The next task's acceptance rule requires at least three net gate questions on each of Dream and LLaDA relative to the frozen bar, paired statistics, and an ablation. This is a proposed review rule, not a completed experiment. A stateful cache using other models' answers is outside scope even if it increases a score. Native protected paths and explicit constraints support review but do not replace it.

### 6.3 Status at draft closure

At 10:27 IST on 21 November, generation had completed 500 questions for Dream at 16 and 32 steps and for LLaDA at 16 steps; LLaDA at 32 steps had 459 of 500. These are progress counts, not completed quality-compute comparisons. A GPU device error interrupted later work and retry limits were exhausted. Partial outputs are preserved and queues are paused. No iteration-2 Arbor search, adopted rule, or new final-split evaluation is reported.

The corrected AR comparison, DLM-native self-scoring, complete step-budget analysis, and per-question preservation analysis are unfinished. Consequently the full reranking-versus-more-denoising question and a frozen transfer to a second dataset remain open. The hardware failure is an infrastructure event and supplies no scientific evidence for or against the hypothesis.

## 7. Limitations

This study covers one numeric-answer benchmark and four adapted models, three near the accuracy floor. Model scale, architecture, pretraining, prompting, and sampling are not isolated interventions on preservation. CKA cannot establish causation, and a positive correlation under this design would not identify why a selector helps. Published verifier and diffusion-specific selector baselines have not been reproduced. Selection and feature-extraction compute must be included in any future comparison with more denoising steps.

The final comparison reports one frozen agent rule against majority, without a component ablation establishing a new mechanism. Repeated use of the 200-question gate can also influence model selection. Static leakage checks and order reversal are incomplete safeguards; independent code review remains necessary. This first draft has not received independent member sign-off and is not submission-ready.

## 8. Conclusion

In the completed fixed-candidate evaluation, majority vote is a strong baseline for Dream and LLaDA, and the agent-designed arithmetic rule shows no detectable gain over it on the held-out final questions. Representation preservation remains an unconfirmed predictor: the model sample is small, dominated by floor effects, and its likelihood-based target is confounded by trailing text. Completed fidelity runs and the extraction audit clarify why aggregate benchmark accuracy alone is not enough to validate a pipeline. A conclusive next study needs corrected scoring, the frozen majority bar, complete compute comparisons, ablations, and transfer. We preserve these requirements rather than presenting unfinished controls as positive results.

<!-- pagebreak -->

## References

[1] Shansan Gong et al. Scaling Diffusion Language Models via Adaptation from Autoregressive Models. ICLR, 2025. https://arxiv.org/abs/2410.17891

[2] Shen Nie et al. Large Language Diffusion Models. 2025. https://arxiv.org/abs/2502.09992

[3] Jiacheng Ye et al. Dream 7B: Diffusion Large Language Models. 2025. https://arxiv.org/abs/2508.15487

[4] Xuezhi Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR, 2023. https://arxiv.org/abs/2203.11171

[5] Karl Cobbe et al. Training Verifiers to Solve Math Word Problems. 2021. https://arxiv.org/abs/2110.14168

[6] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of Neural Network Representations Revisited. ICML, 2019. https://arxiv.org/abs/1905.00414

[7] Jiajie Jin et al. Toward Generalist Autonomous Research via Hypothesis-Tree Refinement. 2026. https://arxiv.org/abs/2606.11926

### Artifact and drafting provenance

Primary numerical evidence is in jarjlol/diffullama, branch iter2/aryan: codification/results/iter1/numbers/baselines_final300.json, paired_final300.txt, analyze_test500.txt; and codification/results/iter2/aryan/fidelity_complete.json, support-N4/, wrapup_snapshot.json. No raw candidates, gold answers, model weights, or credentials accompany this draft.

This assembly adapts the member-authored section drafts on mesh/8-introduction, mesh/9-related-work, mesh/10-problem-formulation, mesh/11-methodology, and mesh/12-experiments. Abstract and conclusion were written after assembling the body. The related-work scope is deliberately narrowed to the seven primary sources above; broader novelty and literature-coverage claims are not made. Source versions and build instructions are recorded beside this file.
