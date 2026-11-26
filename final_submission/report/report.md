# Observations and efforts across the research cycle

BITS F471 Group 3 | Task 5

## Proposal. What could we test with the available compute?

We chose *Scaling Diffusion Language Models via Adaptation from Autoregressive Models* as the anchor paper. Reading the paper alongside its released inference code helped us scope a project we could run on a shared workstation. We used released checkpoints with fixed weights and varied what happened during inference. We also checked the anchor's math evaluation and found that its results on Grade School Math (GSM8K) used fine-tuning, so we treated them as a separate setting from our few-shot evaluation. [S1, S2, S7]

For the research-lifecycle assignment, we proposed periodically checking whether new work has already addressed a research gap. We used that practice when a related-work check led us to withdraw a broad novelty claim about structure-guided remasking. We kept dated records of such decisions and distinguished gaps already addressed in prior work from ideas we set aside because of compute limits. [S1, S2]

## Task 1. How well did the literature review recover relevant work?

We adapted a survey-generation pipeline to retrieve papers from OpenAlex and arXiv, rank them, and draft a structured review. We used term-frequency ranking because learned text embeddings were unavailable. We then implemented the second supplied paper's evaluation framework to compare the generated reference list with the anchor's related work. Some drafting stages ran in recorded agent sessions. [S3]

The survey cited 48 papers. The automatic matcher recovered 9 of the anchor's 33 related-work citations. A later relevance-judged reference set recovered 11 of 33 by the same matcher, though a manual check found one questionable title match. We kept the original survey and its numbered references together so the in-text citations would still point to the right entries. [S3, S4]

## Task 2. Which limitations were worth pursuing?

We assigned separate agents to extract, analyze, review and cite limitations, then collected feedback and consolidated their outputs. The multi-agent pipeline reduced 33 statements to 12 limitations. We checked these against prior work and the inference-only scope before deciding which to pursue. [S2, S5]

Review revealed that one agent had attributed a code result to the released checkpoint when it came from a separately trained model. We traced the error to the review output and corrected the record. We also found that word-overlap clustering missed statements that described the same issue in different language. [S2, S5]

The coverage metric rose from 0.583 to 1.000 on author-stated limitations, the same kind of text the extractor was designed to recover. The output contained fewer generic statements, from three to zero. That reduction lacks independent validation. A separate check of reviewer-stated limitations stopped when review access failed. [S5]

<!-- pagebreak -->

## Task 3. Which idea could we turn into an experiment?

We used ResearchAgent to propose problems, methods and experiment designs, with review and revision at each stage. We gave it the earlier gap analysis and compute constraints, then pursued two directions: answer selection and denoising schedules. Service limits and retired model endpoints interrupted early runs. We added response caching and resumed using locally hosted models. [S6]

The rankings helped us generate a shortlist. In the final rated set, 78 of 100 ratings were at the maximum, and the audit found errors about model ancestry and data availability. We chose answer selection because it was feasible without training and gave us a clear test of whether a selector could pick better answers from the same candidates than majority vote. We also asked whether representation similarity to an autoregressive base predicted the accuracy an autoregressive scorer could recover, and how selection gains changed with more denoising steps. [S6, S7]

## Task 4. What changed between the two implementation rounds?

**Iteration 1.** We built a pipeline to generate ten answers per question, score them with autoregressive models, and measure how closely the diffusion models' internal representations resembled those of their base models. We separated development, the merge decision and the final evaluation. Review feedback kept selectors within one model's candidate pool and directed the Arbor agent toward arithmetic checking. We also added attention checks and fixed a feature lookup error. [S7]

On 300 held-out questions per model, the Arbor rule reached 73.0% accuracy on both Dream and LLaDA. Majority vote reached 73.3% and 73.0%, respectively. The rule improved on its starting selector during development. The final comparison showed no detectable gain over majority. The representation-similarity comparison covered only four adapted models, three of which rarely generated a correct answer. [S7, S9, S10]

**Iteration 2.** We checked the decoding setup against published settings. On the first 100 test questions, greedy decoding at 256 steps raised Dream's first-answer accuracy from 41% to 81% and LLaDA's from 51% to 73%. We recounted the saved outputs independently. The comparison changes both sampling and step count. [S8, S11]

We found that autoregressive scores included unrelated text after the final answer. On 500 questions per model at 64 denoising steps, cutting that text raised Dream's strongest score-based selector by 13.0 percentage points and left LLaDA's unchanged. The corrected selectors reached 48.8% and 52.2%, below majority at 72.4% and 73.2%. The correction also changed Dream's representation comparison enough to leave the original hypothesis unsupported. [S8, S13]

Increasing the budget from 16 to 32 steps improved first-answer accuracy by 9.8 points on Dream and 19.2 on LLaDA. At 256 steps, the best autoregressive selector still trailed majority by 8.6 points on 185 Dream questions and 2.6 on 192 LLaDA questions. Only Dream's gap was statistically clear. Selection using each diffusion model's own score also trailed majority on the separate set of 200 questions used to assess candidate selector changes. An exploratory question-level analysis found no clear association between representation similarity and selection gains. We prepared a second Arbor search but did not run it because the shared accelerators were occupied. [S8, S12, S13]

<!-- pagebreak -->

## Follow-up. What could a decision model select from the same answers?

We gave a four-billion-parameter decision model the question and all ten candidates in a single pass. At 64 denoising steps, across the same 500 GSM8K questions for Dream and LLaDA, it averaged 78.0% accuracy, 5.2 percentage points above majority vote at 72.8%. Resampling by question, keeping both models' results together, gave a 95% confidence interval of 3.0 to 7.4 points for the gain. [S14, S15]

At 256 steps, the decision model averaged 85.7% accuracy against 81.7% for majority vote on 185 Dream and 192 LLaDA question instances. The gain was 4.0 points (95% confidence interval 0.9 to 7.1). On the same questions it was 3.5 points at 64 steps. The change of 0.5 points was inconclusive (95% confidence interval -4.0 to 5.1). Many 256-step candidates still carried extra text on the answer line after trimming. [S17, S22]

We used the development set to choose when to call the decision model. On the 64-step test, calling it only when fewer than five candidates agreed reached 78.5% accuracy with 47.8% of the calls. Calling it on every question reached 78.0%. The accuracy difference was inconclusive. [S21]

Three controls tested what information the selector used. With only the distinct candidate answers and the question, accuracy fell to 42.8%, compared with 78.0% with the full candidates. Where the pool lacked a correct answer, adding one led the model to pick it on 24 of 115 cases (20.9%). Given two answer options on 40 questions, one correct and one drawn from another problem, it chose correctly 37 times (92.5%). The model can select correct answers. How much it relies on solving the problem, recognizing an answer, or checking the candidate's reasoning remains open. [S14, S15]

On 144 HumanEval programming problems, the model selected passing code on 58 (40.3%), matching the number with at least one passing candidate among the ten samples. The corrected filter based on examples in the prompt passed 40 problems (27.8%). On the 38 solvable problems with parseable examples, the decision model passed 38 and the filter passed 36. On the 20 without examples, the counts were 20 and 4. [S14, S15]

On the 142 problems with usable HumanEval+ tests, the same choices passed 40.8% under the standard tests and 35.9% under the stronger tests. At least one candidate passed the stronger tests on 53 problems. The decision model picked one on 51. Five of the seven lost successes came from problems where every candidate failed the stronger tests. HumanEval+ adds tests to the same problems. [S18]

In a separate recognition probe, the model chose a reference solution over a known failing variant on 59 of 60 problems. On the 20 development problems, removing the function descriptions reduced correct picks from 20 to 16. The probe measures selection between two supplied candidates. LLaDA produced mostly newlines at 64 steps, so the comparison of generated code covers Dream only. We stopped the Mostly Basic Python Problems pilot after Dream produced no passing candidates on the first 20 problems at the same budget. [S14, S19, S20]

## Task 5. How did we turn the work into a paper?

We assembled the paper around reproducible comparisons, traced the figures to saved outputs, and kept the final 300-question comparison separate from later test runs. Feedback prompted us to clarify which comparisons shared questions and which lacked a clean compute measurement. We verified the five-page build with four body pages and one page of references. The decision-model results extend the earlier selector study. They leave the representation hypothesis open and identify controls needed for a full comparison. [S16, S14]

[S1]: https://github.com/jarjlol/diffullama/blob/69d76a8/project-docs/01-project-brief.md
[S2]: https://github.com/jarjlol/diffullama/blob/69d76a8/project-docs/02-decision-log.md
[S3]: https://github.com/jarjlol/diffullama/blob/69d76a8/litreview/REPORT.md
[S4]: https://github.com/jarjlol/diffullama/blob/69d76a8/litreview/output/evaluation_results_llm_judged.json
[S5]: https://github.com/jarjlol/diffullama/blob/69d76a8/limitations/REPORT.md
[S6]: https://github.com/jarjlol/diffullama/blob/69d76a8/ideation/REPORT.md
[S7]: https://github.com/jarjlol/diffullama/blob/69d76a8/codification/ITERATION_1.md
[S8]: https://github.com/jarjlol/diffullama/blob/2d7cbb36e089be032ef9d27a8374d2adb3e0f38f/codification/ITERATION_2.md
[S9]: https://github.com/jarjlol/diffullama/tree/69d76a8/codification/results/iter1/numbers
[S10]: https://github.com/jarjlol/diffullama/blob/69d76a8/codification/results/iter1/numbers/analyze_test500.txt
[S11]: evidence/fidelity_snapshot.json
[S12]: https://github.com/jarjlol/diffullama/blob/88bac1b81f3e42899dfa95d09b1afd790748c6e3/final_submission/report/evidence/iter2/k256_step_arm_output.txt
[S13]: evidence/iter2/README.md
[S14]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/DECISION_RERANK_RESULTS.md
[S15]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/results/controls.json
[S16]: https://github.com/jarjlol/diffullama/pull/74
[S17]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/results/k256_decision.json
[S18]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/results/humaneval_plus_result.json
[S19]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/results/humaneval_recognition_result.json
[S20]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/results/humaneval_recognition_nodoc_result.json
[S21]: https://github.com/jarjlol/diffullama/blob/cf8e51c9eafb2284a1f17f1342bf73cd26c29c63/codification/results/weak_vote_threshold.json
[S22]: https://github.com/jarjlol/diffullama/issues/78#issuecomment-7178740360
