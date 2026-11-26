# Automating research on diffusion language models

BITS F471 Group 3 | Task 5 discussion report

Across five tasks, we tested how far language models and scripts could take a research project, from finding related work to writing the paper. Our anchor was *Scaling Diffusion Language Models via Adaptation from Autoregressive Models* (DiffuLLaMA). Reading the paper alongside its released code helped us choose experiments we could run with existing models and shared computing resources. We also proposed a recurring check for whether new papers had already addressed a research gap. A related-work check led us to withdraw a broad novelty claim about structure-guided remasking. [S1, S2]

## Task 1. How much of the literature review could we automate?

**Paper and tools.** We adapted the QUAL-SG framework from *SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models*. Python scripts retrieved papers from OpenAlex and arXiv, ranked them, and prepared a reference brief. A team member prompted Claude with that brief to draft the survey. We adapted the evaluation in *Large Language Models for Automated Literature Review* to check the citations and compare the survey with the anchor's related work. [S3, S4, S5]

**How it worked.** The pipeline assembled 48 real references and Claude turned them into a usable first draft. The automatic matcher found 9 of the anchor's 33 related-work references. An alternative list ranked by a language model matched 11 of 33, though a manual check found one questionable title match. We checked source records, corrected a citation linked to the wrong paper, and changed the evaluator to distinguish lookup failures from missing references. For the manuscript, we reviewed which papers actually supported our questions and kept 12 of the 48 survey references. [S5, S6]

## Task 2. Could agents identify research gaps we would pursue?

**Paper and tools.** We implemented *Multi-Agent LLMs for Generating Research Limitations*. Separate prompts asked agents to extract stated limitations, inspect the method, act as reviewers, and look for evidence in related papers. The workflow scored, revised, and combined 33 initial statements into 12 gaps. [S7, S8]

**How it worked.** The roles helped us look beyond the paper's own list, but a reviewer agent attributed a coding result to the wrong model. We went back to the anchor's results table, traced the mistake to the reviewer output, and corrected the deliverable. We refined the list, ran the final judging pass ourselves, and screened the gaps against prior work and our compute budget. We set aside the twelve proposed directions after that review. The high automated coverage score mainly measured recovery of limitations the authors had already stated. [S2, S8, S9]

<!-- pagebreak -->

## Task 3. Could an agent choose a feasible idea?

**Paper and tools.** We implemented *ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models*. The pipeline proposed problems, methods, and experiments. Reviewer prompts scored each stage and fed comments back over three rounds. We added the earlier gap analysis and our resource limits, cached model replies, and resumed with locally hosted models after service limits interrupted runs. [S10, S11]

**How it worked.** The pipeline produced concrete proposals about selecting answers and changing denoising schedules. The ratings were much less useful. Of 100 final ratings, 78 were the maximum, even though some proposals contained factual or practical errors. We checked the suggestions against original papers and available datasets. Some ideas assigned Dream the wrong base model and incorrectly described LLaDA as an adapted model. Others relied on a training corpus that had never been released. We chose the answer-selection idea because we could test it on released models, with a clear comparison against majority vote. [S11, S12]

## Task 4. What did two rounds of automated experimentation achieve?

**Paper and tools.** We used Arbor, the implementation of *Toward Generalist Autonomous Research via Hypothesis-Tree Refinement*, to propose, code, and test changes to an answer selector. Our Python pipeline generated ten candidate answers per question and reused them across selectors. We compared each selector with majority vote, which picks the most frequent answer. [S13, S14]

**Iteration 1.** Arbor twice proposed using another model's cached answer. We rejected that shortcut to keep every selector within one model's candidate set. We corrected a model-lineage error and directed Arbor to check the arithmetic within each candidate instead. Its resulting rule beat Arbor's starting rule. On 300 held-out questions per model, it reached 73.0% on both models, compared with 73.3% for Dream and 73.0% for LLaDA using majority vote. The automated search did not improve on the baseline we cared about. [S14]

**Iteration 2.** We chose the follow-up checks ourselves and had the agents run and document them. Using published decoding settings recovered much of the first-answer accuracy we had been missing. We also found that the likelihood scorer read unrelated text after the answer and reran it on trimmed outputs. This improved Dream's score, but the corrected likelihood selectors still trailed majority vote by 23.6 points on Dream and 21.0 on LLaDA at 64 steps. The gap narrowed at 256 steps. The representation-preservation hypothesis remained unresolved, and the planned second Arbor search did not run because the shared accelerators were occupied. Both iterations clarified what was failing without producing a selector that beat majority vote. [S15, S16]

<!-- pagebreak -->

## Task 4 follow-up. What changed when we chose the next experiment ourselves?

After the two rounds, we discussed why the selectors were failing and brainstormed a different approach. We chose to test Perplexity's four-billion-parameter decision model through OpenRouter. In one pass, it saw the question and all ten candidates and picked one. We reused the same candidate pools, kept majority vote as the comparison, and asked the pipeline to run controls on what the model could infer from the question or answers alone. [S17, S18]

On 500 Grade School Math (GSM8K) questions per model at 64 steps, the decision model averaged **78.0%**, compared with **72.8%** for majority vote. On the smaller 256-step samples, it averaged **85.7%** versus **81.7%**. With the math questions and distinct final answers alone, accuracy fell to 42.8%. [S18, S23]

On 144 HumanEval programming problems, it selected passing code on **58 (40.3%)**. The example-based filter passed **40 (27.8%)**. The gap was larger where the filter had no examples to use. It passed 4 of 16 solvable problems without examples and 36 of 42 with them. With the stronger HumanEval+ tests, the decision model picked passing code on 51 of the 53 problems with a passing candidate. In a separate code probe, the model picked the reference solution over a known failing variant on 59 of 60 problems. We report the improvement as answer selection and leave open how the model reached each choice. [S18, S19, S24]

## Task 5. How did we automate implementation, collaboration, and writing?

**Tools and inputs.** We used GitHub and IssueMesh to coordinate coding agents working in Claude Code, Codex, and OpenCode. IssueMesh routed assignments, recorded who was working on each item, and gave reviewers a place to find the code, commands, and results. The agents drafted sections from the outputs of Tasks 1–4, then drafted the abstract and conclusion from the assembled manuscript, *When Does Selection Beat Majority Vote?* LaTeX produced the paper PDF. [S20, S21]

**How it worked.** Agents could pick up an issue from its recorded status, run checks, and review another member's changes. We chose which experiments to run and which claims to carry into the paper. For example, an independent recount of the saved HumanEval outputs corrected the breakdown of solvable problems with and without examples in both the paper and report. The workflow made the correction traceable to the data. The team still had to decide how that evidence should shape the final wording. [S20, S21, S22]

[S1]: https://arxiv.org/abs/2410.17891
[S2]: https://github.com/jarjlol/diffullama/blob/69d76a8/project-docs/02-decision-log.md
[S3]: https://arxiv.org/abs/2508.17647
[S4]: https://arxiv.org/abs/2412.13612
[S5]: https://github.com/jarjlol/diffullama/blob/69d76a8/litreview/REPORT.md
[S6]: https://github.com/jarjlol/diffullama/blob/c85bb5c/final_submission/notes/related_work_filtering.md
[S7]: https://arxiv.org/abs/2601.11578
[S8]: https://github.com/jarjlol/diffullama/blob/69d76a8/limitations/REPORT.md
[S9]: https://github.com/jarjlol/diffullama/blob/c85bb5c/docs/pipeline-story.md
[S10]: https://arxiv.org/abs/2404.07738
[S11]: https://github.com/jarjlol/diffullama/blob/4559f77/ideation/REPORT.md
[S12]: https://github.com/jarjlol/diffullama/blob/69d76a8/codification/IDEA_SELECTION.md
[S13]: https://arxiv.org/abs/2606.11926
[S14]: https://github.com/jarjlol/diffullama/blob/69d76a8/codification/ITERATION_1.md
[S15]: https://github.com/jarjlol/diffullama/blob/2d7cbb36e089be032ef9d27a8374d2adb3e0f38f/codification/ITERATION_2.md
[S16]: https://github.com/jarjlol/diffullama/issues/7
[S17]: https://github.com/jarjlol/diffullama/blob/cf8e51c/codification/DECISION_RERANK_PREREG.md
[S18]: https://github.com/jarjlol/diffullama/blob/fcfd46b/codification/DECISION_RERANK_RESULTS.md
[S19]: https://github.com/jarjlol/diffullama/blob/cf8e51c/codification/results/humaneval_plus_result.json
[S20]: https://github.com/jarjlol/diffullama/blob/c85bb5c/docs/pipeline-story.md#coordination
[S21]: https://github.com/jarjlol/diffullama/blob/c85bb5c/final_submission/paper/appendix_provenance.md
[S22]: https://github.com/jarjlol/diffullama/issues/77#issuecomment-7182726710
[S23]: https://github.com/jarjlol/diffullama/blob/cf8e51c/codification/results/k256_decision.json
[S24]: https://github.com/jarjlol/diffullama/blob/cf8e51c/codification/results/humaneval_recognition_result.json
