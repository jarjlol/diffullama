# Observations and efforts across the research cycle

BITS F471 Group 3 | Task 5 | 21 November 2026

Our work moved from finding a defensible research gap to testing answer selection for diffusion language models. The strongest completed experimental result is a limitation: the selector produced by Arbor did not improve on majority vote on the held-out questions. Iteration 2 has already improved the diagnosis of the decoding gap, and its cut-text re-scoring, step-budget arm and self-score have finished; the stronger selector search remains unfinished. This report separates those completed observations from planned work. [S1, S7, S8]

## Anchor paper and understanding the method

We selected *Scaling Diffusion Language Models via Adaptation from Autoregressive Models* as our anchor and inspected its inference code alongside the paper. Understanding the released implementation changed the questions we could responsibly ask: frozen-checkpoint inference was feasible on the shared workstation, whereas adaptation training was outside the available allocation. We therefore narrowed the project to inference-time measurements rather than promising an unrun training study. [S1, S2]

The main difficulty was keeping the paper, checkpoint and evaluation protocol aligned. Later checks found that the anchor's GSM8K results used fine-tuning, so they could not serve as few-shot reproduction targets. In codification we documented this mismatch, retained the low-performing anchor checkpoints as measured cases, and used stronger diffusion models to study selection where correct candidates were actually available. [S7]

## Unique research task

For the research-lifecycle assignment we proposed Continuous Gap Liveness Verification: repeatedly checking whether subsequent work has already addressed a candidate gap. The decision log records this as the submitted task. Its practical value appeared during our own project, when a related-work check led us to withdraw a broad novelty claim about structure-guided remasking. That is an example of the checking task in use; it is not evidence that we completed or evaluated an autonomous novelty-monitoring system. [S1, S2]

Our change in practice was to keep dated decisions and corrections, including rejected directions, rather than treating an agent's first gap assessment as a durable fact. The records distinguish scientific objections from resource constraints so a direction rejected for compute is not mistakenly described as technically disproven. [S2]

## Literature review generation

We implemented a QUAL-SG-inspired retrieval, ranking and survey-generation pipeline, then added the second supplied paper's evaluation framework. The implementation used OpenAlex/arXiv retrieval and TF-IDF in place of unavailable learned embeddings; some LLM stages were executed in recorded agent sessions rather than through a reproducible hosted endpoint. These are documented substitutions, not exact reproductions. [S3]

The original output contained 48 references, with 9 automatic matches against 33 anchor related-work citations and F1 0.222. A later relevance-judged reference set reported 11 matches and F1 0.272. These are the stored matcher's outputs: inspection found one questionable fuzzy match between different titles in the later set, so the apparent gain needs a match-level audit. We preserved the earlier survey because replacing its numbered reference set would invalidate its positional citations. [S3, S4]

<!-- pagebreak -->

## Limitations and gap extraction

We implemented separate Extractor, Analyzer, Reviewer and Citation roles, followed by judging, feedback and consolidation. The multi-agent version consolidated 33 statements into 12 limitations. The team then assessed each against prior work and the inference-only constraint, declining to pursue the list wholesale rather than accepting agent-generated gaps as research directions. [S2, S5]

The exercise exposed weaknesses in both generation and evaluation. An agent confused the released checkpoint's code result with a separately trained model; the correction record traces the claim to the Reviewer output and records its removal. Multiple agents and a high judge score did not establish factual correctness. The lexical clustering substitute also missed differently phrased statements about the same mechanism, illustrating the cost of replacing semantic retrieval with TF-IDF. [S2, S5]

The reported coverage rose from 0.583 to 1.000, but the available ground truth contained author-stated limitations that the Extractor was designed to reproduce. We therefore do not present that increase as evidence of better research quality. OpenReview access failed; reviewer weaknesses were not established to be absent. The recorded reduction in generic statements, from three to zero, is narrower evidence about this output, without independent human validation. [S5]

## Problem formulation

The selected problem P1 asks whether preservation of an autoregressive backbone predicts recoverable accuracy from answer selection, and whether selection is preferable to spending compute on more denoising steps. We narrowed the implementation to frozen diffusion models generating candidate answers to GSM8K questions. Selectors choose among one model's own candidates, while first-sample, majority-vote and oracle accuracy provide reference points. [S6, S7]

This formulation made an important scope gap visible. Iteration 1 measured CKA similarity against selection headroom at one step budget; it did not measure the full reranking-versus-more-steps quantity in the idea specification. We retain that distinction in the report and reserve the compute comparison for completed step-budget results. No mathematical proof or significant preservation effect is claimed. [S7, S8]

## Ideation and choosing an implementable idea

Our ResearchAgent implementation chained problem, method and experimental-design generation with criterion-specific reviews and refinement. We added the prior gaps and actual resource constraints to the prompts, and the team selected P1 and P2 before generating and ranking their candidate ideas. API limits and retired model endpoints interrupted early runs; the workflow gained response caching and resumed on locally served models. [S6]

The output audit found lineage errors, unavailable proposed data and saturated ratings: 78 of the 100 final ratings were the maximum score. Consequently, ranking was a proposal aid rather than evidence of novelty or feasibility. Idea 1.1 was selected for inference-only codification, but its own scorer-family description still needed correction: Dream uses a Qwen-family base and LLaDA has no autoregressive base. The audit record also lacks the final run cache/probe needed to verify its exact generator/reviewer provenance. [S6, S7]

<!-- pagebreak -->

## Codification iteration 1

We built generation, autoregressive scoring, CKA measurement, baseline selection and an Arbor task pack. Arbor edited the selector in review mode, with feedback recorded verbatim. We compared five diffusion models, generating 10 candidates per question. Development used 200 train-derived questions; 500 test questions were divided into a 200-question merge gate and a separate 300-question final split. Iteration 1 used 64 denoising steps. [S7]

Concrete failures changed the implementation. DiffuGPT-S needed a shorter prompt to fit its context. A causal-attention path would have inflated diffusion-versus-base similarity, so a bidirectionality check was added. CKA received debiased and first-token-excluded variants. A dotted scorer name had broken feature lookup. Dream's low-temperature samples lacked useful diversity, leading to a higher sampling temperature with the resulting fidelity confound recorded explicitly. [S7]

Final-split accuracy (%) is shown below. Each model has 300 questions. "Best AR" is the largest observed AR-selector accuracy on this split, a descriptive maximum rather than a selector selected independently of final labels. Its scores use the original trimmed text. [S9]

| Model | First | Majority | Best AR | Arbor | Oracle |
|---|---:|---:|---:|---:|---:|
| DiffuGPT-S | 1.7 | 2.3 | 2.0 | 2.3 | 14.7 |
| DiffuGPT-M | 0.7 | 1.3 | 2.3 | 1.3 | 8.7 |
| DiffuLLaMA | 0.3 | 2.0 | 2.3 | 2.0 | 9.7 |
| Dream | 41.0 | 73.3 | 32.0 | 73.0 | 88.0 |
| LLaDA | 48.7 | 73.0 | 52.0 | 73.0 | 88.3 |

Arbor merged verification-weighted voting against its weaker seed, not against majority vote. On the final split it recovered three Dream questions majority missed and lost four majority got right; for LLaDA the counts were three and three. Both exact McNemar comparisons had p = 1.000. The result supports no demonstrated gain over majority under this protocol; it does not establish equivalence. [S7, S9]

The preservation analysis covered only four models with autoregressive bases, three near the task floor. Reported correlations were descriptive, with exact permutation p-values at least 0.083. Neither the sample size nor the missing step-budget comparison supports the full claim that preservation predicts reranking advantage. The cut-text re-scoring described below confirms the AR loss, so the descriptive correlation remains the only preservation claim. [S7, S8, S10]

## What feedback changed

Review replies rejected cross-model cached answers twice, corrected the claimed LLaDA lineage, and redirected Arbor toward within-item arithmetic checking. These interventions constrained the task and produced a testable selector. They did not make that selector outperform majority. Refinement of the selector through Arbor is recorded; a model-written revision of the original idea specification is a separate planned activity and is not counted as completed. [S7, S8]

<!-- pagebreak -->

## Codification iteration 2

The agreed target is the full iteration run in priority order, with the paper's results frozen on 22 November at 18:00 IST. The completed fidelity checks change our interpretation of iteration 1. On the same first 100 test questions, Dream at 256 steps and greedy decoding scored 81%, compared with 41% for the original first sample; LLaDA scored 73%, compared with 51%. A separate Dream published-prompt run completed 100 questions, scoring 80 correct. [S8, S11]

We independently recounted the saved fidelity records on Blackwell and saved their aggregate counts and SHA-256 digests. These observations support decoding settings as an explanation of much of the earlier accuracy gap. They do not isolate temperature from step count, reproduce the full benchmark, or prove that every pipeline path is bug-free. [S8, S11]

Iteration 2 also identified trailing text after the final answer as a scoring confound. Original trimming stopped at the next question marker, leaving unrelated continuation text in the AR likelihood calculation. The new cut-text path limits scoring through the answer line. The cut re-scoring has now finished: the best cut-text AR selector reaches 48.8% on Dream and 52.2% on LLaDA, against majority 72.4% and 73.2% (exact McNemar p < 0.001); cutting recovered 13.0 points on Dream's strongest scorer and left LLaDA unchanged. The AR-selection loss therefore survives removal of the confound. [S8, S13]

The step-budget arm and diffusion self-score have also finished. Majority goes 3.2→23.4→72.4% on Dream and 3.2→51.0→73.2% on LLaDA for K = 16, 32 and 64; raising K from 16 to 32 adds 9.8 and 19.2 points to the first sample, while majority at K = 16 adds 0.6 and −0.4. Dream's own likelihood reaches 31.5% on the gate against 71.0% for majority. [S13] The per-question preservation test and stronger Arbor run remain pending; code and queued commands are not completed results. The Arbor preparation uses a frozen tuned-majority seed, curated findings and failure memories, protected evaluation files, explicit red lines and an order-invariance check. These are planned validity improvements; no accuracy gain is attributed to them yet. The completed numbers above and the original final-split table remain separate because their samples and decoding settings differ. [S8, S12]

## Effort, limits and remaining verification

The recurring work was checking assumptions before expensive inference: matching checkpoints to their real bases, tracing erroneous statements, distinguishing lookup failures from missing papers, and keeping development, merge-gate and final evaluation roles separate. Agent-generated text and self-evaluated ratings repeatedly needed external checks. A useful negative result required as much attention to baselines and provenance as a positive result would. [S2, S3, S5, S6, S7]

The CPU analyses landed as well: a curated findings memory and ranked GPU queue (T3), a verified SOTA table (N3) in which every published GSM8K test-time number uses an Instruct checkpoint and no prior work links preservation to reranking gain, and a leave-one-out check (N4) that keeps Spearman ρ ≥ +0.5 for 9 of 10 selectors when any one model is dropped. [S14, S15, S16]

Before final submission we must re-pin the iteration-2 evidence at the freeze, attach the independent member review, and verify the final PDF. Pending runs, a published selector comparison and frozen transfer evaluation must remain labelled unfinished if they do not complete. No gain beyond the reported majority-vote baseline is established in the evidence checked for this report. [S8, S12]

## Evidence sources

- [S1] Project brief and [S2] dated decisions and corrections.
- [S3] Literature review report and [S4] saved overlap outputs.
- [S5] Multi-agent limitations report; [S6] ResearchAgent report and output audit.
- [S7] Codification iteration 1; [S8] iteration-2 draft and issue log; [S13] copied iteration-2 result files.
- [S14] N4 analysis; [S15] verified SOTA table; [S16] curated findings memory and queue.
- [S9] Final-split baseline and paired-statistics artifacts; [S10] preservation analysis.
- [S11] Independently recounted fidelity snapshot; [S12] iteration-2 design and priorities.

[S1]: https://github.com/jarjlol/diffullama/blob/69d76a8/project-docs/01-project-brief.md
[S2]: https://github.com/jarjlol/diffullama/blob/69d76a8/project-docs/02-decision-log.md
[S3]: https://github.com/jarjlol/diffullama/blob/69d76a8/litreview/REPORT.md
[S4]: https://github.com/jarjlol/diffullama/blob/69d76a8/litreview/output/evaluation_results_llm_judged.json
[S5]: https://github.com/jarjlol/diffullama/blob/69d76a8/limitations/REPORT.md
[S6]: https://github.com/jarjlol/diffullama/blob/69d76a8/ideation/REPORT.md
[S7]: https://github.com/jarjlol/diffullama/blob/69d76a8/codification/ITERATION_1.md
[S8]: https://github.com/jarjlol/diffullama/blob/66697fd/codification/ITERATION_2.md
[S9]: https://github.com/jarjlol/diffullama/tree/69d76a8/codification/results/iter1/numbers
[S10]: https://github.com/jarjlol/diffullama/blob/69d76a8/codification/results/iter1/numbers/analyze_test500.txt
[S11]: evidence/fidelity_snapshot.json
[S12]: https://github.com/jarjlol/diffullama/blob/258ae43/codification/ITERATION_2_PROMPTS.md
[S13]: evidence/iter2/README.md
[S14]: https://github.com/jarjlol/diffullama/blob/86f5c46/codification/ANALYSIS_ITER2.md
[S15]: https://github.com/jarjlol/diffullama/blob/22a11cd/codification/SOTA.md
[S16]: https://github.com/jarjlol/diffullama/tree/024ab82/codification/memory
