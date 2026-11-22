# Experimentation / Results and Discussion

<!-- Owner: arjun (issue #12). Target length: ~1.25 pages. Delete this block when final. -->
<!-- Sources: codification/ITERATION_1.md (branch assignment/codification @ 69d76a8), §Results and §Published numbers, with the Spearman-count correction in #28 from codification/results/iter1/numbers/analyze_test500.txt; codification/ITERATION_2.md §Results 1 and I-21/I-22 (branch iter2/aaditya-worker @ 3bc19b6); fidelity counts from codification/results/iter2/aryan/fidelity_complete.json; cut re-score and Dream self-score from codification/results/iter2/cpu/{cvt_dream_test,cvt_llada_test,self_dream_gate}.txt; step arm from codification/results/iter2/step_arm.md (Blackwell, 2026-11-21, worker commit 66697fd). Remaining iteration-2 runs are marked pending. Paper results freeze Sun 22 Nov 18:00 IST. Notation follows problem_formulation.md (#10). -->
<!-- DRAFT 2026-11-21. Iteration-1 numbers from ITERATION_1.md §Results; iteration-2 numbers from the result files named above. The AR conclusion now rests on the cut re-score. -->

**Setup.** We evaluate on GSM8K (Cobbe et al., 2021): 200 dev questions for selector development, and 300 held-out test questions for the final score. Each model $m$ draws $N = 10$ candidates per question. The models are DiffuGPT-S, DiffuGPT-M and DiffuLLaMA (Gong et al., 2025), adapted from GPT-2 and LLaMA-2; Dream 7B (Ye et al., 2025), adapted from Qwen2.5-7B; and LLaDA (Nie et al., 2025), trained from scratch and used as the control with no AR base. All models run 64 denoising steps; DiffuGPT-S uses one shot and 192 new tokens because its 512-token context cannot hold the four-shot prompt (I-7). Scorers are GPT-2, Qwen2.5-0.5B and TinyLlama, plus Qwen2.5-7B for Dream and Llama-2-7B for DiffuLLaMA. Preservation $\rho_m$ is layer-averaged linear CKA (Kornblith et al., 2019) on up to 2,000 held-out 128-token WikiText-103 segments.

**Generation fidelity.** Greedy decoding at the published sampler (256 steps, temperature 0) scores 81.0% on Dream with our 4-shot prompt and 80.0% with the published 8-shot prompt ($n = 100$ each), and 73.0% on LLaDA ($n = 100$), against published 77.2 and 70.3. Iteration 1's first sample on the same questions scores 41.0% and 51.0%. Prompt format and answer extraction are therefore not what lowered iteration 1 for Dream and LLaDA (fidelity was not checked for the three floor models); the 64-step, temperature-0.7 sampler used to obtain diverse candidates is (I-21). How much is due to temperature and how much to step count is not yet separated [unverified].

**Table 1.** GSM8K accuracy (%) on the final split ($n = 300$). Headroom is $H_m$ for the agent's selector. AR-scorer results are in Table 2. Source: `codification/ITERATION_1.md` §Results.

| Model | Shots | First | Majority | Agent selector | Oracle | Headroom |
|---|---|---|---|---|---|---|
| DiffuGPT-S | 1 | 1.7 | 2.3 | 2.3 | 14.7 | 5% |
| DiffuGPT-M | 4 | 0.7 | 1.3 | 1.3 | 8.7 | 8% |
| DiffuLLaMA | 4 | 0.3 | 2.0 | 2.0 | 9.7 | 18% |
| Dream 7B | 4 | 41.0 | **73.3** | 73.0 | 88.0 | 68% |
| LLaDA 8B | 4 | 48.7 | **73.0** | 73.0 | 88.3 | 61% |
| Macro | | | | 30.3 | | |

**Selector results.** No selector beat majority on the final split. The agent's best rule, arithmetic-verification-weighted voting, is indistinguishable from majority: −0.3 points on Dream (95% CI [−2.0, +1.3]; $p = 1.00$) and 0.0 on LLaDA ([−1.7, +1.7]; $p = 1.00$), though both beat the first sample by +32.0 and +24.3. Its dev gain on Dream (+1.5) did not replicate, and the gate split agrees (+1.5 and −1.5; $p = 0.38$ and $0.25$). The merge gate compared the rule with the seed, not with majority, so the merge is not evidence against self-consistency. On the three floor models (oracle 8.7–14.7%) all selectors agree on almost every question.

**AR reranking.** With text cut at the final answer line (I-22), the best AR selector reaches 48.8% on Dream and 52.2% on LLaDA, still 23.6 and 21.0 points below majority (both $p < 0.001$; Table 2, $K = 64$). Cutting raised Dream's Qwen2.5-7B summed score by 13.0 points (95% CI [9.0, 17.0]) and left LLaDA's unchanged (−0.2 [−2.6, 2.2]), so trailing text had lowered Dream's AR accuracy; cutting lifts AR above the first sample (48.8 vs 40.4) but leaves it 23.6 points below majority. Likelihood tie-breaks and vote weights, tried on dev, also did not beat majority. AR reranking loses to majority vote.

**Step budget.** Table 2 varies the denoising budget with the same $N = 10$ and temperature. Both improve steeply: majority goes 3.2→23.4→72.4% on Dream and 3.2→51.0→73.2% on LLaDA for $K = 16, 32, 64$. Raising $K$ from 16 to 32 adds 9.8 (Dream) and 19.2 (LLaDA) points to the first sample, whereas majority at $K = 16$ adds 0.6 and −0.4; at $K = 64$ it adds 32.0 and 24.6. On these runs more denoising is the better use of compute at low $K$; a clean cost comparison is unavailable because generation times come from shared GPUs [unverified as a cost comparison].

**Table 2.** Accuracy (%) and generation time per question ($N = 10$) by denoising budget, on 500 GSM8K test questions. "Best AR" is the best AR selector at that budget: at $K = 64$ the cut-text best, at $K = 16$ and $32$ the best reported; the scorer name records the text. Source: `codification/results/iter2/step_arm.md` (2026-11-21); the $K = 64$ cut rows are from `cvt_dream_test.txt` / `cvt_llada_test.txt`.

| Model | $K$ | First | Majority | Oracle | Best AR (scorer) | s/q |
|---|---|---|---|---|---|---|
| Dream 7B | 16 | 2.6 | 3.2 | 17.4 | 2.4 (GPT-2 mean) | 9.5 |
| Dream 7B | 32 | 12.4 | 23.4 | 54.0 | 13.8 (Qwen2.5-7B cut sum) | 19.3 |
| Dream 7B | 64 | 40.4 | **72.4** | 87.8 | 48.8 (Qwen2.5-7B cut sum) | 36.8 |
| LLaDA 8B | 16 | 3.6 | 3.2 | 24.6 | 3.6 (Qwen2.5-0.5B cut sum) | 13.7 |
| LLaDA 8B | 32 | 22.8 | 51.0 | 76.0 | 28.6 (Qwen2.5-7B cut sum) | 27.4 |
| LLaDA 8B | 64 | 48.6 | **73.2** | 89.2 | 52.2 (Qwen2.5-7B cut sum) | 55.0 |

**DLM self-score.** On the Dream gate split the model's own likelihood reaches 31.5% against 71.0% for majority (2 wins, 81 losses; $p < 0.001$; $n = 200$), so it is not a substitute for an external scorer.

**What worked.** Majority vote is the strong training-free selector on the two non-floor models: it recovers 61–69% of the first-sample-to-oracle gap, and no in-item signal we tried closed the remaining ~15 points on the iteration-1 scores. The recoverable gain is agreement, not likelihood, consistent with self-consistency on AR models (Wang et al., 2023).

**Preservation and gain.** Table 3 reports the iteration-1 correlation, computed on trimmed-text headroom. It does not carry over to the corrected scores: Dream's Qwen2.5-7B-sum headroom flips from −9.7% to +17.7% after cutting ((48.8−40.4)/(87.8−40.4)), and the three floor models have no cut scores, so the four-model Spearman $\rho$ cannot be recomputed. On trimmed scores the order was as predicted ($\rho$ $+1.0$ for 3 selectors, $+0.8$ for 6, $+0.4$ for 1; exact $p \ge 0.083$), but Dream is the non-floor point that set it, and its corrected sign is opposite. We therefore do not claim preservation is supported; even the descriptive trend is not retained. A per-question test (500 test questions, exploratory) does not support it either: the primary predictor's AUC is 0.469 (permutation $p = 0.32$; OR 0.94 per SD [0.75, 1.17]), and no layer is significant after Benjamini–Hochberg correction.

**Table 3.** Layer-averaged CKA $\rho_m$ (DLM vs. its AR base) and AR-recoverable headroom $H_m(s_\psi)$ on the 500-question test set. Headroom is for the Qwen2.5-7B scorer, summed and per token. Headroom uses the iteration-1 trimmed-text AR scores; on cut scores Dream is +17.7% and LLaDA +8.9%, and the floor models have no cut scores. The CKA values do not change. LLaDA has no AR base and is a control. Descriptive only. Source: `codification/results/iter1/numbers/analyze_test500.txt`; the selector counts carry the dated correction in #28.

| Model | CKA $\rho_m$ | Headroom (sum) | Headroom (mean) |
|---|---|---|---|
| DiffuGPT-S | 0.890 | +5.3% | +5.3% |
| DiffuGPT-M | 0.866 | +2.5% | +10.0% |
| DiffuLLaMA | 0.748 | −2.2% | 0.0% |
| Dream 7B | 0.509 | −9.7% | −13.9% |
| LLaDA 8B (control) | — | +9.4% | −8.4% |

## Iteration 2

Landed: the cut re-score, the step-budget arm, the Dream self-score and the fidelity check above; the N4 failure classes are in the report. Also landed: the per-question preservation test (above, exploratory). Not finished: 256-step majority and the iteration-2 search (Arbor); the GPUs were held by another lab user from Saturday 17:02 IST, and no number is reported for them. Results freeze: Sun 22 Nov 18:00 IST. [pending]
