# Experimentation / Results and Discussion

<!-- Owner: arjun (issue #12). Target length: ~1.25 pages. Delete this block when final. -->
<!-- Sources: codification/ITERATION_1.md (branch assignment/codification @ 69d76a8), §Results and §Published numbers, with the Spearman-count correction in #28 from codification/results/iter1/numbers/analyze_test500.txt; codification/ITERATION_2.md §Results 1 and I-21/I-22 (branch iter2/aaditya-worker @ 3bc19b6); fidelity counts from codification/results/iter2/aryan/fidelity_complete.json; cut re-score and Dream self-score from codification/results/iter2/cpu/{cvt_dream_test,cvt_llada_test,self_dream_gate}.txt; step arm from codification/results/iter2/step_arm.md (Blackwell, 2026-11-21, worker commit 66697fd). Remaining iteration-2 runs are marked pending. Paper results freeze Sun 22 Nov 18:00 IST. Notation follows problem_formulation.md (#10). -->
<!-- DRAFT 2026-11-21. Iteration-1 numbers from ITERATION_1.md §Results; iteration-2 numbers from the result files named above. The AR conclusion now rests on the cut re-score. -->

**Setup.** We evaluate on GSM8K (Cobbe et al., 2021): 200 dev questions for selector development and 300 held-out test questions for the final score. Each model $m$ draws $N = 10$ candidates per question. The models are DiffuGPT-S, DiffuGPT-M and DiffuLLaMA (Gong et al., 2025), adapted from GPT-2 and LLaMA-2; Dream 7B (Ye et al., 2025), adapted from Qwen2.5-7B; and LLaDA (Nie et al., 2025), trained from scratch and used as the control. All run 64 denoising steps; DiffuGPT-S uses one shot and 192 new tokens because its 512-token context cannot hold the four-shot prompt (I-7). Scorers are GPT-2, Qwen2.5-0.5B and TinyLlama, plus Qwen2.5-7B for Dream and Llama-2-7B for DiffuLLaMA. Preservation $\rho_m$ is layer-averaged linear CKA (Kornblith et al., 2019) on up to 2,000 WikiText-103 segments.

**Generation fidelity.** Greedy decoding at 256 steps (temperature 0) scores 81.0% on Dream with our 4-shot prompt and 73.0% on LLaDA ($n = 100$ each), against published 77.2 and 70.3 and 41.0% and 51.0% for iteration 1's first sample. Prompt format and answer extraction are therefore not what lowered iteration 1; the 64-step, temperature-0.7 sampler is (I-21). At 256 steps the temperature-0.7 first sample is 7.0 points below greedy on Dream and 2.9 above on LLaDA (paired, $n = 185$ and $137$), so the step budget, not temperature, accounts for most of the gap.

**Table 1.** GSM8K accuracy (%) on the final split ($n = 300$); macro agent-selector mean 30.3. Headroom is $H_m$ for the agent's selector. Source: `ITERATION_1.md` §Results.

| Model | Shots | First | Majority | Agent selector | Oracle | Headroom |
|---|---|---|---|---|---|---|
| DiffuGPT-S | 1 | 1.7 | 2.3 | 2.3 | 14.7 | 5% |
| DiffuGPT-M | 4 | 0.7 | 1.3 | 1.3 | 8.7 | 8% |
| DiffuLLaMA | 4 | 0.3 | 2.0 | 2.0 | 9.7 | 18% |
| Dream 7B | 4 | 41.0 | **73.3** | 73.0 | 88.0 | 68% |
| LLaDA 8B | 4 | 48.7 | **73.0** | 73.0 | 88.3 | 61% |

**Selector results.** No selector beat majority on the final split. The agent's best rule, arithmetic-verification-weighted voting, is indistinguishable from it: −0.3 points on Dream (95% CI [−2.0, +1.3]; $p = 1.00$) and 0.0 on LLaDA ([−1.7, +1.7]; $p = 1.00$), though both beat the first sample by +32.0 and +24.3; its dev gain on Dream did not replicate. Majority recovers 61–69% of the first-sample-to-oracle gap through agreement, not likelihood (Wang et al., 2023).

**AR reranking.** With text cut at the final answer line (I-22), the best AR selector reaches 48.8% on Dream and 52.2% on LLaDA, still 23.6 and 21.0 points below majority (both $p < 0.001$; Table 2, $K = 64$). Cutting raised Dream's Qwen2.5-7B summed score by 13.0 points (95% CI [9.0, 17.0]) and left LLaDA's unchanged.

**Step budget.** Table 2 varies the denoising budget ($N = 10$, same temperature). Majority goes 3.2→23.4→72.4% on Dream and 3.2→51.0→73.2% on LLaDA for $K = 16, 32, 64$, and at $K = 64$ it adds 32.0 and 24.6 points over the first sample. At $K = 256$ (random subsample, $n = 185$ and $192$, stopped by the clock) it adds only 10.3 and 6.8, and the best AR selector is 8.6 points below majority on Dream (95% CI $[-14.1, -3.2]$, $p = 0.002$) but 2.6 on LLaDA ($[-6.8, +1.6]$, $p = 0.33$): the value of selection shrinks as the budget grows. Against one greedy sample majority is $+3.2$ ($p = 0.24$) and $+8.0$ points ($n = 137$, $p = 0.007$). $T = 0.7$ at 256 steps is not like-for-like with the published greedy numbers, and generation times come from shared GPUs, so no clean cost comparison exists [unverified as a cost comparison].

**Table 2.** Accuracy (%) and generation time per question ($N = 10$) by denoising budget; $K \le 64$ on 500 GSM8K test questions, $K = 256$ on a random subsample ($n = 185$ Dream, $192$ LLaDA; 23 and 41 rows from an Ada GPU; s/q from Blackwell rows only). "Best AR" is the best AR selector at that budget: Qwen2.5-7B cut sum, except GPT-2 mean (Dream, $K = 16$) and Qwen2.5-0.5B cut sum (LLaDA, $K = 16$). $K = 64$ on the $K = 256$ subsample: first 42.2 / 46.4, majority 74.1 / 74.5. Sources: `step_arm.md`, `cvt_*_test.txt`, `k256_results.md`.

| Model | $K$ | First | Majority | Oracle | Best AR | s/q |
|---|---|---|---|---|---|---|
| Dream 7B | 16 | 2.6 | 3.2 | 17.4 | 2.4 | 9.5 |
| Dream 7B | 32 | 12.4 | 23.4 | 54.0 | 13.8 | 19.3 |
| Dream 7B | 64 | 40.4 | **72.4** | 87.8 | 48.8 | 36.8 |
| Dream 7B | 256 | 74.1 | **84.3** | 93.5 | 75.7 | 151.1 |
| LLaDA 8B | 16 | 3.6 | 3.2 | 24.6 | 3.6 | 13.7 |
| LLaDA 8B | 32 | 22.8 | 51.0 | 76.0 | 28.6 | 27.4 |
| LLaDA 8B | 64 | 48.6 | **73.2** | 89.2 | 52.2 | 55.0 |
| LLaDA 8B | 256 | 72.4 | **79.2** | 89.6 | 76.6 | 216.6 |

**Preservation and gain.** Table 3 gives the iteration-1 correlation, on trimmed-text headroom. It does not carry over to the corrected scores: Dream's Qwen2.5-7B-sum headroom flips from −9.7% to +17.7% after cutting, and the three floor models have no cut scores, so the four-model Spearman $\rho$ cannot be recomputed. On trimmed scores the order was as predicted ($\rho$ $+1.0$ for 3 selectors, $+0.8$ for 6, $+0.4$ for 1; exact $p \ge 0.083$), but Dream set it and its corrected sign is opposite, so we do not claim preservation is supported. A per-question test (500 questions, exploratory) does not support it either: AUC 0.469 (permutation $p = 0.32$; OR 0.94 per SD [0.75, 1.17]), and no layer is significant after Benjamini–Hochberg correction.

**Table 3.** Layer-averaged CKA $\rho_m$ (DLM vs. its AR base) and AR-recoverable headroom $H_m(s_\psi)$ on the 500-question test set (Qwen2.5-7B scorer, summed and per token, iteration-1 trimmed-text scores; on cut scores Dream is +17.7% and LLaDA +8.9%, and the floor models have none). LLaDA has no AR base and is a control. Descriptive only. Source: `analyze_test500.txt`.

| Model | CKA $\rho_m$ | Headroom (sum) | Headroom (mean) |
|---|---|---|---|
| DiffuGPT-S | 0.890 | +5.3% | +5.3% |
| DiffuGPT-M | 0.866 | +2.5% | +10.0% |
| DiffuLLaMA | 0.748 | −2.2% | 0.0% |
| Dream 7B | 0.509 | −9.7% | −13.9% |
| LLaDA 8B (control) | — | +9.4% | −8.4% |

**Iteration 2.** The 256-step check was pre-registered and run once; its size was set by the clock, not by a result (two Dream restarts after CUDA launch timeouts, no rows lost). The iteration-2 search (Arbor) is not finished: the GPUs were held by another lab user from Saturday 17:02 IST, and no number is reported for it.
