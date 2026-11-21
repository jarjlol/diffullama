# Experimentation / Results and Discussion

<!-- Owner: arjun (issue #12). Target length: ~1.25 pages. Delete this block when final. -->
<!-- Sources: codification/ITERATION_1.md (branch assignment/codification @ 69d76a8), §Results and §Published numbers, with the Spearman-count correction in #28 from codification/results/iter1/numbers/analyze_test500.txt; codification/ITERATION_2.md §Results 1 and I-21/I-22 (branch iter2/aaditya-worker @ 3bc19b6); fidelity counts from codification/results/iter2/aryan/fidelity_complete.json (branch iter2/aryan, observed 2026-11-21T02:57 IST); failure classes from codification/ANALYSIS_ITER2.md §Failure classes (branch iter2/neel @ 75195d2). Iteration-2 runs other than fidelity and N4 are still queued; no number for them appears here. Paper results freeze Sun 22 Nov 18:00 IST. Notation follows problem_formulation.md (#10). -->
<!-- DRAFT 2026-11-21. Iteration-1 numbers from ITERATION_1.md §Results; iteration-2 numbers from ITERATION_2.md §Results 1. AR-reranking conclusions are withheld until the cut re-score (I-22) lands. -->

**Setup.** We evaluate on GSM8K (Cobbe et al., 2021): 200 dev questions for selector development, and 300 held-out test questions for the final score. Each model $m$ draws $N = 10$ candidates per question, and every selector reuses the same candidates. The models are DiffuGPT-S, DiffuGPT-M and DiffuLLaMA (Gong et al., 2025), adapted from GPT-2 and LLaMA-2; Dream 7B (Ye et al., 2025), adapted from Qwen2.5-7B; and LLaDA (Nie et al., 2025), trained from scratch and used as the control with no AR base. All models run 64 denoising steps; DiffuGPT-S uses one shot and 192 new tokens because its 512-token context cannot hold the four-shot prompt (I-7). Scorers are GPT-2, Qwen2.5-0.5B and TinyLlama, plus Qwen2.5-7B for Dream and Llama-2-7B for DiffuLLaMA. Preservation $\rho_m$ is layer-averaged linear CKA (Kornblith et al., 2019) on up to 2,000 held-out 128-token WikiText-103 segments.

**Generation fidelity.** Greedy decoding at the published sampler (256 steps, temperature 0) scores 81.0% on Dream with our 4-shot prompt and 80.0% with the published 8-shot prompt ($n = 100$ each), and 73.0% on LLaDA ($n = 100$), against published 77.2 and 70.3. Iteration 1's first sample on the same questions scores 41.0% and 51.0%. Prompt format and answer extraction are therefore not what lowered iteration 1 for Dream and LLaDA (fidelity was not checked for the three floor models); the 64-step, temperature-0.7 sampler used to obtain diverse candidates is (I-21). How much is due to temperature and how much to step count is not yet separated [unverified].

**Table 1.** GSM8K accuracy (%) on the final split ($n = 300$). "AR best" is the best summed or per-token AR selector on the iteration-1 trimmed text, with its scorer in parentheses; those AR figures are confounded by I-22 and the cut re-score is pending. Headroom is $H_m$ for the agent's selector. Source: `codification/ITERATION_1.md` §Results.

| Model | Shots | First | Majority | AR best (scorer) | Agent selector | Oracle | Headroom |
|---|---|---|---|---|---|---|---|
| DiffuGPT-S | 1 | 1.7 | 2.3 | 2.0 (Llama-2-7B mean) | 2.3 | 14.7 | 5% |
| DiffuGPT-M | 4 | 0.7 | 1.3 | 2.3 (TinyLlama mean) | 1.3 | 8.7 | 8% |
| DiffuLLaMA | 4 | 0.3 | 2.0 | 2.3 (GPT-2 mean) | 2.0 | 9.7 | 18% |
| Dream 7B | 4 | 41.0 | **73.3** | 32.0 (Qwen2.5-7B sum) | 73.0 | 88.0 | 68% |
| LLaDA 8B | 4 | 48.7 | **73.0** | 52.0 (Qwen2.5-7B sum) | 73.0 | 88.3 | 61% |
| Macro | | | | | 30.3 | | |

**Selector results.** No selector beat majority vote on the final split. The agent's best rule, arithmetic-verification-weighted voting (it down-weights candidates whose parsed calculations do not re-check), is indistinguishable from majority: −0.3 points on Dream (95% CI [−2.0, +1.3]; 3 wins, 4 losses; $p = 1.00$) and 0.0 on LLaDA ([−1.7, +1.7]; 3/3; $p = 1.00$), though both beat the first sample by +32.0 and +24.3. Its dev gain on Dream (+1.5) did not replicate. The gate split ($n = 200$) agrees: +1.5 [−0.5, +4.0] on Dream and −1.5 [−3.5, 0.0] on LLaDA ($p = 0.38$ and $0.25$). The merge gate compared the rule with the seed, not with majority, so the merge is not evidence against self-consistency. On the three floor models (oracle 8.7–14.7%) all selectors agree on almost every question.

**AR reranking: not interpreted yet.** On the iteration-1 text the best AR scorer trailed majority (32.0% vs 73.3% on Dream; 52.0% vs 73.0% on LLaDA), even with each model's own 7B base as scorer. But 660 of 1,000 Dream candidates and 497 of 1,000 LLaDA candidates on the first 100 test questions had more than 20 characters after the `#### N` answer line, which the scorers included (I-22). The cut re-score is queued, so we withhold any claim that AR reranking loses until it lands. Likelihood tie-breaks and vote weights, tried on dev, also did not beat majority, but they used the same confounded scores.

**What worked.** Majority vote is the strong training-free selector on the two non-floor models: it recovers 61–69% of the first-sample-to-oracle gap, and no in-item signal we tried closed the remaining ~15 points on the iteration-1 scores. The recoverable gain is agreement, not likelihood, consistent with self-consistency on AR models (Wang et al., 2023). Where majority is wrong but a correct candidate exists (dev: 29 Dream and 26 LLaDA questions), the largest classes are a close second (13/29 and 7/26) and an arithmetic slip in the wrong plurality (10/29 and 11/26); the agent's verification-weighted rule targets the latter but did not move the vote (ANALYSIS_ITER2 §Failure classes).

**Preservation and gain.** The CKA values are fixed, but the headroom compared with them uses the iteration-1 AR scores and may change after the cut re-score (Table 2). The order is as predicted: DiffuGPT-S (0.890) has positive headroom and Dream (0.509) the most negative. Across the ten AR selectors, Spearman $\rho$ over the four adapted models is $+1.0$ for 3, $+0.8$ for 6 and $+0.4$ for 1. This is not evidence: with four adapted models the smallest attainable exact permutation $p$ is $2/24 = 0.083$; three points are floor models whose headroom moves by one or two questions out of 500; and Dream alone sets the order. LLaDA, the no-base control, also has negative headroom under most AR scorers on those scores, so the pattern is not specific to models with a base. All four CKA variants give the same rank correlation, so the trend is not a CKA-estimator artifact.

**Table 2.** Layer-averaged CKA $\rho_m$ (DLM vs. its AR base) and AR-recoverable headroom $H_m(s_\psi)$ on the 500-question test set. Headroom is for the Qwen2.5-7B scorer, summed and per token. Headroom uses the iteration-1 trimmed-text AR scores and may change under the I-22 re-score; the CKA values do not. LLaDA has no AR base and is a control. Descriptive only. Source: `codification/results/iter1/numbers/analyze_test500.txt`; the selector counts carry the dated correction in #28.

| Model | CKA $\rho_m$ | Headroom (sum) | Headroom (mean) |
|---|---|---|---|
| DiffuGPT-S | 0.890 | +5.3% | +5.3% |
| DiffuGPT-M | 0.866 | +2.5% | +10.0% |
| DiffuLLaMA | 0.748 | −2.2% | 0.0% |
| Dream 7B | 0.509 | −9.7% | −13.9% |
| LLaDA 8B (control) | — | +9.4% | −8.4% |

## Iteration 2

The only landed iteration-2 result is the fidelity check. Still queued: the I-22 cut re-score, the $K = 16$ and $32$ step-budget arm (LLaDA $K = 16$ had 65 of 500 questions at the 20 Nov interruption), 256-step majority on the 100 fidelity questions, and the DLM self-score. No numbers from those runs appear here; only runs finished by the Sun 22 Nov 18:00 IST freeze will be added, with the same paired statistics. [pending]
