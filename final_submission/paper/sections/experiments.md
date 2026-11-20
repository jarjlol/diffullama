# Experimentation / Results and Discussion

<!-- Owner: arjun (issue #12). Target length in the final paper: ~1.25 pages. Delete this comment block when the section is final. -->
<!-- Sources: codification/ITERATION_1.md (branch assignment/codification @ 69d76a8), §Results and §Published numbers; codification/results/iter1/. Iteration 2 not run at draft time: results freeze Sun 22 Nov 18:00 IST, scope pending human-only decision #7. Notation follows problem_formulation.md (#10): m, b(m), N, y_{j,i}, s, s_first, s_maj, s_psi, H_m, rho_m. -->
<!-- DRAFT 2026-11-21. Every number below is copied from ITERATION_1.md §Results; anything not checked there is tagged [unverified]. -->

**Setup.** We evaluate on GSM8K (Cobbe et al., 2021). Selectors are developed on 200 dev questions and scored on 300 held-out test questions that the agent never saw. Each model $m$ draws $N = 10$ candidates per question at non-zero temperature, and every selector reuses the same candidates. Five released DLMs are covered: DiffuGPT-S, DiffuGPT-M and DiffuLLaMA (Gong et al., 2025), adapted from GPT-2 and LLaMA-2; Dream 7B (Ye et al., 2025); and LLaDA (Nie et al., 2025), trained from scratch and serving as the control. All models run 64 denoising steps; DiffuGPT-S uses one shot and 192 new tokens because its 512-token context cannot hold the four-shot prompt (spec gap I-7). Scorers are GPT-2, Qwen2.5-0.5B and TinyLlama, plus each adapted model's own 7B AR base where one exists (Qwen2.5-7B for Dream, Llama-2-7B for DiffuLLaMA). Preservation $\rho_m$ is layer-averaged linear CKA (Kornblith et al., 2019) on up to 2,000 held-out 128-token WikiText-103 segments. Table 1 gives accuracy on the final split; Table 2 gives the paired comparison.

**Table 1.** GSM8K accuracy (%) on the final split ($n = 300$). "AR best" is the best summed or per-token AR selector, with its scorer in parentheses; headroom is $H_m$ for the agent's selector. Source: `codification/ITERATION_1.md` §Results.

| Model | Shots | First | Majority | AR best (scorer) | Agent selector | Oracle | Headroom |
|---|---|---|---|---|---|---|---|
| DiffuGPT-S | 1 | 1.7 | 2.3 | 2.0 (Llama-2-7B mean) | 2.3 | 14.7 | 5% |
| DiffuGPT-M | 4 | 0.7 | 1.3 | 2.3 (TinyLlama mean) | 1.3 | 8.7 | 8% |
| DiffuLLaMA | 4 | 0.3 | 2.0 | 2.3 (GPT-2 mean) | 2.0 | 9.7 | 18% |
| Dream 7B | 4 | 41.0 | **73.3** | 32.0 (Qwen2.5-7B sum) | 73.0 | 88.0 | 68% |
| LLaDA 8B | 4 | 48.7 | **73.0** | 52.0 (Qwen2.5-7B sum) | 73.0 | 88.3 | 61% |
| Macro | | | | | 30.3 | | |

**Table 2.** Paired difference in accuracy (percentage points) between the agent's selector and majority vote, with a 95% bootstrap CI over questions, discordant wins/losses, and the exact McNemar $p$; last column: the same selector minus the first sample. Only Dream and LLaDA have non-trivial accuracy. Source: `codification/ITERATION_1.md` §Results.

| Split | Model | vs. majority | 95% CI | wins/losses | McNemar $p$ | vs. first |
|---|---|---|---|---|---|---|
| final ($n=300$) | Dream | −0.3 | [−2.0, +1.3] | 3/4 | 1.00 | +32.0 |
| final ($n=300$) | LLaDA | 0.0 | [−1.7, +1.7] | 3/3 | 1.00 | +24.3 |
| gate ($n=200$) | Dream | +1.5 | [−0.5, +4.0] | 4/1 | 0.38 | +33.0 |
| gate ($n=200$) | LLaDA | −1.5 | [−3.5, 0.0] | 0/3 | 0.25 | +23.5 |

**What did not work.** The idea as specified is refuted on this split. Picking the lowest-AR-NLL candidate is worse than the first sample on Dream (best 32.0% vs 41.0%) and only marginally better on LLaDA (52.0% vs 48.7%), even with each model's own 7B AR base as scorer; majority vote beats every AR selector by at least 21 points on both. Using likelihood only to break vote ties or to weight votes also lost to plain majority on dev. On the three floor models, whose oracle accuracy is 9–15%, all selectors agree on almost every question, so the diffusion-family checkpoints are too weak few-shot for selection to matter. One confound: iteration 1 scored text after the final `#### N` answer line, which distorts AR scores, especially Dream's (issue I-22). These losses are provisional until the queued re-score on text cut at that line finishes; the refutation is claimed only if it survives.

**The agent's search.** The codification agent ran 7 of 12 planned cycles (68 min), executed four selector ideas and merged one: arithmetic-verification-weighted voting, which down-weights candidates whose parsed calculations do not re-check. On the final split it is indistinguishable from majority (−0.3 on Dream, 0.0 on LLaDA; both CIs span zero). Its dev gain on Dream (+1.5) did not replicate (3 wins, 4 losses; $p = 1.00$). The merge gate compared it with the seed (summed GPT-2 NLL), not with majority vote, so the merge is not evidence that it beats self-consistency.

**What worked.** Majority vote is the strong training-free selector on the two non-floor models: it recovers 61–69% of the first-sample-to-oracle gap, and no in-item signal tried closed the remaining ~15 points. The recoverable gain is agreement, not likelihood, consistent with self-consistency on AR models (Wang et al., 2023).

**Preservation and gain.** The rank order is the predicted one: the highest-CKA model, DiffuGPT-S (0.890), has positive AR-recoverable headroom, and Dream, the lowest (0.509), has the most negative (Table 3). Spearman $\rho$ is $+1.0$ across four models and $+0.8$ across five. This is not evidence. With four adapted models the smallest attainable exact permutation $p$ is $2/24 = 0.083$; three points are floor models whose headroom moves by one or two questions out of 500; and Dream alone sets the order. LLaDA, the no-base control, also loses under most AR scorers, so low preservation does not explain Dream's loss. All four CKA variants give the same rank correlation, so the trend is not a CKA-estimator artifact.

**Table 3.** Layer-averaged CKA $\rho_m$ (DLM vs. its AR base) and AR-recoverable headroom $H_m(s_\psi)$ on the 500-question test set, for the best summed and per-token selector. LLaDA has no AR base and is a control. Descriptive only. Source: `codification/ITERATION_1.md` §Results.

| Model | CKA $\rho_m$ | Headroom (sum) | Headroom (mean) |
|---|---|---|---|
| DiffuGPT-S | 0.890 | +5.3% | +5.3% |
| DiffuGPT-M | 0.866 | +2.5% | +10.0% |
| DiffuLLaMA | 0.748 | −2.2% | 0.0% |
| Dream 7B | 0.509 | −9.7% | −13.9% |
| LLaDA 8B (control) | — | +9.4% | −8.4% |

## Iteration 2

Out of this draft. Decision #7 is human-only and was still open on 21 Nov; the results freeze is Sun 22 Nov 18:00 IST. When finished runs land, this section will add the I-22 re-score and whichever iteration-2 results are complete, with the same paired statistics. No iteration-2 number is reported here. [pending #7]
