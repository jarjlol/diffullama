# GPU queue, ranked (T3, neel's agent, 21 Nov 2026)

Scores ⟨utility, quality, exploration⟩ (0–100) follow DeepScientist. **Surrogate reviewer:** neel's agent (Claude, this session, 2026-11-21), with `FINDINGS.md`, the red lines and the constraints in its context. This **deviates** from T3's "one surrogate LLM" call: the scoring was done in context, not by a separate API call with a set temperature, so it is not independent of the author. As T3 step 4d asks, exploration is high for runs that test idea 1.1's own claim and low for engineering add-ons. GPU-hours are estimates from `ITERATION_1.md` I-11 (Dream 36 s/question at 64 steps) and the #4 plan; none is measured [estimate].

Constraints in context: paper results freeze Sun 22 Nov 18:00 IST; aaditya's worker is the only GPU operator; GPU 0 reported a device error on 21 Nov (#17); 2 Blackwell GPUs, shared.

| Rank (κ=1) | Rank (0.5/0.5/0.3) | ID | Experiment | Status 21 Nov | GPU-h | u | q | e | UCB κ=1 | UCB 0.5/0.5/0.3 | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | A1 | Step-budget arm: K=16/32 vs 64 steps, Dream+LLaDA, test 500, N=10 | generation mostly done (Dream K16/K32 500, LLaDA K16 500, K32 459; #12 status, 21 Nov 06:16 UTC); scoring not run | ~2-4 (scoring + remaining LLaDA K32) [estimate] | 90 | 75 | 85 | 250 | 108.0 | answers P1 part (b), the unanswered half of the research problem; tests idea 1.1's own claim (ITERATION_2_PROMPTS A1) |
| 2 | 2 | PQP | Per-question preservation test on Dream (CKA/cosine per question vs per-question rerank success, n=500) | queued, not run | ~1 (one forward pass per question, DLM + base) [estimate] | 85 | 70 | 95 | 250 | 106.0 | only test of the preservation->gain claim with n=500 instead of n=4; directly idea 1.1 (audit/task4 04-science-audit.md §2) |
| 3 | 3 | I22 | Re-score iteration-1 64-step candidates on text cut after #### N | queued, not run | ~1-2 (5 AR scorers x 2 models x 1000 candidates) [estimate] | 95 | 90 | 40 | 225 | 104.5 | every AR-selection sentence in the paper depends on it; correctness fix, not new knowledge (ITERATION_2.md I-22) |
| 4 | 4 | A2s | DLM self-score as a selector (masked-likelihood, k=8 draws) | queued, not run | ~2-3 [estimate] | 75 | 70 | 45 | 190 | 86.0 | obvious competitor a reader asks about; engineering of an existing idea (A2 step 2 / NEXT_ITERATION 2) |
| 5 | 5 | A4 | More headroom models (diffullama-gsm adapter, Dream-Instruct, DiffuCoder) | not started; needs downloads | ~11 per 7B model at 64 steps on the subset (audit/task4 05) [estimate] | 60 | 55 | 60 | 175 | 75.5 | each adds a CKA point, but n stays tiny; Instruct models make SOTA rows comparable (A4 / NEXT_ITERATION 4) |
| 6 | 6 | BMC | BMC best-of-N selector (re-mask 0.9, K=16, MiniLM cosine) | not queued | ~3-5 (16 forward passes per candidate, 2 models x 1000 candidates per split) [estimate] | 65 | 60 | 40 | 165 | 74.5 | published training-free SOTA selector; re-implementation, Instruct->Base transfer untested (N3 SOTA.md S-1 (2604.16565)) |
| 7 | 7 | DS2 | Second dataset (MATH-500 or SVAMP) | not started | ~10-20 (generation for 2 models) [estimate] | 55 | 60 | 50 | 165 | 72.5 | needed to promote any finding (FINDINGS.md rule 3); too large before the freeze (NEXT_ITERATION 7) |
| 8 | 8 | A3 | Arbor iteration 2 from tuned majority, with T1/T3/T4 memory | not started; GPU 0 reported a device error (#17, arjun) | ~2 (both GPUs ~1 h, vLLM) [estimate] | 60 | 50 | 55 | 165 | 71.5 | only route to a positive selector result; iteration 1 found ~0.296-0.298 practical ceiling for in-item rules (ITERATION_2_PROMPTS A3) |
| 9 | 9 | F256 | Majority vote at 256 steps on the 100 fidelity questions | queued, not run | ~6 for LLaDA alone (aaditya's #4 plan) [estimate] | 55 | 65 | 30 | 150 | 69.0 | separates sampler effect from selector effect; fidelity, not the claim (ITERATION_2.md Tier 2 / NEXT_ITERATION 6) |
| 10 | 10 | VRM | Off-the-shelf reward/verifier model selection (ORM/PRM) | not started; no public GSM8K ORM for Dream identified (N3) | ~2-4 [estimate] | 50 | 45 | 30 | 125 | 56.5 | SOTA comparison, but the strong published ORM is trained and not released (N3) (NEXT_ITERATION 3) |

Done, not ranked: F8, Dream 8-shot fidelity: done: 80.0% 8-shot vs 81.0% 4-shot, n=100 (#12 status).

Weights: DeepScientist App. C uses w_u = w_q = κ = 1 (primary order). Our 0.5/0.5/0.3 (T3 step 2) is the sensitivity check.
Both orders agree on the top three: A1, PQP, I22 (same set).
Recommendation: run the top three first. Their estimated GPU-hours are the smallest of the high-value items. Whatever has not finished by the freeze goes to the report as future work, not to the paper. Whether all three fit depends on GPU 0's state, which is unknown here [unverified].
