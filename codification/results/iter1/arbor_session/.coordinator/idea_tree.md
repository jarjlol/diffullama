# Idea Tree

**Baseline**: 0.1% | **Trunk**: 0.3%

## ROOT: Goal: maximize `score` from `bash eval.sh dev`. Merges are gated on `bash eval.sh test` (a held-out gate split). Only edit solution.py. Never touch task.py, eval.py, eval.sh, baselines.py, research_config.yaml or anything under data/.

What the first trials showed (iteration 1, GSM8K, N = 10 cached candidates per question):
1. The idea as specified fails where it matters. On Dream (dev: first sample 37.5%, oracle 84.5%) picking the candidate with the lowest AR-scorer NLL is WORSE than the first sample: 23.5-36.0% depending on scorer, with summed NLL the worst because it prefers short answers. Majority vote over the extracted answers reaches 70.0%.
2. The correct answer is usually among the candidates (oracle 85-92% on Dream and LLaDA), so the job is choosing among answers, not finding new ones.
3. DiffuGPT-S/M and DiffuLLaMA are at the floor (first sample under 2%, oracle 7-13%). Gains there are a few questions; do not tune for them. The macro score is driven by Dream and LLaDA.
4. Features per candidate: ar.<scorer>.nll_sum and ar.<scorer>.n_tokens for gpt2, qwen0.5b, tinyllama, qwen7b (Qwen2.5-7B, Dream's own AR base) and llama2-7b (Llama-2-7B, DiffuLLaMA's own AR base). The 7B scorers are the strongest; the small ones are weak.

Direction: start from majority vote, which already beats every AR selector. Use AR likelihood only where it can add information: to break ties between answers with equal votes, or to weight votes (for example, score each distinct answer by the sum over its candidates of exp(-per-token NLL)). Prefer per-token NLL to summed NLL. Keep select() deterministic and simple; a rule you can explain beats a tuned one. Report dev and gate scores per model, not only the macro average. [DONE]

### 1: Direction: answer-level self-consistency — decide by consensus over the 10 extracted candidate answers (vote counting) instead of ranking candidates by AR likelihood. The brief's iteration-1 evidence: majority already beats every AR selector; the job is choosing among answers. [DONE] (score: 0.3%)

**Insight**: Plain majority vote (earliest-index tie-break, all-None -> 0) verified at B_dev macro 0.2940 (dream 0.7000, llada 0.7450, floors 0.000-0.015) — measured directly in both executors' comparison tables (nodes 2.1 and 3.1 each ran plain majority as their reference). Doubles the seed (0.148). The standalone pure-majority node was skipped at idea review, so no dedicated run; the score is established by the two independent executor evaluations of the identical core.

### 2: Direction: AR likelihood as a TIE-BREAKER on top of self-consistency — keep vote counting primary, but when multiple answers tie for the top vote count, use per-token NLL under qwen7b to pick among the tied answers. Strictly per-item (no cross-model state, no evaluation-order dependence). [PRUNED]

**Insight**: AR-likelihood tie-breaking is not useful here. When vote counts tie, using qwen7b per-token NLL to select among tied answers does not improve accuracy; it is roughly neutral to anti-predictive. The method helps only in a narrow subset where the base model is already strong, while elsewhere it flips correct answers as often as it fixes wrong ones. Plain majority vote remains safer. Actionable conclusion: do not use raw AR NLL as a self-consistency tie-breaker for these per-item, closed-model comparisons. Future tie-breakers should use signals better aligned with answer correctness, such as calibrated answer probability, semantic similarity, uncertainty, or external verifier evidence.
[Pruned: Verified negative on B_dev (node 2.1, 0.2900 < 0.294 majority core): qwen7b per-token NLL tie-breaking resolved 27 dream ties but was net -5 items (1 fix vs 6 breaks); llada net 0 (4/4). The anti-predictiveness of per-token NLL survives restriction to tied answers — NLL measures fluency/length artifacts, not correctness. No recovery path within the tie-break axis.]

#### 2.1: Mechanism: majority vote with qwen7b per-token NLL tie-break — count non-None answers; among answers TIED for the top count, pick the answer whose best candidate has the lowest per-token NLL (min of ar.qwen7b.nll_sum/ar.qwen7b.n_tokens over that answer's candidates); remaining ties/missing features -> earliest candidate index; all-None -> 0.
Hypothesis: ties are where majority loses most (23 dev dream tie items, 8 of 23 earliest-choice correct), and qwen7b per-token NLL may separate tied answers even though it is anti-predictive across all candidates.
Observable: dream-base and llada-base dev >= 0.7000/0.7450; coordinator diagnostic predicts ~neutral to slightly negative (qwen7b fixed only 1/14 dream, 4/4 llada majority-wrong ties) — report per-model deltas vs plain majority.
Conflicts: none — complements node 1; same decision-rule axis (vote), different tie-handling. [PRUNED] (score: 0.3%)

**Insight**: The qwen7b per-token NLL tie-break is anti-predictive even when restricted to majority-vote ties, flipping correct answers on dream-base and llada-base as often as it fixes them. It yields a small net loss versus plain majority, helping only where the base model is already strong and its wrong ties happen to be separable by NLL.

**Result**: Majority voting with qwen7b per-token NLL tie-breaking reached a dev macro score of 0.2900, slightly below plain majority at 0.2940. dream-base scored 0.6750, below the 0.7000 target, while llada-base hit the 0.7450 floor exactly.

**Branch**: `coordinator/n2-1-mechanism-majority-vote-with-qwe-deb33ae3`

### 3: Direction: AR likelihood as a VOTE WEIGHT (the brief's proposed mechanism) — score each distinct answer by summing exp(-per-token NLL) over its candidates. Run as a falsification test: dev diagnostics predict per-token NLL is anti-predictive in the decisive cases, so this should land at or below plain majority; either outcome is a verified, recorded result. [PRUNED]

**Insight**: Using AR per-token NLL as a vote weight did not improve answer selection and was often counterproductive. In decisive cases, NLL acted as an anti-predictor: it favored fluent but incorrect minority responses over correct majority answers. The signal appears to track surface-level fluency, length, and stylistic artifacts rather than factual correctness.

Actionable conclusion: do not use raw per-token likelihood as a confidence or vote-weighting mechanism for candidate answers. Plain majority voting is stronger and simpler in this setting. Future work should treat likelihood weighting as failed unless re-calibrated against correctness, or replace it with more discriminative signals such as calibrated confidence, answer-agreement metrics, or error-aware scoring that penalizes fluent-but-wrong candidates.
[Pruned: Verified negative on B_dev (node 3.1, 0.2820): likelihood-weighted voting (sum exp(-per-token NLL) per answer, own-base scorer) scored 0.650/0.730 vs 0.700/0.745 majority on dream/llada. In the decisive cases the correct minority has lower NLL than the plurality in only 2/14 dev-dream and 8/18 dev-llada — at/below chance, so weights systematically flip plausibility-correct majorities to fluent-wrong minorities. The brief's proposed mechanism is audited dead; per-token NLL captures fluency, not correctness.]

#### 3.1: Mechanism: likelihood-weighted voting (falsification) — score each distinct extracted answer by sum of exp(-per-token NLL) over its candidates; per-token NLL = ar.<scorer>.nll_sum/ar.<scorer>.n_tokens; scorer: dream-base -> qwen7b, diffullama -> llama2-7b (its own AR base), all other models -> qwen7b; pick highest-scoring answer; ties -> earliest index; all-None -> 0.
Hypothesis: per-token NLL carries no usable signal in the decisive cases (correct-minority lower NLL in 2/14 dev-dream, 8/18 dev-llada — at/below chance), so this lands below plain majority; node exists to record the audited negative (or a surprising positive).
Observable: B_dev macro below node 1.1's ~0.294 (dream ~0.65, llada ~0.71-0.73 predicted) -> prune with lesson; >= 0.295 -> retain.
Conflicts: none — tests the brief's proposed mechanism head-on. [PRUNED] (score: 0.3%)

**Insight**: Per-token NLL under the AR scorer is anti-predictive in decisive cases, often favoring fluent but wrong minority answers over correct majorities. NLL magnitude appears to capture fluency/length-style artifacts rather than correctness, so plain majority outperforms likelihood-weighted voting.

**Result**: Likelihood-weighted voting was implemented in solution.py and evaluated on B_dev, achieving a macro score of 0.2820 versus 0.2940 for plain majority, below the retention threshold and leading to pruning.

**Branch**: `coordinator/n3-1-mechanism-likelihood-weighted-vo-4bdb7eb4`

### 4: Direction: within-item arithmetic verification as a vote weight. Rejected again: cross-model consensus is out of scope for this task (one model's N samples per item; reading other models' answers is ensembling, unfair to the per-model baselines and order-dependent). Do not propose it again. Instead use the candidate text itself: parse every explicit calculation in a candidate's reasoning (patterns like "a op b = c", including GSM8K "<<a*b=c>>" annotations), re-evaluate it exactly, and record the fraction of correct calculations and whether the final answer equals the result of the last calculation. Weight each candidate's vote by a simple function of that (e.g. 1 if all checks pass, 0.5 otherwise), then majority over weighted votes; ties -> earliest index; all-None -> 0. Hypothesis: when majority is wrong, the wrong plurality often comes from samples with an arithmetic slip, which a deterministic checker can see and AR likelihood cannot. Observable: dream-base and llada-base dev >= majority (0.700 / 0.745); report per-model counts of majority-wrong items that the weighting fixes and breaks. [DONE]

**Insight**: Within-item arithmetic verification is the only remaining in-item signal with positive value on the coherent models (child 4.1 merged: verification-weighted voting, verified B_test 0.2960). All other axes audited dead: AR NLL as tie-break (2, pruned) and vote weight (3, pruned) are anti-predictive; cross-candidate result consistency and question-number derivability priors are anti-predictive in the decisive cases; extraction-commitment discounting (#### tiers) is sub-merge-threshold (simulated 0.2980 dev/test vs 0.29896 gate); floor models have no usable in-item signal (best selector 2-5/200 items, dev/test inconsistent). Cross-model consensus is out of scope per operator. Practical ceiling for in-item explainable rules: ~0.296-0.298 macro.

#### 4.1: Mechanism: verification-weighted voting — for each candidate (on its trimmed text: everything before a line matching /\n\s*(?:Question|Q)\s*:/), parse explicit calculations with the regex "<<\s*(num)\s*([+\-*/])\s*(num)\s*=\s*(num)\s*>>" and "(num)\s*([+\-*/x×])\s*(num)\s*=\s*(num)" (num = -?[\d,]+(\.\d+)?; 'x'/'×' treated as *; divisions by 0 skipped, unparsable pairs skipped); re-evaluate each exactly and flag candidates having >=1 checked calculation with at least one mismatch (or none at all) as unverified. Weight = 1.0 if all checked calculations pass (and >=1 was checked), else 0.5. Answer score = sum of weights of its candidates; select the highest; ties -> earliest candidate index; all answers None -> 0. Uniform across all 5 models; deterministic, no state.
Hypothesis: when the majority-vote answer is wrong, the wrong plurality comes from samples containing an arithmetic slip that a deterministic checker sees and AR likelihood cannot. Coordinator dev diagnostics: in the 29 dream majority-wrong cases the correct candidate has all-passing calcs 9x vs 4x for the plurality (and last-calc==answer 11 vs 8); expected dream dev ~0.715 (+3 items). LLaDA shows the OPPOSITE pattern (correct all-pass 4/26 vs plurality 8/26) so expect llada dev ~0.740 (-1 item); net macro ~0.296 > 0.294 majority.
Observable: B_dev macro >= 0.295 with dream-base ~0.710-0.720; per-model report incl. floor models. If dream < 0.705 the mechanism is dead -> prune.
Conflicts: none — new evidence axis (in-item arithmetic consistency); orthogonal to the pruned AR-NLL axes (2, 3). [MERGED] (score: 0.3%)

**Insight**: MERGED (verified B_test 0.2960, up from 0.157 baseline). Verification-weighted voting: majority vote over extracted answers, candidate weight 1.0 if all its explicit calculations re-evaluate correctly (>=1 checked) else 0.5; answer score = sum of weights; ties -> earliest. Dev 0.296 (dream 0.715 = +3 fixes/0 breaks, llada 0.740 = 1 fix/2 breaks). Test 0.296 (dream 0.725, llada 0.720, floors 0.005-0.02). In-item arithmetic consistency is a real, if modest, signal on the coherent models; floors have ~no calculations and are untouched.

**Result**: The implementation replaced raw-NLL selection with deterministic verification-weighted voting in solution.py. Dev evaluation achieved B_dev macro 0.296 with dream-base 0.715 and llada-base 0.740, meeting the stated observable threshold.

**Branch**: `coordinator/n4-1-mechanism-verification-weighted-6f5aa728`
