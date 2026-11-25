# Framing the negative preservation result

Status: working note, 25 Nov 2026. **Not a paper change.** The paper stays as it is unless a human asks for a change.

## 1. What the project actually tested

DiffuLLaMA adapts an AR model (GPT-2, LLaMA-2) into a diffusion LM by continual pre-training. The motivation is compute and capability reuse: a good DLM without training from scratch. The anchor also shows a large selection gap: on SATMath, 23.6% few-shot, 27.7% with self-consistency, but 57.7% hit@3. It does not try a verifier.

Our testable idea (ideation 1.1) was narrower than "is the conversion good":

> A frozen AR scorer reranks N samples from one DLM. The more the DLM preserves its AR base's representations (layer-wise CKA), the more of the first-sample-to-oracle gap that AR scorer should recover.

We did **not** train or adapt any model. Everything is inference on released checkpoints. We varied (a) the selector and (b) the number of **denoising steps at inference**, not training steps or training compute.

## 2. Fact-check of the current mental model

| Claim | Verdict | Correction |
|---|---|---|
| "We trained a diffusion model with fewer steps and let an AR model finish the training." | **Not what we did** | No training run. We sampled from frozen DLMs and reranked at inference. The AR model never trains or completes training; it only scores candidates. |
| "This failed horribly." | **Partly, but about a different thing** | Inference-time AR reranking loses to majority vote (23.6/21.0 points at 64 steps; 8.6/2.6 at 256). It does beat the first sample from K=32 upward, and the gap narrows at 256. It is a robust negative for the selector, not a training-compute result. |
| "We tested whether DLM quality correlates with how much it shares with the AR base." | **Too broad** | We tested whether CKA preservation predicts **AR-recoverable headroom** (gain over first sample from AR selection), not general DLM quality. We have five models, one dataset (GSM8K), and three are at the accuracy floor. |
| "The correlation is false." | **Too strong** | It is **not supported**, and the apparent positive rank order disappears after correcting the scoring. With n=4 adapted models, p >= 1/12, and a sign flip in the one strong point, this is not a disproof of preservation mattering in general. |
| "DiffuLLaMA's whole point is to make the DLM as close to the AR model as possible." | **No** | The point is compute-efficient adaptation and capability inheritance. Closeness/preservation is a hypothesized mechanism, not the objective. A useful DLM may need to diverge from its base. |
| "So AR-to-diffusion conversion may not be the best approach." | **Non sequitur** | Our data do not evaluate the conversion recipe or DLM quality. Low CKA can be normal or necessary. What we show is that CKA is not a useful predictor of AR verifier compatibility. |

The strongest defensible sentence is:

> In a controlled five-model GSM8K testbed, a frozen AR likelihood scorer did not beat majority vote, and CKA similarity to its AR base did not predict how much AR selection recovered. The apparent correlation was an artifact of scoring text after the final answer.

## 3. Why the result is still interesting (ranked framings)

### A. Representation similarity is not functional compatibility (strongest)
CKA-to-base is widely used as evidence that an adapted model "retained" its base. We test a concrete functional payoff of that retention and find none. The lesson generalises beyond DLMs: **internal similarity is not a proxy for whether a component can be reused as a verifier, adapter or head.** This is a measurement/negative-result contribution, not a claim that conversion is wrong.

### B. Self-consistency is a hard baseline; AR likelihood is not a free verifier
Majority vote reaches 73.3/73.0 on the final split, ties the agent's best rule, and beats every AR selector at 64 and 256 steps. A cheap AR model is not automatically a good judge of another model's samples. Practical guidance: evaluate against self-consistency before claiming a selection gain.

### C. Compute allocation: steps before selection at low budget
At K=16, majority adds 0.6/-0.4 points while raising K to 32 adds 9.8/19.2 to the first sample. Majority's advantage over the first sample shrinks from 32.0/24.6 at K=64 to 10.3/6.8 at K=256. If compute is scarce, more denoising is the better first purchase; selection is not a substitute.

### D. Evaluation audit: the effect can reverse under a scoring fix
Trailing text after `#### N` inflated/deflated AR scores; cutting it raised Dream's AR accuracy by 13 points and flipped its headroom from -9.7% to +17.7%. A sampler mismatch explained most of the iteration-1 "gap". This is a reusable methodological lesson: fix the scoring boundary and the operating point before interpreting a selector.

## 4. Claim ladder

**Safe to say now**
- In our testbed, AR-likelihood selection did not beat majority vote; the agent's best rule tied it.
- CKA preservation did not predict AR-recoverable headroom; the apparent order vanished after correcting scoring.
- At low step budgets, more denoising was a better use of compute than selection.
- The trailing-text and sampler confounds materially changed the iteration-1 conclusions.

**Needs more evidence**
- "CKA is generally useless for predicting verifier transfer." (One domain, five models.)
- "AR-to-diffusion conversion is suboptimal." (Not tested; would need quality-vs-preservation across many models.)
- "Preservation does not matter for DLM quality." (Not tested; only AR-recoverable selection.)
- "Training fewer diffusion steps and letting an AR model finish is a bad idea." (Not run.)

**Do not claim**
- "DiffuLLaMA's approach is wrong."
- "AR models can complete diffusion training."
- "We disproved the preservation hypothesis."
- Any training-compute saving.

## 5. How to sell it

- **The hook:** a plausible, widely used intuition fails a clean test. Not "we built a better selector", but "we found where a popular proxy breaks."
- **The unit of contribution:** a controlled testbed (five models, cached candidates, protected harness, first/majority/AR/oracle), a direct negative, and an audit that shows how the conclusion can flip.
- **The audience:** evaluation/negative-results workshops; course paper on honest experimentation; anyone using CKA as evidence of retention.
- **Title options:**
  - current: *Does Representation Preservation Predict AR Reranking Gains in Adapted Diffusion LMs?*
  - more assertive: *Retained Representations Do Not Make AR Verifiers Work*
  - methods-first: *CKA Is Not a Proxy for Verifier Transfer in Adapted Diffusion LMs*
- **Abstract shape:** intuition -> controlled test -> null -> confound that reverses it -> implication for how we evaluate adaptation.
- **Do not lead with** "DiffuLLaMA is inefficient" or "training with fewer steps failed"; neither is what we measured.

## 6. Difference from the current paper

The current paper already uses the safe framing: the title asks the question, the abstract says "unsupported" and "inconclusive", contribution 2 reports the negative, contribution 4 reports the audit, and the conclusion asks for a clean compute comparison and a second dataset. It does **not** claim conversion is wrong or that training compute was saved.

What the user's framing adds that the paper does not have:
- a training-compute story (not tested);
- a critique of the conversion recipe (not tested);
- a general claim that preservation is irrelevant (too broad).

What could be added later, with human approval, without changing results:
- one "broader implications" paragraph in the Discussion/Conclusion: CKA-to-base is not a functional proxy; evaluate downstream;
- a limitations sentence: five models, one dataset, three floor models, n=4;
- no title/abstract rewrite unless the team wants the more assertive framing.

## 7. Experiments that would strengthen the broader claim (post-submission)

1. More non-floor adapted models (e.g. DiffuCoder, a Dream-Instruct variant, a second Llama-based DLM) to give the CKA correlation real power.
2. A second dataset and a larger N; report per-question preservation with pre-registration.
3. Correlate CKA with **model quality** across many checkpoints, not just selection headroom.
4. A genuine training-compute experiment (fewer adaptation steps plus AR regularization) if the claim is about training, not inference. This is a new project, not a paper patch.
5. AR scorer families and calibration; test whether the failure is likelihood-specific or verifier-specific.

## 8. Language heuristics

- Prefer "did not predict", "no evidence", "unsupported" over "false" or "disproved".
- Always name the quantity: AR-recoverable headroom, not "quality".
- Separate **inference** compute (denoising steps, sampling, reranking) from **training** compute.
- Say "in our testbed / on GSM8K / five released models".
- CKA is correlational; do not write causal verbs.
- Distinguish first sample, majority vote, oracle and headroom every time.
- A negative result is a result; do not apologise for it, but do not inflate it.
