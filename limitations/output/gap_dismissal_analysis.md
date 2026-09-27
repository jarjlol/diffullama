# Why we decline all twelve generated research gaps

**Anchor paper.** Gong et al., *Scaling Diffusion Language Models via Adaptation from
Autoregressive Models*, ICLR 2025.

**Where the twelve came from.** A faithful implementation of Al Azher, Guo & Alhoori,
*Multi-Agent LLMs for Generating Research Limitations*
([arXiv:2601.11578](https://arxiv.org/abs/2601.11578)) — one of the limitation-generation
frameworks provided for this task. Four worker agents (Extractor, Analyzer, Reviewer,
Citation) over the anchor's full text plus a retrieval-augmented corpus, then a Judge, a
Self-Feedback loop and a Master consolidation. 33 raw limitations from four agents,
consolidated to twelve. Pipeline and methodology: [`REPORT.md`](../REPORT.md). The twelve
themselves: [`limitations_and_research_problem.md`](limitations_and_research_problem.md).

**What this document is.** We are declining all twelve. This is the argument for that,
scored against an explicitly anchored rubric and mapped onto the literature review this
project generated in the prior assignment using QUAL-SG.

---

## The rubric

| Dimension | Anchoring |
|---|---|
| **Liveness** | Is the gap still open? 0-3 closed or preempted | 4-6 crowded, actively being closed | 7-8 open, adjacent work exists | 9-10 verified untouched |
| **Feasibility** | Executable on 1x96GB contended in ~11 weeks? 0-3 needs adaptation training or inaccessible data | 4-6 possible but tight | 7-8 comfortable | 9-10 trivial |
| **Claim strength** | If it succeeds, how large is the result? 0-3 a footnote or one sentence | 4-6 a section or appendix | 7-8 a paper | 9-10 field-relevant |
| **Anchor specificity** | Is it about DiffuLLaMA? 0-3 wraps any model, anchor interchangeable | 4-6 partly | 7-8 mostly | 9-10 only answerable with this anchor |

**Bar:** A direction must reach 70/100 AND score >=4 on every dimension. None of the twelve does.

---

## Scorecard

| Gap | Live | Feas | Claim | Spec | Total | In our review | Reason declined |
|---|---|---|---|---|---|---|---|
| L1 | 7 | 1 | 8 | 10 | **65.0** | 21/48 | infeasible |
| L2 | 5 | 1 | 3 | 8 | **42.5** | 31/48 | infeasible + weak claim |
| L3 | 7 | 2 | 4 | 8 | **52.5** | 26/48 | infeasible |
| L4 | 8 | 0 | 9 | 9 | **65.0** | 25/48 | infeasible |
| L5 | 2 | 2 | 3 | 5 | **30.0** | 3/48 | closed by our own literature review |
| L6 | 5 | 9 | 5 | 6 | **62.5** | 18/48 | thin claim, novelty unverified |
| L7 | 4 | 1 | 2 | 6 | **32.5** | 33/48 | weak claim |
| L8 | 6 | 8 | 4 | 7 | **62.5** | 3/48 | thin claim |
| L9 | 2 | 8 | 3 | 4 | **42.5** | 35/48 | settled + most crowded area in our review |
| L10 | 3 | 1 | 6 | 6 | **40.0** | 27/48 | being actively closed + infeasible |
| L11 | 6 | 4 | 2 | 3 | **37.5** | 14/48 | weak claim, generic |
| L12 | 6 | 1 | 4 | 5 | **40.0** | 14/48 | blocked by its own premise |

---

## The structural finding

The twelve do not fail individually and for unrelated reasons. They partition into two
groups, and **nothing survives both filters**:

- **Training-bound, strong claims** — L1, L2, L3, L4, L5, L7, L10, L12. These include the best science in
  the set. L1 is a verified open gap: nobody has tested DiffuLLaMA's actual annealing at 7B for
  full-attention diffusion. L4 is the sharpest criticism available, since L1, L2 and L3 are each
  an instance of it. All are unreachable without adaptation training we cannot run.
- **Inference-reachable, weak claims** — L6, L8, L9, L11. Every gap we can actually execute is
  either already closed, already crowded, or too thin to carry a contribution.

**This is a property of the anchor paper, not a failure of the generation method.**
DiffuLLaMA's substantive weaknesses live in its *training recipe* — which components were kept,
which were dropped for implementation convenience, and which were never ablated at the scale
deployed. A frozen released checkpoint cannot answer any of those questions, because the
counterfactual was never trained. What remains observable at inference is a thin surface.

We therefore decline all twelve rather than select the least-bad one, and note explicitly that
L1 and L4 are declined **on resources, not on merit**. Presenting them as scientifically weak
would be indefensible; they are good questions we cannot afford to ask.

---

## Per-gap detail

### L1 — Attention-mask annealing is dropped at the deployment scale on a justification the paper's own ablation contradicts

**Score 65.0/100** — liveness 7, feasibility 1, claim 8, anchor-specificity 10. Fails the floor on: feasibility.

**Why we decline it.** Verified genuinely open (F-22: the only 7B annealing result uses modified annealing on block diffusion). Good science. Dismissed purely on resources: paired adaptation runs, ~160 GPU-hours for a scale ladder and roughly 1000 for a faithful Table 3 reproduction.

**Mapped onto our generated literature review.** 21 of 48 retrieved references engage this topic; the survey discusses it chiefly under *6. Recent Developments Since DiffuLLaMA (2025-2026): Scaling, Reasoning, and Applications*. Closest retrieved work:
  - *PIMNet: A Parallel, Iterative and Mimicking Network for Scene Text Recognition* (2021), LLM relevance 3.0, match 1.0
  - *TESS 2: A Large-Scale Generalist Diffusion Language Model* (2025), LLM relevance 4.5, match 0.776
  - *TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning* (2026), LLM relevance 5.0, match 0.625

### L2 — The shift operation is the larger effect and receives the smaller share of scrutiny

**Score 42.5/100** — liveness 5, feasibility 1, claim 3, anchor-specificity 8. Fails the floor on: feasibility, claim.

**Why we decline it.** 'No shift' is a configuration nobody ships -- Dream, DiffuCoder and NBDiff all keep it. Showing a variant nobody uses is worse is not a finding. Its interesting half (does the shift carry AR bias forward?) is not a training question at all.

**Mapped onto our generated literature review.** 31 of 48 retrieved references engage this topic; the survey discusses it chiefly under *6. Recent Developments Since DiffuLLaMA (2025-2026): Scaling, Reasoning, and Applications*. Closest retrieved work:
  - *Continual Pre-Training of Large Language Models: How to (re)warm your model?* (2023), LLM relevance 3.0, match 0.986
  - *Non-Autoregressive Text Generation with Pre-trained Language Models* (2021), LLM relevance 3.0, match 0.9
  - *Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning* (2025), LLM relevance 5.0, match 0.739

### L3 — The [MASK] token is an admitted workaround, applied inconsistently and confounded with model size

**Score 52.5/100** — liveness 7, feasibility 2, claim 4, anchor-specificity 8. Fails the floor on: feasibility.

**Why we decline it.** Breaking the confound needs a 2x2 training grid (reused vs new token, at two scales). The one CPU-only piece -- measuring the actual corpus frequency of tokens 811 and 10541 -- yields a single sentence, not a project.

**Mapped onto our generated literature review.** 26 of 48 retrieved references engage this topic; the survey discusses it chiefly under *4. Adapting and Continually Pre-training Language Models*. Closest retrieved work:
  - *Muse: Text-To-Image Generation via Masked Generative Transformers* (2023), LLM relevance 5.0, match 0.977
  - *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023), LLM relevance 3.0, match 0.894
  - *Flexible-length Text Infilling for Discrete Diffusion Models* (2025), LLM relevance 5.0, match 0.858

### L4 — The adaptation recipe is selected on a proxy task and never validated on adaptation training itself

**Score 65.0/100** — liveness 8, feasibility 0, claim 9, anchor-specificity 9. Fails the floor on: feasibility.

**Why we decline it.** The sharpest criticism in the set and the least executable. Validating a proxy requires running BOTH the proxy and the thing it proxies for -- strictly more expensive than L1. L1, L2 and L3 are each an instance of L4.

**Mapped onto our generated literature review.** 25 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *Continual Pre-training of Language Models* (2023), LLM relevance 3.0, match 0.991
  - *Dream-Coder 7B: An Open Diffusion Language Model for Code* (2025), LLM relevance 5.0, match 0.952
  - *Simple and Effective Masked Diffusion Language Models* (2024), LLM relevance 5.0, match 0.842

### L5 — Instruction tuning is deferred despite the paper's own evidence that its absence is causing failures

**Score 30.0/100** — liveness 2, feasibility 2, claim 3, anchor-specificity 5. Fails the floor on: liveness, feasibility, claim.

**Why we decline it.** TESS 2 -- reference [13] in our own 48-paper pool, LLM relevance graded -- already performs adaptation plus instruction tuning and reports both it and base-model choice are decisive. Our survey retrieved the paper that closes this gap.

**Mapped onto our generated literature review.** 3 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *TESS 2: A Large-Scale Generalist Diffusion Language Model* (2025), LLM relevance 4.5, match 1.0
  - *Simple and Effective Masked Diffusion Language Models* (2024), LLM relevance 5.0, match 0.345
  - *SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control* (2023), LLM relevance 5.0, match 0.303

### L6 — A large accuracy headroom is measured, diagnosed, and then left uninvestigated

**Score 62.5/100** — liveness 5, feasibility 9, claim 5, anchor-specificity 6. Fails the floor on: none.

**Why we decline it.** Best of the twelve and still short. Test-time selection is heavily worked in the AR literature, and DiffuCoder already reports pass@k analysis on adapted diffusion models. Without a matched LLaMA2 control the headline reduces to 'an oracle beats majority vote', which is true of every model ever sampled from.

**Mapped onto our generated literature review.** 18 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning* (2025), LLM relevance 5.0, match 1.0
  - *Efficient Continual Pre-training for Building Domain Specific Large Language Models* (2023), LLM relevance 3.0, match 0.463
  - *Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition* (2022), LLM relevance 3.0, match 0.447

### L7 — Every reported number is an unconverged lower bound, and the budget mismatch with the baseline is disclosed only in the appendix

**Score 32.5/100** — liveness 4, feasibility 1, claim 2, anchor-specificity 6. Fails the floor on: feasibility, claim.

**Why we decline it.** 'Is the model undertrained?' is answered by training it more. It is also the authors' own honest disclosure -- criticising a paper for stating it is undertrained is weaker ground than criticising one that hides it.

**Mapped onto our generated literature review.** 33 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization* (2022), LLM relevance 3.0, match 1.0
  - *A Cheaper and Better Diffusion Language Model with Soft-Masked Noise* (2023), LLM relevance 4.5, match 0.846
  - *Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition* (2022), LLM relevance 3.0, match 0.822

### L8 — The code and infilling claims are broader than the evaluation that supports them

**Score 62.5/100** — liveness 6, feasibility 8, claim 4, anchor-specificity 7. Fails the floor on: none.

**Why we decline it.** Running the comparison the authors admitted was unfair is honest and cheap, but it is a correction rather than a research programme. Note also that this item as generated contained a factual error about which checkpoint owns the 15.5 pass@1 figure; corrected before submission.

**Mapped onto our generated literature review.** 3 of 48 retrieved references engage this topic; the survey discusses it chiefly under *5. Non-Autoregressive Text Generation*. Closest retrieved work:
  - *Flexible-length Text Infilling for Discrete Diffusion Models* (2025), LLM relevance 5.0, match 1.0
  - *Dream-Coder 7B: An Open Diffusion Language Model for Code* (2025), LLM relevance 5.0, match 0.338
  - *Non-Autoregressive Text Generation with Pre-trained Language Models* (2021), LLM relevance 3.0, match 0.317

### L9 — The efficiency claim is not compute-normalised, and later measurement of adapted models contests it

**Score 42.5/100** — liveness 2, feasibility 8, claim 3, anchor-specificity 4. Fails the floor on: liveness, claim.

**Why we decline it.** Already examined and rejected by this team as Bet 5 ('wall-clock reporting is standard now'). Our own literature review engages this topic more than any other limitation -- 35 of 48 references -- confirming a crowded area.

**Mapped onto our generated literature review.** 35 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition* (2022), LLM relevance 3.0, match 0.909
  - *TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning* (2026), LLM relevance 5.0, match 0.898
  - *Self-Augmenting Retrieval for Diffusion Language Models* (2026), LLM relevance 5.0, match 0.844

### L10 — One adaptation route, two model families, and no comparison against cheaper or alternative routes

**Score 40.0/100** — liveness 3, feasibility 1, claim 6, anchor-specificity 6. Fails the floor on: liveness, feasibility.

**Why we decline it.** Our literature review surfaced three papers already pursuing the alternative routes this gap proposes: Don't Retrain Align (representation alignment), UNIFUSION (uniform-noise adaptation), and Enabling AR Models to Fill In Masked Tokens. Every alternative is itself a training procedure.

**Mapped onto our generated literature review.** 27 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *A Cheaper and Better Diffusion Language Model with Soft-Masked Noise* (2023), LLM relevance 4.5, match 0.945
  - *TESS 2: A Large-Scale Generalist Diffusion Language Model* (2025), LLM relevance 4.5, match 0.931
  - *DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models* (2023), LLM relevance 5.0, match 0.869

### L11 — Statistical and reproducibility rigor is insufficient to support the recipe decisions

**Score 37.5/100** — liveness 6, feasibility 4, claim 2, anchor-specificity 3. Fails the floor on: claim, specificity.

**Why we decline it.** 'They reported no seeds or confidence intervals' is a peer-review comment, not a research contribution, and it applies to a large fraction of the ML literature rather than to this paper specifically. The inference half is one good figure.

**Mapped onto our generated literature review.** 14 of 48 retrieved references engage this topic; the survey discusses it chiefly under *3. Text Diffusion Models*. Closest retrieved work:
  - *Continual Pre-Training of Large Language Models: How to (re)warm your model?* (2023), LLM relevance 3.0, match 1.0
  - *TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning* (2026), LLM relevance 5.0, match 0.663
  - *Diffusion-State Policy Optimization for Masked Diffusion Language Models* (2026), LLM relevance 5.0, match 0.613

### L12 — Data provenance, contamination, and release transparency are not addressed

**Score 40.0/100** — liveness 6, feasibility 1, claim 4, anchor-specificity 5. Fails the floor on: feasibility.

**Why we decline it.** A contamination audit needs the adaptation corpus, and this gap's own central complaint is that no data manifest was released. The obstacle is data access, not compute, so the constraint that blocks L1-L10 does not spare this one. A negative result would be nearly uninformative.

**Mapped onto our generated literature review.** 14 of 48 retrieved references engage this topic; the survey discusses it chiefly under *4. Adapting and Continually Pre-training Language Models*. Closest retrieved work:
  - *Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities* (2024), LLM relevance 3.0, match 0.999
  - *Continual Pre-Training of Large Language Models: How to (re)warm your model?* (2023), LLM relevance 3.0, match 0.804
  - *Efficient Continual Pre-training for Building Domain Specific Large Language Models* (2023), LLM relevance 3.0, match 0.744

---

## Method note

Reference matching uses the same hybrid BM25 + TF-IDF retrieval as the limitation pipeline
(`scripts/map_to_litreview.py`), run over the 48-reference pool and the section structure of the
generated survey. Engagement counts references scoring above a fixed similarity threshold.

**Low engagement is not itself grounds for dismissal** — an untouched gap may simply be novel, and
L5 and L8 both have low engagement while being declined for unrelated reasons. Coverage is one
input to the rubric, never the verdict.

**Correction recorded.** L8 as originally generated asserted that the anchor's HumanEval infilling
figure belonged to a separately trained Diffu-CodeLLaMA. This is false — Table 1 assigns
DiffuLLaMA 7B its own Code score of 15.5 pass@1, while Diffu-CodeLLaMA's 0.76 appears in Table 8,
a different finetune. The error originated in the Reviewer agent and was corrected before
submission. It is recorded rather than quietly fixed, because it is a real failure mode of the
generation pipeline.
