# Decision log

Chronological. Newest pending items first, then settled decisions oldest-to-newest.
**Do not re-litigate settled decisions** without new evidence — if you have new evidence, add a dated
entry rather than editing the old one.

---

# ⏳ PENDING — blocking

## P-1 — Which project direction? (blocks all of stages 2–6)

Three options, fully analysed. **Options B and C are different papers; you cannot do both.** They even
need different reproduction targets — B needs DiffuGPT *training* reproduced, C needs
DiffuLLaMA/LLaDA/Dream *inference*. There is no shared head start.

| Option | What | Verdict |
|---|---|---|
| **B** — adaptation audit | "What does AR→diffusion adaptation actually cost?" Three sub-questions: annealing dropped at 7B, mask-token reuse, vestigial any-order capability | **RECOMMENDED** |
| **C** — Structure-Guided ReMasking | Neel's design doc, reframed after CDC | Viable, highest effort, smallest claim |
| **E** — scientific infilling | Use DiffuLLaMA's infilling on a scientific task | **NARROW** — checked and rejected |

Full options analysis with evidence:
[`audit/DECISIONS.md`](https://github.com/jarjlol/diffullama/blob/audit/opus-review/audit/DECISIONS.md)
(⚠️ on the throwaway branch — read before it is deleted).

**Status:** awaiting team decision. No direction-specific work has been started, deliberately.

## P-2 — Confirm target venue with TTV
See `04-open-questions.md` Q-1. Agents4Science 2025 is over; no 2026 edition found.

## P-3 — Confirm Sharanga access with TTV
Specs verified and excellent. Not a dependency — local 192 GB suffices for Option B — but a strong accelerator.

## ~~P-4 — Add the second SOTA paper, or submit as-is?~~ ✅ DONE 2026-08-29
Implemented. See D-2026-08-29-d below.

---

# ✅ SETTLED

## D-2026-08-13-a — Anchor paper: DiffuLLaMA
**Decided:** DiffuLLaMA (ICLR 2025) as the anchor paper, chosen independently rather than from TTV's list
of seven pre-defined topics.
**Why:** met all constraints (NLP, GenAI, top venue, recent); public code; and the team preferred an
independent paper over the TCS-linked topics.
**Alternative considered:** seven TTV-provided topics traceable to Manasi Patwardhan's TCS group, which
would likely have given closer mentorship. Rejected in favour of independence.

## D-2026-08-13-b — Reject best-paper-award winners as anchor candidates
**Decided:** exclude flagship award-winning papers from anchor-paper search.
**Why:** award winners attract the heaviest follow-up work, making a genuinely open gap unlikely. A
50-paper survey across NLP/CV/multimodal produced 15 gap-verified candidates and 35 rejects, each reject
naming the specific paper that closed its gap.

## D-2026-08-18-a — Assignment Part 2 task
**Decided:** "Continuous Gap Liveness Verification (Automated Novelty Invalidation)" — repeatedly checking
that an already-formulated contribution has not been published by someone else.
**Why:** distinct from the generic examples TTV listed; adversarial rather than associative reasoning;
demonstrated on the team's own 50-paper anchor search.
**Status:** submitted as PDF.

## D-2026-08-24-a — SOTA method: QUAL-SG from SurveyGen
**Decided:** implement QUAL-SG ([2508.17647](https://arxiv.org/abs/2508.17647)) rather than the alternative.
**Why:** QUAL-SG is a *generation* methodology with an explicit two-phase retrieval+generation pipeline
mapping directly onto the assignment's structure; the alternative ([2412.13612](https://arxiv.org/abs/2412.13612))
is an evaluation harness, not a method to implement.
**Result:** 48 references retrieved (all verified real), ~3,800-word survey, F1 = 0.222 vs the anchor's own
§5 related work.

## D-2026-08-28-a — Do not claim novelty for structure-guided remasking
**Decided:** the claim "first to use program structure to select remasking positions" must be **deleted**.
**Why:** Aryan found **CDC** ([arXiv:2605.16829](https://arxiv.org/abs/2605.16829), 16 May 2026), whose
MDFI operator does exactly this — static analysis → program-graph node → AST/dataflow neighbourhood →
token spans → budget → remask. Verified by reading the paper in full. Three specific claims must go; see
`03-established-facts.md` F-12.
**Impact:** this is the highest-value single contribution to the project so far. It prevented a
desk-rejectable novelty claim.

## D-2026-08-28-b — LLM-judge extension to the litreview pipeline
**Decided:** replace the keyword-regex relevance scorer with real semantic relevance judgment; keep results
in separate `*_llm_judged` files rather than overwriting the submitted survey.
**Why:** the keyword scorer was the #1 limitation flagged in the project's own REPORT.md §8, and it let
off-topic papers (protein, RNA, tire design) into the reference pool. Kept separate because
`generated_survey.md` cites by position against the original `reference_brief.txt`; swapping it would
desync every citation marker.
**Result:** F1 0.222 → 0.272, 9/33 → 11/33 matched, zero off-topic papers. Non-monotonic caveat documented
(DiffusionBERT dropped out despite scoring 5/5 relevant).

## D-2026-08-29-a — Reject bets 1, 2, 5 (Aalhad's systems/evaluation directions)
**Decided:** do not pursue multi-GPU single-request decode, dual-mode serving, or honest cost accounting.
**Why, per bet:**
- *Bet 1:* premise factually false (AR models do split single requests via tensor parallelism; vLLM shipped
  decode context parallelism Aug 2026). Core claim published by dInfer at batch size 1; Sangam is already
  the scheduler paper.
- *Bet 2:* thesis published near-verbatim by FLARE ([2606.01774](https://arxiv.org/abs/2606.01774)) three
  months prior. Also DiffuLLaMA is full-attention non-block, so getting two modes from it means
  re-implementing SDAR/BD3-LM first.
- *Bet 5:* premise decayed — wall-clock reporting became standard in 2026; strongest form of the argument
  is already a theorem ([2502.09622](https://arxiv.org/abs/2502.09622)); energy measurement needs exclusive
  GPU access unavailable on a shared cluster.

## D-2026-08-29-b — Reject option E (scientific infilling)
**Decided:** do not pursue, but revisit if Option B becomes blocked.
**Why:** molecular/SMILES and protein/peptide diffusion are crowded with purpose-built models; the
scientific-text angle is partly absorbed by [2606.19475](https://arxiv.org/abs/2606.19475). No direct hit
on the exact combination, but the resulting claim would be thinner than Option B's.
**Caveat recorded:** this was two targeted searches, not a full novelty pass — two background agents died
on an API spend limit. The crowded-sub-area finding is solid; the no-direct-hit finding is weak.

## D-2026-08-29-d — Implement the second SOTA paper's evaluation framework
**Decided:** implement [2412.13612](https://arxiv.org/abs/2412.13612)'s three-task evaluation framework and
apply it to the QUAL-SG survey, rather than submitting generation-only.
**Why:** uses both papers the instructor provided; converts the assignment from generate-only to
generate-and-evaluate; costs no GPU and no direction decision.
**Result:** reference accuracy 1.000, hallucination rate 0.000 over 47 scored references. Task 2 vs Task 3
split is informative — Introduction wins on ROUGE-1 (0.357 vs 0.187), full survey wins on coverage (0.705
vs 0.304).
**Honest caveat recorded in REPORT.md §11.5:** Task 1 is close to tautological here, since our references
are retrieved rather than LLM-generated and are verified against the same databases they came from.

## D-2026-08-29-c — Keep the audit on a throwaway branch
**Decided:** all audit and verification work lives on `audit/opus-review`, to be deleted once P-1 is
settled; durable conclusions copied to `project-docs/`.
**Why:** keeps `main` clean and submission-ready while preserving the evidence trail during the decision
period.
