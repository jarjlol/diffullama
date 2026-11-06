# Borrowing from the other three reference systems

The Task 4 brief gives four reference papers "only as references ... to borrow some of the ideas from".
We built on **Arbor** because it is the only one whose public code implements its paper
(`REFERENCE_SYSTEMS.md`, D-2026-11-01-a). The other three papers still describe mechanisms we can
re-implement ourselves around Arbor. This file says which ones, why, and how. It is the design for
iteration 2. The prompts that build it are in `ITERATION_2_PROMPTS.md`.

**Evidence.** Every mechanism below was read on 2026-11-06 from the paper's full text (arXiv HTML), at
the section given. None of their code is used: DeepScientist's and EvoScientist's repositories no
longer contain these mechanisms, and AutoSOTA's system was never released (`REFERENCE_SYSTEMS.md`).
What we build is our own implementation of the idea, and we say so in the write-up.

## 1. What each paper offers

### DeepScientist (arXiv:2509.26603)

- **Findings Memory with three tiers** (§3.2). Each record is a finding. It starts as an
  *Idea Finding* (an unverified hypothesis), becomes an *Implement Finding* when it is selected and
  run, and becomes a *Progress Finding* only if it beats the baseline. Failed records stay in memory
  "to inform subsequent explorations".
- **Cheap surrogate before expensive runs** (§3.2, Eq. 2). An LLM reviewer, given the whole Findings
  Memory, scores each new hypothesis with a vector ⟨utility, quality, exploration⟩ (integers 0-100).
  A UCB rule picks what to run: `w_u·v_u + w_q·v_q + κ·v_e`.
- **Promotion needs more than one win** (§3.2, "Analyze & Report"). A Progress Finding triggers deeper
  analysis: ablations and evaluation on new datasets.
- **Why it matters** (§4.3, App. B). Of over 5,000 ideas, about 1,100 were run and 21 made progress.
  About 60% of failed runs failed on implementation errors, not on the idea. Their advice: filter
  before running, and invest in verification.

### EvoScientist (arXiv:2603.08127)

- **Two persistent memories** (§3.2-3.4). An *ideation memory* holds promising and failed research
  directions; an *experimentation memory* holds reusable execution strategies (data processing,
  training). Agents retrieve the most similar items before they generate an idea or code.
- **Elo tournament for idea selection** (§3.3, Eq. 3). Candidate ideas are ranked by pairwise
  comparisons on novelty, feasibility, relevance and clarity, "because it relies on pairwise
  comparisons and can produce a stable ranking under noisy judgments without requiring calibrated
  absolute scores".
- **Experiment tree in four stages** (§3.4): initial implementation, hyperparameter tuning, proposed
  method, ablation. Failed runs are diagnosed from logs and fixed.
- **Evolution after each task** (§3.5). An evolution manager writes what worked (idea direction
  evolution), what failed against the baselines (idea validation evolution) and reusable execution
  strategies (experiment strategy evolution) back into the memories.
- **Ablation** (§5.4). Removing idea validation evolution hurt feasibility more than removing idea
  direction evolution did (losing 63.3% vs 50.0% of feasibility comparisons); removing all idea
  evolution hurt novelty (80.0%) and feasibility (83.3%) most.

### AutoSOTA (arXiv:2604.05550)

- **Red Line System** (§3.9, AgentSupervisor). Non-negotiable rules checked during optimisation. R1
  metric parameters fixed; R2 evaluation script untouched; R3 outputs must come from real inference;
  R4 no gain on one metric bought by a loss on another; R5 split integrity, no test data in any form;
  R6 no dataset changes; R7 paper-specific constraints written down and followed. (The text says
  "six" constraints and lists seven.) They report that such invalid "improvements" happened "more
  frequently than anticipated".
- **Typed, audited Idea Library** (§3.7, §3.8.1 Phase 2). At least ten ideas, each typed PARAM / CODE /
  ALGO with a risk level, mixing low-risk tuning and structural changes. A red-line audit marks each
  CLEARED or REJECTED before anything runs.
- **Leap Path and Honeymoon Period** (§3.8.1 Phase 3). If the last three iterations were all PARAM
  tweaks, the next one must be a structural idea. A structural change gets up to five follow-up
  iterations before rollback.
- **Measured baseline first** (§3.8.1 Phase 0). Run the baseline yourself and anchor every comparison
  to the measured value, because reported results "often diverge from reproducible values".
- **Tree-structured rubric** (§3.3, AgentObjective). The goal is broken into weighted Pass/Fail
  sub-tasks (e.g. Result Match, Methodology Implementation, Environment Configuration), so a failure
  can be located on a tree path.
- **Retrieval before repair** (§3.6, AgentFix) with a failure memory of known fixes (e.g. no internet
  in the container, use mirrors for HuggingFace), and a **monitor** that stops dead-end debugging
  loops (§3.5).

## 2. What iteration 1 shows we need

| Iteration-1 problem (evidence) | Borrowed fix |
|---|---|
| Arbor proposed cross-model consensus twice; a human had to reject it each time (`ITERATION_1.md` §Arbor) | AutoSOTA red lines, given to Arbor up front and used as the review checklist |
| Arbor's merge gate compared against the seed, not majority vote (`ITERATION_1.md` §Results, finding 2) | AutoSOTA "measured baseline first": the seed and the bar are majority vote, measured on our own candidates |
| The ideation scores were saturated (78 of 100 ratings were 5), so the top-5 order is weak (`ideation/REPORT.md` §7) | EvoScientist Elo tournament: re-rank P1's ideas by pairwise comparison with a different-family judge |
| Most fixes were human code edits; lessons lived only in notes (`NEXT_ITERATION.md`) | DeepScientist Findings Memory + EvoScientist memories, fed to Arbor and to every teammate's agent |
| One shared GPU; we must choose which costly run to do next (`project-docs/01-project-brief.md` compute) | DeepScientist surrogate + UCB, to rank GPU experiments before spending hours on them |
| 4.1's dev gain did not hold on the final split (`ITERATION_1.md` §Results) | DeepScientist promotion rule: a win counts only after the final split, an ablation and a second dataset |
| No ablation of the merged rule; no stage structure | EvoScientist four-stage experiment tree |
| Several spec cuts were not logged; the write-up had stale parts (`ITERATION_2_PROMPTS.md` review) | AutoSOTA weighted Pass/Fail rubric, applied to the write-up before submission |

## 3. The combined loop for iteration 2

```
 BEFORE                         DURING (Arbor)                    AFTER
 ------                         --------------                    -----
 red lines (AutoSOTA R1-R7) --> Arbor coordinator + executors --> validation analysis (EvoScientist IVE)
 typed idea library, audited    seed = majority vote (measured)   Findings Memory update (DeepScientist tiers)
 Elo re-rank of ideas (Evo)     review replies use red lines      promotion: final split once + ablation
 surrogate + UCB picks GPU      leap rule: 3 PARAM -> structural    + second dataset (DeepScientist)
   jobs (DeepScientist)         retrieval before repair           rubric check of the write-up (AutoSOTA)
 findings + strategies memory   (experimentation memory)
```

What stays Arbor's: the coordinator, the executors in git worktrees, the idea tree, and the
held-out merge gate. What we add are files and review rules around it, not a new system:

| Piece | File | Built by prompt |
|---|---|---|
| Red lines for our task | `codification/memory/RED_LINES.md` | T1 |
| Typed, audited idea library | `codification/memory/IDEA_LIBRARY.md` | T1 |
| Elo re-ranking of P1 ideas | `codification/memory/ELO_RANKING.md` | T2 |
| Findings Memory (ideation memory) | `codification/memory/FINDINGS.jsonl` + `FINDINGS.md` | T3 |
| Surrogate + UCB ranking of GPU jobs | `codification/memory/GPU_QUEUE.md` | T3 |
| Experimentation memory | `codification/memory/STRATEGIES.md` | T4 |
| Rubric for the write-up | `codification/memory/RUBRIC.md` | T5 |
| Arbor plugin: red lines as prompt injections, protected paths, convergence (added 2026-11-06, §6) | `codification/rerank_task/plugins/rerank_guard.yaml` | T1 |
| Curated Arbor experience, without the banned cross-model "leverage" (added 2026-11-06) | `codification/memory/EXPERIENCE_CURATED.md` | T3 |
| Research report for ideation (added 2026-11-06) | `codification/memory/RESEARCH_REPORT.md` | N3 |
| Majority-vote failure taxonomy and tuned-majority bar (added 2026-11-06) | `codification/ANALYSIS_ITER2.md` | N4 |
| Arbor instruction for iteration 2 | built from the files above | A3 |

## 4. Our red lines (draft, to be finished in T1)

These follow AutoSOTA's R1-R7, mapped to our task:

1. **R1 metric.** Accuracy per model and the macro score stay as defined in `task.py` and `eval.py`.
2. **R2 evaluation.** Do not edit `task.py`, `eval.py`, `eval.sh` or `baselines.py` (already protected).
3. **R3 real outputs.** `select()` returns an index into this item's own 10 candidates; it never
   writes an answer.
4. **R4 no trade-off.** A selector that gains on Dream but loses on LLaDA (or the reverse) by more than
   noise is not an improvement. Report every model.
5. **R5 splits.** Arbor sees dev and gate only; the 300 final questions stay outside its repository
   and are scored once per iteration, after the rule is frozen.
6. **R6 data.** No change to candidates, N, or the question set.
7. **R7 task-specific.** One model's own 10 samples per item only: no other model's answers (no
   cross-model consensus), no gold, no question-order effects. Deterministic: same input, same output.

## 5. Caveats

- These are our re-implementations of ideas described in prose. The papers' numbers (for example
  DeepScientist's 1-3% success rate) are about their systems and tasks, not ours.
- DeepScientist's UCB weights and EvoScientist's retrieval settings (top-k, embedding model) are
  hyperparameters; we choose our own and record them.
- With the free OpenRouter tier at 50 requests per day (D-2026-10-08-b), LLM-judged steps (Elo,
  surrogate) must stay small, or run on another endpoint. Record the model used for every call.

## 6. Second review and amendments (2026-11-06)

An independent second reading of all four papers, plus Arbor's v0.1.4 source code, is in
`BORROWED_IDEAS_REVIEW.md`. It confirms §1-§5 above, with these amendments. The prompts in
`ITERATION_2_PROMPTS.md` were changed to match (marked "Review 2026-11-06").

**Corrections to this file.**
- DeepScientist's own UCB weights are w_u = w_q = κ = 1, with retrieval top-K = 15 (App. C).
- AutoSOTA's red-line count is inconsistent three ways: "six" in the text, seven listed, and "all
  eight" in the third enforcement layer (§3.9).
- AutoSOTA enforces the red lines in four layers: list the constraints before ideation, audit the
  idea library, re-check each Leap idea, and state them in the main prompt as absolute
  prohibitions. Our plan should copy the layering, not only the list.

**Arbor already does part of this (read in its code; we used none of it in iteration 1).**
- Cross-run memory (`EXPERIENCE.md`, recalled at intake).
- A plateau detector that forces a "paradigm shift", close to the leap rule.
- SHA-256 runtime protection of protected files.
- Plugins with six prompt-injection points.

So we build the red lines, the curated memory and the leap rule **as an Arbor project plugin**
(`codification/rerank_task/plugins/rerank_guard.yaml`, written in T1, used in A3), not only as side
files.

**Two safety problems.**
- Iteration 1's `EXPERIENCE.md` records cross-model consensus as a "[leverage]" finding. Never
  reuse it unedited; T3 curates it.
- `eval.py` does not stop `select()` from keeping state across items, which is how the
  cross-model cache worked. Fix before A3: an order-invariance check (call `select()` in two orders;
  any difference scores 0.0).

**Two Arbor settings that do not do what our files say.**
- `merge_threshold` is not enforced at the merge gate (`src/coordinator/tools/git_ops.py`
  merges on any strict held-out gain).
- Experience recall runs only in intake, not under `arbor run`.

**Red lines, additions to §4.**
- **R7 (tightened):** `select(item)` is a pure function of one item. No state across calls, and
  the output does not depend on item order.
- **R8 (new, compute):** report the extra inference compute per question. A gain that needs more
  compute is compared at matched compute against more samples or more steps.

**Twelve more mechanisms (B-12 to B-23):**
- DeepScientist's limitation-first search, starting from majority vote's failure classes.
- AutoSOTA's external research report, built from N3's verified SOTA table.
- AutoSOTA's "debug the baseline first": the fidelity gap goes before Arbor.
- EvoScientist's "tune the baseline before the method": a frozen, tuned majority-vote bar.
- EvoScientist's failure-memory schema.
- EvoScientist's check of the LLM judge against human judgments.
- Arbor's own frozen transfer test, as the second-dataset step.
- Arbor's BrowseComp lessons on the selection ceiling and normalization effects.
- R8 compute.
- Automatic R7 enforcement.
- An exploration score that rewards testing the idea's core claim.
- DeepScientist's reviewer complaints as rubric leaves.

Details, with sections and owners: `BORROWED_IDEAS_REVIEW.md` §4.

**Main conclusion of the review.** Borrowed process alone will not beat majority vote. Arbor's own
iteration-1 audit found the in-item signals over the current features exhausted, at about 0.296-0.298
macro. Point the borrowed machinery at four levers, in this order:
1. Fix the fidelity gap.
2. Add new per-candidate information (DLM self-likelihood, a verifier).
3. Raise the oracle ceiling at matched compute (the step arm).
4. Test a restated, per-question version of idea 1.1's claim.

See `BORROWED_IDEAS_REVIEW.md` §6.
