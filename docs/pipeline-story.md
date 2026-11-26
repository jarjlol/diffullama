# How the pipeline ran: manual, agent-assisted, coordinated

This is the human/agent split behind the code, written so a reader can check it. Each
stage names the artefact and the commit that produced it; the full reports are in the
linked directories.

| Stage | Manual | Agent-assisted | Artefact |
|---|---|---|---|
| 1. Literature review | Choice of anchor, reading, and the survey prose (a person prompted Claude with the reference brief) | Deterministic retrieval and ranking (`litreview/scripts/`): 48 references retrieved and ranked; all 48 verified as real (47 by DOI, 1 via Crossref); overlap with the anchor's related work measured by exact and fuzzy title matching | `litreview/REPORT.md` @ `6316fd1` |
| 2. Research gaps | Refinement of the extracted list; the final judge pass | Extractor → Analyzer → Reviewer → Citation → Judge roles; every gap carries provenance, and author-stated is kept separate from inferred | `limitations/REPORT.md` @ `a131e13` |
| 3. Ideation | The team's audit of the output and the final choice of idea 1.1 for task fit, not for rank | ResearchAgent implementation: generate → review → refine on five criteria over three rounds; output kept unedited, including the audit that found model-lineage errors and saturated ratings | `ideation/REPORT.md` @ `4559f77` (`assignment/research-ideation`) |
| 4. Codification | Review of Arbor proposals, with unanswered ideas auto-approved after a timeout; the scope decision for iteration 2 | Arbor: hypothesis tree, one selector function per change, held-out gate. Iteration 2 added the fidelity check, cut-text re-scoring, the step arm, the self-score, the per-question test and a pre-registered K = 256 check | `codification/ITERATION_1.md` @ `b4688db` (`assignment/codification`), `ITERATION_2.md` |

## Coordination

Work ran through GitHub issues with IssueMesh, and the rules are the reason it stayed
reviewable:

- **One item, one owner.** An agent claims an issue with a lease; two agents cannot work
  the same item, and an expired claim returns to the pool.
- **The claim comment is the status.** Plan, checklist with a receipt on every "done"
  line, next step. A fresh session resumes from it instead of re-reading the thread.
- **Reviews are independent.** A different member's agent reads the acceptance criteria
  and the diff first, runs the checks itself, and names what it found.
- **Humans keep the decisions that should not be delegated.** In this project: the Task 4
  iteration-2 scope, the final upload, the author line, and the claims that went into the
  paper. Agents drafted, recomputed numbers against result files, reviewed each other's
  PRs, and kept the paper inside its page budget.

A worked example a reader can open: issue #57 (the K = 256 result into the paper) — claim,
draft PR, an independent review that recomputed every number against
`k256_step_arm_output.txt`, a page-count check, merge. The catalogue item itself (#66,
PR #67) is a second one.

## Research utilities

IssueMesh v1.2.1 ships `mesh util`: a catalogue of published methods mapped to
research-lifecycle stages, with a receipt ledger. Our catalogue (`.issuemesh/utilities.json`,
issue #66 / PR #67) records the four methods used here — SurveyGen, multi-agent limitation
generation, ResearchAgent and Arbor — with the artefact and commit each produced;
`mesh util report` prints that table. The records are marked **retroactive** because
Tasks 1–4 predate the toolkit, and the ledger says so.

## What this is not

- Not every step was agent-run: the survey prose was manually prompted, and parts of the
  limitation pipeline are manual passes.
- The paper's formal verification was partly waived by the team; the provenance appendix
  in `final_submission/paper/` records what was and was not checked.
