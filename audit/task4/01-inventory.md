# 01 — Inventory: what exists (2026-11-02)

## Course and team

BITS F471, Group 3, seven members. The project is 60% of the grade and the comprehensive exam
(30%) is on 2026-12-03. The anchor paper is DiffuLLaMA (Gong et al., ICLR 2025, arXiv:2410.17891),
forked at `jarjlol/diffullama`. Source: `project-docs/01-project-brief.md`.

## Assignments

| # | Assignment | Directory / branch | State | Evidence |
|---|---|---|---|---|
| 1 | SOTA literature review (QUAL-SG + 2412.13612 evaluation) | `litreview/` on `main` | submitted | `litreview/REPORT.md`; F1 0.222, 0.272 with LLM judge |
| 2 | Limitations / research gaps (2601.11578 multi-agent) | `limitations/` | submitted; all 12 gaps declined with justification | `limitations/output/gap_dismissal_analysis.md`, D-2026-09-27-a |
| 3 | Problem discovery + ideation (ResearchAgent) | `ideation/`, branch `assignment/research-ideation` | run complete; P1 + P2, top 5 each | `ideation/output/research_problems_and_ideas.md`, REPORT §7 audit |
| 4 | **Codification** | `codification/`, branch `assignment/codification` | engineering done, no GPU results | this audit |

Main project (stages 2–6): **direction still pending** (P-1 → P-5). D-2026-09-21-a made it
inference-only. `project-docs/10` recommends L6 (answer-selection headroom) as primary and the
adaptation-residue audit as secondary. Neither is ratified.

## Branches on `origin` (verified with `git fetch`)

| Branch | Last commit | Owner | Status |
|---|---|---|---|
| `main` | 066d8f2, 2026-09-28 | Neel | does not contain ideation or codification |
| `assignment/research-ideation` | 4559f77, 2026-11-01 | Neel / Arjun | merged into codification |
| `assignment/codification` | cc5f1da, 2026-11-02 | Neel, Aaditya | **current work** |
| `assignment/limitations-multiagent` | 0bca405, 2026-09-27 | Neel | |
| `plan/inference-time-direction` | 666df2f, 2026-09-21 | Aaditya | docs only |
| `plan/execution-grounded-repair` | 35a6486, 2026-09-19 | Aalhad | EGR, preempted by DiffPDE |
| `audit/opus-review` | | | throwaway, scheduled for deletion |

`main` is five weeks behind the work. Nothing from ideation or codification has been merged.

## Task 4 artefacts (`codification/`)

| Path | Purpose |
|---|---|
| `REFERENCE_SYSTEMS.md` | Which of the four reference papers has a usable implementation (only Arbor) |
| `IDEA_SELECTION.md` | Why idea 1.1, the six specification issues found before codification |
| `README.md` | Full runbook (venvs, steps 1–7, Arbor) |
| `ITERATION_1.md` | Iteration-1 scope, issue log I-1…I-12, published numbers, run plan, result tables |
| `rerank_task/` | Arbor task pack: protected `task.py`/`eval.py`/`baselines.py`, editable `solution.py` |
| `pipeline/` | GPU side: prepare → generate → score → assemble → CKA → analyze; `selftest.py` |
| `campaign/campaign.py` | Supervisor for the shared Blackwell workstation (added 2026-11-02) |

## Machines

| Machine | GPU | Access | State on 2026-11-02 |
|---|---|---|---|
| Blackwell workstation (`jump-blackwell`, 10.1.19.153, user `antpc`) | 2× RTX PRO 6000 Blackwell, 97,887 MiB each | key-only through the two dashlab jump hosts; works | **both cards held by user `atharva`**: a vLLM benchmark sweep (`run_cell.py`, Llama-2-13B) restarting its server every few minutes |
| ada (`jump-ada-gw`, 10.1.19.63) | 1× RTX 5000 Ada, 32,760 MiB | key-only through the same jumps | idle (one 0.5 GB streaming process) |
| Aaditya's laptop | RTX 3050 4 GB | local | no torch installed; CPU tests only |
| Sharanga cluster | many | not approved | not a dependency |

On Blackwell, codification lives in `~/codify/diffullama`, a fresh clone kept identical to
`origin/assignment/codification`. Its venvs are `~/venvs/{anchor,dream,llada,scorer}` (torch
2.7.1+cu128, sm_120 present, transformers pinned per family). The campaign state is in
`~/codify/campaign/`.

**Risk found:** the older checkout `~/diffullama` on Blackwell (branch `assignment/research-ideation`)
holds **997 lines of uncommitted changes across 9 files, including `project-docs/02-decision-log.md`**.
They were last touched on 2026-10-21. Someone's work exists only on that disk. It was not touched
by this audit. Whoever made it should commit and push it.

## Decisions on record that constrain Task 4

- D-2026-09-21-a: inference only, no training.
- D-2026-10-08-a: P1 and P2 chosen.
- D-2026-11-01-a: Arbor is the codification system.
- D-2026-11-01-b: idea 1.1, marked "to be confirmed by the team", is still unconfirmed.
- Iteration-1 decisions on 2026-11-02 (Aaditya): dev 200 / test 500 subset, all models at 64 steps,
  DiffuGPT-S 1-shot with 192 new tokens, Llama-2 via the NousResearch copy.
