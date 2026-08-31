# Project Context — START HERE

Durable record of the BITS F471 group project. Exists so that **context survives** chat compression,
new conversations, and switching between AI agents.

---

## If you are an AI agent (Claude, Codex, Gemini, DeepSeek, or any other)

**Read these four files before doing anything or answering any question about this project.** They are
short and they are the source of truth.

| Order | File | What it gives you |
|---|---|---|
| 1 | [`01-project-brief.md`](01-project-brief.md) | What the course requires, who is on the team, what is due |
| 2 | [`02-decision-log.md`](02-decision-log.md) | What has been decided, when, and why — **do not re-litigate settled decisions** |
| 3 | [`03-established-facts.md`](03-established-facts.md) | Verified findings with sources and confidence levels |
| 4 | [`04-open-questions.md`](04-open-questions.md) | What is genuinely unknown or blocked — this is where help is useful |

Then, as needed:

| File | What it gives you |
|---|---|
| [`05-mistakes-and-bugs.md`](05-mistakes-and-bugs.md) | **Every error made so far and its root cause.** Read before writing code or citing anything — three bug *classes* have already recurred |
| [`06-findings-and-wins.md`](06-findings-and-wins.md) | Novel findings and results worth writing up |
| [`07-decision-tree.md`](07-decision-tree.md) | Diagrammed reasoning behind every fork — read this before proposing a direction that was already closed |
| [`08-trace-guided-repair-proposal.md`](08-trace-guided-repair-proposal.md) | Proposed Option C2, its exact scope, baselines, and feasibility gate — **not yet a team decision** |

### Rules for agents working on this project

1. **Distinguish verified from unverified.** `03-established-facts.md` labels every claim with how it was
   checked. Do not upgrade a "single-source" claim to a fact because it is convenient. Do not state
   something as established when the file says it is not.
2. **Never invent a paper.** Every citation in these docs resolves to a working link. If you cite
   something new, verify it resolves first. This project has already been nearly derailed once by a
   concurrent paper nobody checked for — see decision D-2026-08-28-a.
3. **Check the date on facts.** This field publishes roughly ten preprints a week. A novelty finding from
   two months ago may be stale. Re-verify before relying on it.
4. **Append, do not rewrite.** These are logs. If a fact changes, add a dated correction underneath it
   rather than editing history away. Retractions are recorded on purpose — see §23 of the audit for an
   example of why that matters.
5. **If you disagree with something here, say so explicitly** rather than silently working around it.
   Prior disagreements between analyses are recorded, not resolved by fiat.

---

## For humans

`02-decision-log.md` is the one to read if you have been away. It is chronological and each entry says
what was decided and why.

Pending decisions live at the top of that file and in
[`audit/DECISIONS.md`](https://github.com/jarjlol/diffullama/blob/audit/opus-review/audit/DECISIONS.md)
on the `audit/opus-review` branch.

---

## Repository layout

| Location | What it is | Permanent? |
|---|---|---|
| `project-docs/` | This folder — durable project record | ✅ yes |
| `litreview/` | SOTA assignment: QUAL-SG literature-review pipeline | ✅ yes |
| `model.py`, `inf_*.py`, `DiffuLLaMA-training/`, `LLaMA-Factory/` | Upstream DiffuLLaMA code (forked) | ✅ yes |
| `audit/` on branch `audit/opus-review` | Design-doc audit, novelty verification, decision options | ❌ **throwaway** — will be deleted |

> ⚠️ **The `audit/opus-review` branch is scheduled for deletion.** Its durable conclusions have been
> copied into `03-established-facts.md`. If you need the full reasoning and evidence trail before it is
> deleted, read
> [`audit/OPUS_Audit.md`](https://github.com/jarjlol/diffullama/blob/audit/opus-review/audit/OPUS_Audit.md).

---

## Maintaining this folder

- Add a dated entry to `02-decision-log.md` whenever the team decides something.
- Add to `03-established-facts.md` only what has been **verified**, with the source and how it was checked.
- Move items from `04-open-questions.md` to `03-established-facts.md` when they get resolved — and say
  what resolved them.
- Keep entries short. This folder is read at the start of every new agent session; bloat costs everyone
  context budget.

**Last updated:** 2026-08-31
