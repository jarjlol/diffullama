# Start-up prompts for each member's agent (Task 5)

**Why this exists.** `to:` and `ping:` labels notify **nobody**. GitHub only notifies people on @mentions, assignment and subscribed threads, and nothing wakes an agent. So every agent must (a) subscribe to the repo and (b) look at its issues itself, at the start of each session, after each item, and every ~10 minutes while working. The prompts below say so.

khushi and diya do not use agents and have no Task 5 issues. Replace nothing except what is in `<ANGLE BRACKETS>`.

## Who has what (labels `to:<handle>` are authoritative)

| Member | Agent | Issues |
|---|---|---|
| neel | Claude Code | #8 Introduction, #10 Problem Formulation (in review), #14 assembly (after drafts) |
| aaditya | Claude Code | #9 Related Work, #11 Methodology, #17 epic tracker (keep it current) |
| arjun | OpenCode | #12 Experiments / Results; #16 upload to the group folder (a human step, on Wed 25 / Thu 26) |
| aryan | Claude Code (+ Codex) | #15 Report, #13 Abstract + Conclusion (blocked until #8 to #12 are merged) |

Decision #7 (how much of Task 4 iteration 2) needs a one-line answer from a human, not an agent.

## Master prompt (Claude Code, Codex, any agent)

```text
Use aalhadsawane/IssueMesh with jarjlol/diffullama. I am already on the team as <HANDLE>, so skip joining if `mesh whoami` works there.

Do this, in order:

1. Install or update the CLI: npm i -g github:aalhadsawane/issuemesh#v1 . In the repo run `mesh doctor` and `mesh whoami`. (Claude Code only: also /plugin marketplace add aalhadsawane/issuemesh then /plugin install issuemesh@issuemesh, once.)

2. IMPORTANT: ping/to labels do NOT notify anyone. GitHub only notifies on @mentions, assignment and subscribed threads, and nothing wakes an agent. So you must look yourself:
   a. Subscribe me to the repo: gh api -X PUT repos/jarjlol/diffullama/subscription -F subscribed=true -F ignored=false . If that is refused, tell me to click Watch > Custom > Issues on the repo.
   b. At the start of every session, after finishing each item, and every ~10 minutes while working, run: mesh inbox ; mesh board ; gh issue list --repo jarjlol/diffullama --state open --milestone "Task 5: Final Submission (due Thu 26 Nov 2026)" . Anything labelled to:<HANDLE> is mine even if it is missing from `mesh inbox`.
   c. Re-read the comments of every issue and PR I own before you push anything; claiming an issue (mesh claim) subscribes me to it.

3. Read AGENTS.md, docs/issuemesh.md and final_submission/README.md.

4. Work my issues: <MY ISSUES, e.g. #9 and #11>. For each: mesh claim <n> with a short plan; one branch mesh/<n>-<slug> from the latest main containing ONLY that section's file (final_submission/paper/sections/<section>.md); open a draft PR early; keep one status comment (mesh status <n> --set ...) with receipts. Every number or claim must trace to a repo file or a paper read at full text; otherwise write [unverified]. When a draft is ready: mesh release <n> --to-status review and ping nobody, a different member's agent will review. Deadlines: drafts Mon 23 Nov, results freeze Sun 22 Nov, final Wed 25 Nov, submit Thu 26 Nov.

5. After your own items, review ONE other member's section PR (never your own): read the issue's acceptance criteria and the diff first, run what can be run, then list concrete weaknesses or say what you checked and found clean.

6. Do not merge PRs, do not edit other members' sections, do not answer decision issue #7 (a human does). Report what you did in at most 10 lines.
```

Fill in `<HANDLE>` (neel, aaditya, aryan) and `<MY ISSUES>` from the table above.

## Arjun (OpenCode)

OpenCode has no IssueMesh plugin and no hooks, so it needs the portable prompt and does the polling itself.

```text
Use aalhadsawane/IssueMesh with jarjlol/diffullama. I am arjun (agent name: opencode) and already on the team.

1. Install: npm i -g github:aalhadsawane/issuemesh#v1 . In the repo: mkdir -p .issuemesh && curl -fsSL https://raw.githubusercontent.com/aalhadsawane/issuemesh/v1/prompt/ISSUEMESH.md -o .issuemesh/ISSUEMESH.md . Then run `mesh doctor` and `mesh whoami` (expect handle arjun, agent opencode). Follow .issuemesh/ISSUEMESH.md for the work loop; it lists every mesh command and the plain gh fallback for each step.

2. IMPORTANT: ping/to labels do NOT notify anyone, and OpenCode has no hooks, so you will not be told about new work. You must look yourself: at the start of the session, after each item and every ~10 minutes, run: mesh inbox ; mesh board ; gh issue list --repo jarjlol/diffullama --state open --milestone "Task 5: Final Submission (due Thu 26 Nov 2026)" . Anything labelled to:arjun is mine even if it is not in `mesh inbox`. Subscribe me to the repo: gh api -X PUT repos/jarjlol/diffullama/subscription -F subscribed=true -F ignored=false (if refused, tell me to click Watch > Custom > Issues).

3. Read AGENTS.md, docs/issuemesh.md, final_submission/README.md.

4. My issues: #12 Experiments / Results (write the iteration 1 part now from codification/ITERATION_1.md on branch assignment/codification; add iteration 2 numbers only after the results freeze on Sun 22 Nov and only for finished runs). #16 is a human upload step: remind me on Wed 25 Nov, do not claim it. For #12: mesh claim 12; branch mesh/12-experiments from the latest main containing only final_submission/paper/sections/experiments.md; draft PR early; one status comment with receipts; every number must match the source file exactly; unchecked claims are marked [unverified]; mesh release 12 --to-status review when the draft is ready.

5. Then review ONE other member's section PR (not mine). Do not merge, do not edit other sections, do not answer decision #7 (a human does). Report in at most 10 lines.
```

If your OpenCode version supports a non-interactive run, a hands-off loop is
`mesh watch --exec "opencode run 'Follow .issuemesh/ISSUEMESH.md and work my mesh inbox'"` (check `opencode --help` first). Otherwise start the session by hand every time you sit down, and ask it to re-run step 2 every ~10 minutes.

## Codex (aryan's second agent)

Same as the master prompt, plus: `export ISSUEMESH_AGENT=codex` before running `mesh`, step 1 without the Claude Code plugin line, and save the portable prompt (`mkdir -p .issuemesh && curl -fsSL https://raw.githubusercontent.com/aalhadsawane/issuemesh/v1/prompt/ISSUEMESH.md -o .issuemesh/ISSUEMESH.md`), then "follow .issuemesh/ISSUEMESH.md". Mailbox items for aryan go to the primary agent (claude); address codex with `to:aryan/codex`.
