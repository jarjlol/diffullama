# Superseded: pre-#106 Markdown build of the paper

**These seven section files are not the submitted paper.** They are the earlier Markdown
build, kept for history. The submitted paper is the LaTeX build at
[`../paper.pdf`](../paper.pdf), built from [`../latex/`](../latex/).

The two differ in more than formatting. Until PR #106 (`1f1ec7a`, 26 Nov) the paper was
framed around a negative result — "no selector beat majority", with the CKA/preservation
hypothesis as a second contribution. PRs #106–#108 rebuilt it around the decision-model
reranking result: a listwise decision model beats majority at both budgets (+5.2 points
at $K=64$, +4.0 at $K=256$; see `codification/DECISION_RERANK_RESULTS.md`), and the
preservation hypothesis was removed (#107). So numbers here can directly contradict the
submitted paper, for example:

| Claim | This directory (stale) | Submitted paper (`../paper.pdf`) |
|---|---|---|
| Best selector at $K=64$ | "No selector beat majority"; best AR 48.8 / 52.2 | Decision listwise 77.0 / 79.0, +5.2 over majority |
| CKA preservation | Abstract and Conclusion report an unsupported hypothesis | Removed in #107 |
| Paper title/framing | AR-selection negative result | Decision-model reranking for diffusion LMs |

Do not cite numbers from these files, and do not rebuild `../paper.pdf` from them:
`../build.py` writes to `../build/paper.pdf` (gitignored), so it cannot overwrite the
canonical PDF, but its output is still the superseded paper.

Canonical sources: [`../latex/sections/`](../latex/sections/) and
[`../latex/references.bib`](../latex/references.bib).
