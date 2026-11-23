# Report evidence and remaining review

Snapshot: 21 Nov 2026, before the 22 Nov 18:00 IST paper-results freeze.
The four-page report is a reviewable draft, not the final post-freeze submission.
Authored with Aryan's Codex session on 21 Nov 2026. The underlying model identifier
and temperature are not exposed by this session, so they are not invented here.
No separate generator/judge API was called to produce this report.

## Checked in this session

- All 25 accuracy cells in the five-model final table match
  `69d76a8:codification/results/iter1/numbers/baselines_final300.json`, with Arbor
  values cross-checked against `paired_final300.txt`. Each model has n = 300.
- Dream's 3/4 and LLaDA's 3/3 paired wins/losses and p = 1.000 match that paired artifact.
  The report says no demonstrated gain, not statistical equivalence.
- Literature overlap counts come from `evaluation_results.json` and
  `evaluation_results_llm_judged.json` at `69d76a8`. The later matcher pairs
  **Non-Autoregressive Neural Machine Translation** with **Glancing Transformer for
  Non-Autoregressive Neural Machine Translation**, score 0.625. This is a suspect
  paper-identity match; reported F1 is not presented as an independently verified gain.
  The submitted survey and earlier result files were not edited.
- `evidence/fidelity_snapshot.json` was independently recounted from Blackwell's saved
  JSONL files at the timestamp in the snapshot, using `correct`, unique uid counts and
  SHA-256 file digests. No candidate text, item data or gold files are committed here.
- `fidelity_check.py --report` was rerun read-only on Blackwell. Dream-sampler n=100
  greedy 0.810 vs first 0.410; LLaDA n=100 greedy 0.730 vs first 0.510;
  Dream-published n=55 greedy 0.782 vs first 0.400. These sample sizes are separate
  from the iteration-1 final300 table. This does not isolate temperature from steps.
- The source labels link to pinned repository versions. No new paper bibliography or
  current novelty assessment is introduced by this report.
- PDF built with ReportLab, parsed with pypdf: **4 pages**. All four page images
  inspected: no clipped paragraphs, overflowing table cells or missing content.

## Before finalising issue 15

1. Read the newest #4 comments and completed GPU artifacts after the results freeze.
   Refresh the iteration-2 section without overwriting iteration-1 history. The
   cut-text comparison can change the scope of the earlier AR-loss interpretation.
2. Keep incomplete step-arm, self-score, preservation, Arbor, published-baseline and
   transfer experiments explicitly incomplete if no completed evidence arrives.
3. A different member must review the report and record findings on #15. Self-audit
   does not meet that acceptance criterion.
4. Rebuild after edits and inspect every PDF page; keep the report within 2–4 pages.
5. Aryan prohibits opening PRs. Publish this branch by commit/push and link it from
   #15; do not merge or create a PR as a shortcut to review.

## Build

```bash
python3 final_submission/report/build_report.py
```

Dependency: `reportlab`. The build reads `report.md`; measured values are not duplicated
in the renderer. The preserved `<!-- pagebreak -->` markers group content into four pages.
Recheck page count after every content edit. `evidence/fidelity_snapshot.json` must travel
with the PDF to keep its relative source link useful.
