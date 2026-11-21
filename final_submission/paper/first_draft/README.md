# First assembled paper draft

`output/pdf/paper_first_draft.pdf` has four body pages and one reference page.
`paper.md` is the editable content source. This is an assembled first draft, not
independent member sign-off or a submission-ready paper. It does not overwrite
member-authored section drafts or any submitted deliverable.

Build with Python plus ReportLab:

```bash
python3 final_submission/paper/first_draft/build_paper.py
```

The renderer uses invariant PDF metadata for reproducible output. The document's
visible date is 21 November 2026. All five pages were rendered and visually reviewed;
25 main-table accuracy entries were checked against frozen iteration-1 artifacts.
The seven citation identifiers resolve to the stated primary papers. Bibliography
coverage and novelty claims are deliberately narrow; broader related-work material
that had only abstract-level review was not imported.

Completed iteration-2 fidelity and baseline-calibration results are included.
Step-budget generation counts describe progress only. GPU 0 reported a device error,
retry budgets were exhausted, and incomplete scoring/search/final evaluation is not
presented as finished. The final300 split was not re-evaluated for this assembly.
No positive selector improvement, equivalence proof, or causal preservation claim
is made. A matched-compute comparison, corrected AR scores, native confidence,
ablation and transfer remain unfinished. These limitations are explicit in the paper.

The standing operator automation was deleted at Aryan's request to wrap up and save
tokens. GPU/CPU STOP files are set, partial files and pending work are preserved,
and pending unattended A3 launches were removed. No foreign processes were signalled,
GPU reset attempted, or PR/merge created. Resume only after checking hardware and
following the documented operator rules. Source receipts are in SOURCES.json and
codification/results/iter2/aryan/wrapup_snapshot.json.
