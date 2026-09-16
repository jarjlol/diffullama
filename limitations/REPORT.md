# Limitation Generation / Research Gap Identification — Methodology Report

## 1. Assignment

Extract limitations/research gaps of the base paper (DiffuLLaMA, ICLR 2025), optionally using the
literature survey's papers to find limitations, and using the generated survey to identify
domain-level gaps. Techniques from three offered papers "can be useful." Deliverable: a bulleted
list of limitations, followed by a paragraph-level research-problem statement continuing the base
paper's discussion.

**Scope note, stated explicitly:** this implementation does not reuse any limitation, finding, or
conclusion from this repository's prior audit work (`project-docs/03-established-facts.md`,
`06-findings-and-wins.md`) or the `litreview/` SOTA assignment's outputs, per the explicit
instruction this work was scoped under. Every finding below was re-derived from the anchor paper's
primary text and fresh literature retrieval, on this branch, independent of that prior work. Where a
finding happens to overlap in topic with something already logged elsewhere in this repo (e.g., the
mask-token treatment, or the annealing ablation), that is because it is a real, independently
verifiable property of the paper — not because it was copied. The anchor's full text was re-extracted
via `pdftotext` from the source PDF into `limitations/data/anchor_fulltext.txt`, a fresh copy
independent of `litreview/data/anchor_fulltext.txt`.

## 2. Papers implemented, and how

All three were read in full (not summarized from search results) before implementation began:

| Paper | What it contributes | How it's used here |
|---|---|---|
| **BAGELS** (Al Azher et al. 2025, arXiv:2505.18207, EMNLP Findings 2025) | Explicit/implicit limitation span extraction: keyword-triggered extraction, LLM-refined to remove noise, strictly "select, don't paraphrase or invent" | `extract_stated_limitations.py` (Extractor role) |
| **LimitGen** (Xu et al. 2025, arXiv:2507.02694 — co-authored by Manasi Patwardhan, TCS Research, this course's own industry collaborator) | Four-aspect limitation taxonomy (Methodological / Experimental Design / Result Analysis / Literature Review) built from real peer-review patterns; demonstrates RAG-augmented limitation generation beats zero-shot | Taxonomy applied directly in Master consolidation (`data/master_consolidated.md`); RAG-grounding role in `citation_agent.py` |
| **Multi-Agent LLMs for Generating Research Limitations** (Al Azher et al. 2026, arXiv:2601.11578) | Seven-role pipeline: Extractor, Analyzer, Reviewer, Citation Agent, Judge, Self-Feedback, Master | Overall pipeline architecture (see §3) |

## 3. Pipeline (what was actually run)

```
anchor_fulltext.txt (fresh pdftotext extraction)
  -> Extractor (regex keyword scan, BAGELS-style) -> extractor_candidates.txt (14 raw hits)
  -> LLM refinement (BAGELS step 2: select genuine spans, discard noise, no paraphrasing)
       -> extractor_refined.md (5 kept, 9 discarded with reasons)
  -> Analyzer (fresh methodological audit of the full paper, independent read)
       -> analyzer_findings.md (8 inferred implicit limitations, A1-A8)
  -> Citation Agent (fresh OpenAlex + arXiv retrieval per candidate, RAG-style per LimitGen)
       -> citation_agent_results.txt (raw retrieval) + citation_agent_verdict.md (groundedness judgments)
  -> Judge (score groundedness/depth/actionability, filter) + Master (dedupe, merge, taxonomy-tag)
       -> master_consolidated.md (10 final items, ranked)
  -> output/limitations_and_research_problem.md (the assignment's required deliverable)
```

**Roles not separately implemented:** the Multi-Agent paper's **Reviewer** agent (peer-review-lens
critique — reproducibility, transparency) and **Self-Feedback** agent (regeneration loop below a
quality threshold) were not built as distinct steps. Reviewer's concerns are largely already covered
by the Analyzer findings above (e.g., A1/A2's ablation-completeness critique is exactly what a peer
reviewer would flag); Self-Feedback's regeneration loop was substituted with a single, careful
first-pass Judge filter (one item, E1, was filtered — see `master_consolidated.md`) rather than an
iterative regenerate-and-rescore loop, given this pipeline's Extractor/Analyzer/Judge/Master steps
are performed directly by an LLM (see §4) rather than via a scriptable regeneration call.

## 4. Deviations from the papers, and why (same standard as `litreview/REPORT.md`)

No hosted LLM API (`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`) is available in this build environment —
identical constraint to the SOTA assignment. Every step that the source papers implement as an LLM
API call (BAGELS' refinement LLM, LimitGen's generation LLM, the Multi-Agent paper's Analyzer/
Reviewer/Citation/Judge/Master LLM instances) is instead performed directly by Claude, reading the
intermediate files and producing the next stage's output — the same substitution `litreview/`
documents, applied consistently here. This is not a silent shortcut: every judgment call (which
candidates to keep, how to score them, how to merge duplicates) is written out in the corresponding
`data/*.md` file with its reasoning, so the process is auditable rather than opaque.

**Retrieval substitution:** LimitGen's Citation-Agent-equivalent step queries Semantic Scholar.
Unauthenticated Semantic Scholar returns HTTP 429 almost immediately under real use — the exact
failure mode `litreview/REPORT.md` and `project-docs/05-mistakes-and-bugs.md` (M-19) already
document for a different script in this repo. `citation_agent.py` uses OpenAlex + arXiv instead, the
same substitution `litreview/` made, arrived at independently for this script rather than copied
from it (no code is imported across the two assignment folders).

**A real, encountered infra failure, kept rather than hidden:** arXiv's API rate-limited this
script (HTTP 429) after the combined OpenAlex+arXiv query volume in one run. Rather than retry in a
loop until it cleared, the Citation Agent step proceeded on OpenAlex results alone for the affected
queries; `citation_agent_verdict.md` states exactly which conclusions rest on OpenAlex-only evidence
and flags that result as weaker/inconclusive rather than confirmatory where retrieval came back
empty or off-topic.

**Retrieval-recall limitation, stated directly:** the Citation Agent's automated queries did not
reproduce a specific paper (NBDiff, arXiv:2512.06776) that a *differently-phrased* search elsewhere
in this repository's history found relevant to the same underlying question (annealing tested at
7B scale). This is recorded in `citation_agent_verdict.md` as a genuine limitation of this
implementation's query phrasing, not glossed over.

## 5. Results

10 final limitations (down from 14 raw Extractor hits + 8 Analyzer inferences, after merging
overlapping items and filtering one below the Judge's quality threshold), tagged with LimitGen's
four-aspect taxonomy and provenance (author-stated / inferred / literature-grounded). Full ranked
list and reasoning: `data/master_consolidated.md`. Final deliverable (bulleted list + research-
problem paragraph): `output/limitations_and_research_problem.md`.

## 6. Honest limitations of this implementation itself

- The Extractor's regex keyword list is not exhaustive — it demonstrably misses limitation-relevant
  passages phrased without a keyword trigger (the annealing "minimal impact" sentence, found only by
  the Analyzer's full-paper read; see `extractor_refined.md`'s final section). This is expected and
  is exactly why the pipeline pairs extraction with an independent analytical pass, per all three
  source papers' own design — not a bug specific to this implementation.
- The Citation Agent's OpenAlex-based retrieval is noisy for this narrow a technical domain (most
  returned results across all four query groups were off-topic high-citation surveys), consistent
  with the same weakness already documented for `litreview/`'s first retrieval pass before it was
  tuned. This implementation did not go through an equivalent tuning pass, given the assignment's
  narrower scope; `citation_agent_verdict.md` reports groundedness honestly rather than overstating
  confidence in a noisy retrieval result.
- No human expert review of the final 10 items was performed (BAGELS and the Multi-Agent paper both
  use human/SME verification as part of their evaluation; that step is out of scope here given no
  panel of human annotators is available).
