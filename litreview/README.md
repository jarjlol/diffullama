# Literature Review Generation — QUAL-SG Implementation

This directory implements **QUAL-SG**, the retrieval + generation methodology from *SurveyGen: Quality-Aware
Scientific Survey Generation with Large Language Models* (Bao et al., arXiv:2508.17647), and uses it to
generate a literature-review survey for the domain of the repo's anchor paper — *Scaling Diffusion Language
Models via Adaptation from Autoregressive Models* (DiffuLLaMA, ICLR 2025) — then compares the generated survey
against that paper's own Related Work section.

**Start here:** [`REPORT.md`](REPORT.md) is the full write-up (methodology, what was implemented vs. adapted
and why, results, verification, limitations). This README is a shorter map of the directory for anyone picking
up the code.

## What's in here

```
litreview/
  README.md                    <- you are here
  REPORT.md                    <- full write-up: methodology, results, verification, limitations
  scripts/                     <- the pipeline, run in this order:
    common.py                    HTTP+retry, pure-Python TF-IDF/cosine, percentile-rank (no numpy available)
    retrieve.py                  Step 1: OpenAlex search -> co-citation expansion -> quality-indicator enrichment
    arxiv_fetch.py                recency supplement: same topic, searched against arXiv directly
    merge_finalize.py             merges OpenAlex + arXiv candidate pools, re-ranks, stratifies old/new papers
    rerank.py                     (standalone single-source re-ranker; merge_finalize.py imports its scorer)
    format_refs.py                 formats the final top-K into a prompt-ready reference brief
    evaluate.py                     scores the generated survey against the anchor paper's ground truth
  data/                         all intermediate + final data artifacts (JSON/text), see below
  output/                       outline.md, generated_survey.md, evaluation_results.json
```

### `data/` contents

| File | What it is |
|---|---|
| `anchor_fulltext.txt` | Full text of the anchor (DiffuLLaMA) paper, extracted via `pdftotext -layout` |
| `anchor_related_work_groundtruth.json` | The anchor's 33 §5 citations, resolved to real titles with a `resolution` field per entry explaining how each was verified |
| `candidates_all.json` | Raw OpenAlex retrieval results before re-ranking (~210 papers) |
| `arxiv_candidates.json` | Raw arXiv retrieval results (~172 papers) |
| `candidates_merged_reranked.json` | Combined OpenAlex+arXiv pool after relevance gating and rank-averaged re-ranking (~177 papers) |
| `candidates_reranked.json` | Intermediate single-source (OpenAlex-only) re-rank, superseded by the merged version |
| `candidates_topK.json` | **The final 48 references used in the generated survey**, with full metadata (citation counts, h-index, per-criterion ranks) |
| `reference_brief.txt` | Human-readable numbered version of `candidates_topK.json`, `[1]`-`[48]`, used as the generation-step context |
| `candidates_brief.txt` | Early debugging dump, not load-bearing |

## Running the pipeline

Requires only Python 3 stdlib + the `requests` package (no `numpy`/`pip install` needed — this was built in an
environment without package-install access, so all math is pure Python).

```bash
cd scripts
python3 retrieve.py          # Step 1a-1d: OpenAlex search, co-citation expansion, quality scoring, re-rank
python3 arxiv_fetch.py       # recency supplement
python3 merge_finalize.py    # merge both sources, re-rank, stratify -> writes data/candidates_topK.json
python3 format_refs.py       # -> data/reference_brief.txt
# --- generation step is manual: an LLM reads reference_brief.txt and writes output/generated_survey.md
#     (see REPORT.md §4 for why this step isn't scripted in this implementation) ---
python3 evaluate.py          # -> prints + writes output/evaluation_results.json
```

Each script prints its own progress/results to stderr/stdout — there is no silent failure mode; if a step
produces obviously wrong output (this happened twice during development — see REPORT.md §4 and §7 for both
bugs found and how they were fixed), it will be visible in that script's own printed ranking.

## Key results (see REPORT.md for full discussion)

- 48 references retrieved and cited in the generated survey; **all 48 independently verified as real papers**
  (DOI/arXiv resolution + live title cross-check — see REPORT.md §7).
- Citation overlap against the anchor paper's actual §5: **9/33 matched, F1 = 0.222**.
- All 3 of the anchor's related-work sub-topics are represented as full sections in the generated survey.

## Known limitations (see REPORT.md §8 for full list)

The biggest one: no hosted LLM API was available in the build environment, so QUAL-SG's "LLM-judge" (topical
relevance scoring) and "generator LLM" (outline + section writing) roles were played directly by the AI
assistant building this, rather than through a scripted, repeatable API call. Anyone continuing this project
with an API key available should prioritize replacing `scripts/rerank.py`'s keyword-based relevance scorer with
a real LLM-judge call — this is the single change most likely to close the retrieval-recall gap documented in
REPORT.md §6.
