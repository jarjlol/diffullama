# Limitation Generation via Multi-Agent LLMs

Faithful implementation of **Al Azher, Guo & Alhoori, *Multi-Agent LLMs for Generating
Research Limitations*** ([arXiv:2601.11578](https://arxiv.org/abs/2601.11578)), applied to
this project's anchor paper, **DiffuLLaMA** (Gong et al., ICLR 2025).

**The deliverable is [`output/limitations_and_research_problem.md`](output/limitations_and_research_problem.md)** —
a bulleted list of research gaps/limitations followed by the paragraph-level research-problem
statement. Full methodology, every deviation from the source paper, and honest limitations of
this pipeline are in **[`REPORT.md`](REPORT.md)**.

## What is implemented

All seven agents of the source paper, in its execution order:

```
Extractor ─┐
Analyzer  ─┤
Reviewer  ─┼─► Judge ─► [score < 8/10] ─► Self-Feedback ─► Master ─► deliverable
Citation  ─┘                                   │
    ▲                                          └─ regenerate, max 2 retries
    │
  RAG: hybrid BM25 + dense retrieval over cited-in / cited-by chunks,
       top-20, LLM re-rank, retain ≥ 8/10
```

| Component | Source paper | Here |
|---|---|---|
| Worker agents | Extractor, Analyzer, Reviewer, Citation | all four, prompts in `data/agent_requests/` |
| Judge | Depth .2 / Originality .2 / Actionability .3 / Coverage .3, 0–100 | implemented exactly, `scripts/judge.py` |
| Self-Feedback | regenerate below 8/10, ≤2 retries | implemented; **fired for 2 of 4 agents** |
| Master | cluster → merge → provenance-tag | implemented, `scripts/master.py` |
| RAG | BM25 + FAISS, top-20, re-rank ≥8 | BM25 exact; FAISS substituted (REPORT §3.2) |
| Evaluation | pointwise LLM-as-Judge C_GT | implemented, `scripts/evaluate.py` |
| Ground truth | author-stated **+ OpenReview** | author-stated only — OpenReview unreachable (REPORT §3.4) |

## Input: the prior assignment

Per the assignment brief, the literature surveyed in the previous assignment feeds this one.
The RAG corpus is built from [`../litreview/`](../litreview/) — its 48 retrieved and verified
references supply the paper's `X_in` ("Cited In") stream, and OpenAlex supplies `X_by`.

## Results

| | |
|---|---|
| RAG chunks built | 95 (48 cited-in + 47 cited-by) |
| Retained after re-rank ≥8/10 | 13 |
| Raw limitations from 4 agents | 33 |
| Consolidated by Master Agent | **12**, none dropped |
| Corroborated by >1 agent | 10 of 12 |
| Self-Feedback triggered | Extractor 72→84, Reviewer 78→86 |
| C_GT vs zero-shot baseline | 1.000 vs 0.583 — **inflated, see REPORT §6.2** |
| Generic statements | 0 (multi-agent) vs 3 (zero-shot) |

## Run it

```bash
cd limitations/scripts && python3 run_pipeline.py
```

Pure standard library plus `urllib` for the OpenAlex calls. No numpy, sklearn, faiss, or
torch — consistent with the constraint documented in the repository's `AGENTS.md`.
