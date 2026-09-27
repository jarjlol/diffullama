# Research Problem Discovery and Ideation — ResearchAgent

Implementation of **ResearchAgent** (Baek, Jauhar, Cucerzan, Hwang — *Iterative Research Idea
Generation over Scientific Literature with Large Language Models*,
[arXiv:2404.07738](https://arxiv.org/abs/2404.07738)) for the course's ideation assignment:
derive 1–2 fine-grained research problems from the prior assignment's gaps, then generate,
review, refine and rank research ideas, submitting the top 5 per problem.

**Deliverable:** `output/research_problems_and_ideas.md` (produced once the pipeline has run).
**Methodology, every deviation from the paper, and known risks:** [`REPORT.md`](REPORT.md).

> **Status.** Pipeline implemented and verified end to end on a mock backend (10/10 wiring
> checks). **Not yet run with a real model** — no problems or ideas have been generated. The
> first real request is waiting in `data/llm/requests/`.

## Pipeline

```
litreview/ + limitations/ ─► 1 literature ─► 2 entity store (Eq. 2)
                                                  │
                                                  ▼
            3 problems  ──  ResearchAgent (Table 6) ⇄ ReviewingAgents ×5 criteria (Tables 9, 13), ×3 rounds
                                                  │
                                                  ▼
            4 SELECTION GATE — the team picks 1–2 problems
                                                  │
                                                  ▼
            5 ideas, per problem: 8 × [ method (Tables 7, 10, 14) ⇄ reviews, ×3 ]
                                      → [ experiment (Tables 8, 11, 15) ⇄ reviews, ×3 ]
                                                  │
                                                  ▼
            6 rank by mean of 10 review ratings → top 5 per problem → deliverable
```

## Inputs

| Input | Source |
|---|---|
| Target paper (title + abstract) | `litreview/data/anchor_fulltext.txt` |
| Related papers — works citing the target, top 10 by abstract similarity | `limitations/data/rag_corpus.json` (OpenAlex mention search) |
| Entity corpus for the knowledge store | the above + `litreview/data/candidates_topK.json` (88 papers after de-duplication) |
| Research gaps L1–L12 and their assessment | `limitations/data/master_merged.json`, `limitations/data/dismissal_scores.json` |
| Resource constraints | `config.json` |

Earlier assignment directories are read, never modified.

## Running

```bash
python3 ideation/scripts/run_pipeline.py --estimate        # call budget: 864 at the default config
python3 ideation/scripts/run_pipeline.py                   # manual backend: writes prompts, stops
LLM_API_KEY=... python3 ideation/scripts/run_pipeline.py --backend openai   # or --backend gemini
python3 ideation/scripts/selftest.py                       # wiring test on the mock backend
```

**Manual backend** (default, no API key). Each run writes pending prompts to
`data/llm/requests/<id>.md` and lists them in `data/llm/PENDING.txt`. Put each answer in
`data/llm/responses/<id>.md` and re-run. Completed work is cached and never re-requested.

**API backends.** `--backend openai` works with any OpenAI-compatible endpoint (`LLM_BASE_URL`,
`LLM_MODEL`); `--backend gemini` uses Google's API. Set `REVIEWER_API_KEY` / `REVIEWER_MODEL`
(and optionally `REVIEWER_BASE_URL`) to have a **different model** act as the ReviewingAgents.

**Selection gate.** After stage 3 the pipeline stops (exit 3) and writes
`output/problem_candidates.md`. The team chooses 1–2 problems and records the decision in
`data/selected_problems.json`:

```json
{"selected": ["P2"], "selected_by": "team, 2026-10-01", "reason": "..."}
```

Exit codes: `0` done, `2` waiting on model responses, `3` waiting on the selection gate.

## Choosing a model (open — see `project-docs/02-decision-log.md` P-7)

A full run is **864 calls and ~4–6M input tokens**, so manual mode is impractical.

- **Local open model on the workstation** — no key, no rate limit. Serve it with vLLM or Ollama and use the
  OpenAI-compatible backend:
  ```bash
  LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=local LLM_MODEL=<served-model> \
    python3 ideation/scripts/run_pipeline.py --backend openai
  ```
- **Gemini free tier** — limits are per project and no longer published; read them in AI Studio before
  starting. Requests per day is the binding limit.

Use one model for the whole run, and preferably a different one for the reviewers (`REVIEWER_*`).

Stdlib only, per the repository convention for assignment directories.
