# Archived: OpenRouter idea-stage cache (2026-10)

Idea-stage (method/experiment) requests and responses from the run on free OpenRouter models:
generator `nvidia/nemotron-3-super-120b-a12b:free`, reviewer `inclusionai/ling-3.0-flash-sante:free`.
114 distinct completed steps for P1 (M1-M4 complete, M5-M8 partial) plus orphaned forks.

Moved here on 2026-10-08 when the run switched to locally hosted vLLM models
(`ideation/WORKSTATION_RUN.md`). The cache is keyed by prompt, not by model, so leaving these in
`data/llm/responses/` would have made the local run silently reuse them and mix two generators and two
reviewers in one ranking. Kept as a record and as reference ratings for `local/probe_models.py`,
which measures a candidate reviewer's agreement with this run's reviewer.

`key_state.json` here is the OpenRouter key-rotation index (indices only, no keys).
Restore with `git mv` back to `data/llm/{requests,responses}/` if ever needed.
