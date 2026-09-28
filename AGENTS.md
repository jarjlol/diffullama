# AGENTS.md — shared instructions for all agents (opencode, Claude Code, Codex)

> Source of truth. `CLAUDE.md` imports this file. Keep under ~100 lines.
> Details live in `docs/` and `project-docs/` — load on demand, see the index below.

## Project overview

Fork of **DiffuLLaMA** (Gong et al., ICLR 2025) used as the anchor paper for a BITS F471
group research project. Two things live here: the upstream model code (converting
autoregressive LMs into diffusion LMs), and the team's own research work built around it.

Stack: Python 3.11, PyTorch 2.1.1+cu121, transformers 4.44.2. No root package manifest —
dependencies come from the vendored `LLaMA-Factory/requirements.txt`.
Entry points: `inf_diffullama.py`, `inf_diffugpt.py`, `model.py`, `attention_patch.py`.

## Commands

```sh
pip install -r LLaMA-Factory/requirements.txt          # install
pip install flash-attn==2.6.3 --no-build-isolation     # optional, recommended
cd LLaMA-Factory && pip install -e ".[torch,metrics]"  # only for training/finetuning

python inf_diffullama.py --model_name diffusionfamily/diffullama --flash_attn flash_attention_2
python inf_diffugpt.py --model_name diffusionfamily/diffugpt-s --base_model_name gpt2
```

There is **no test suite, linter, or build step** in this repo. The only test is
`audit/test_real_tokenizer.py` (stdlib + `transformers`, no GPU). Do not invent a
`npm test` / `pytest` equivalent — there isn't one.

Set `export HF_HOME=/path-to-huggingface/cache/` before running inference.

## Repo map

- `model.py`, `inf_*.py`, `attention_patch.py` — upstream DiffuLLaMA inference code
- `evaluation/` — upstream eval harness (see its own README)
- `LLaMA-Factory/`, `DiffuLLaMA-training/` — **vendored upstream forks. Do not restructure.**
- `project-docs/` — the team's durable project record (decisions, facts, bugs, open questions)
- `litreview/` — SOTA assignment: literature-review generation pipeline
- `docs/` — triggered detail files for agents (this index points at them)

## Conventions

- Python: stdlib-first. `litreview/` and `limitations/` are deliberately pure-stdlib +
  `requests` only (no numpy/sklearn) — the environment they were built in had no installer.
- Always pass `encoding='utf-8'` to `open()`. Platform default is not UTF-8 on Windows and
  this has broken this repo three times.
- Commits: `type(scope): imperative summary` (e.g. `docs(project): add repair proposal`).
- Work on branches, never commit directly to `main` without reason.

## Boundaries — Do NOT

- Do not edit `LLaMA-Factory/` or `DiffuLLaMA-training/` — vendored upstream.
- Do not force-push, hard-reset, or delete branches without explicit confirmation.
- Do not commit secrets, `.env`, model weights, or large binaries.
- Do not state a research claim as verified unless you checked the primary source. This
  project has been burned by this repeatedly — see `docs/verification-standards.md`.
- Do not preload every doc below. Follow the trigger rules.

## Additional instructions — load on demand

Read the matching file **before** working in that area. Do not preload the others.

- `docs/verification-standards.md` — how to verify claims, cite papers, avoid the recurring
  bug classes. **Triggers:** citation, paper, claim, novelty, verify, benchmark, index mapping
- `docs/research-workstreams.md` — the assignment pipelines and how they're structured.
  **Triggers:** litreview, limitations, survey, QUAL-SG, assignment, submission
- `docs/model-code.md` — upstream inference/training code and its known traps.
  **Triggers:** inference, sampling, remasking, attention, tokenizer, shift, mask token
- `project-docs/README.md` — team decision log, established facts, open questions.
  **Triggers:** decision, status, why did we, what's blocked, direction, P-1
- `README.md` / `README_SIMPLIFIED.md` — upstream usage; the latter is a from-zero
  walkthrough of the inference call path. **Triggers:** how does inference work, onboarding

## Workflow for agents (all tools)

1. Read this file plus only the triggered detail file(s) for your task.
2. Check `project-docs/02-decision-log.md` before proposing a direction — several have
   already been settled or rejected with reasons. Do not re-litigate them.
3. Verify claims against primary sources, not search summaries or abstracts.
4. If you change behavior, commands, or a settled decision, update this file and the
   relevant doc in the same change.
5. State plainly what you could not verify rather than presenting it as done.

## External File Loading (opencode lazy-load contract)

CRITICAL: When you encounter a file reference above (e.g. `docs/model-code.md`), load it
with the Read tool **only if** the current task matches its Triggers. Do NOT preload all
references. When loaded, treat the content as mandatory; follow nested references only if
the task needs them.
