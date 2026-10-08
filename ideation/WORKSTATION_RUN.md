# WORKSTATION RUN — finish the ideation pipeline on locally hosted models

**Audience:** the teammate with access to the lab workstation (RTX 6000 Pro Blackwell, 96 GB), and
the coding agent (Claude Code or similar) working on their behalf. Written so that an agent with no
prior context can carry this out safely. **Read all of it before running anything.**

**Branch:** `assignment/research-ideation`. **Directory:** `ideation/`. Methodology and deviations:
[`REPORT.md`](REPORT.md). Project-wide rules for agents: [`../AGENTS.md`](../AGENTS.md).

---

## 0. TL;DR

```bash
git clone https://github.com/jarjlol/diffullama && cd diffullama
git checkout assignment/research-ideation && git pull
python3 ideation/scripts/selftest.py                 # expect 10/10, no GPU, no network

# vLLM in its own venv (do NOT install into the repo's torch 2.1 environment) — §3
ideation/local/serve_models.sh                       # §5: starts generator :8000 + reviewer :8001
cp ideation/local/local.env.example ~/.ideation-local-env && chmod 600 ~/.ideation-local-env
python3 ideation/local/probe_models.py               # §7: must print "verdict: PASS"
IDEATION_ENV_FILE=~/.ideation-local-env ideation/scripts/daily_run.sh   # §8: ~2-4 h, commits + pushes
ideation/local/stop_models.sh                        # §9: free the GPU
```

---

## 1. What this is, and where it stands

`ideation/` implements **ResearchAgent** (Baek et al., arXiv:2404.07738) for a course assignment:
derive 1–2 research problems from the prior assignment's gaps, then generate, review, refine and rank
research ideas, submitting the top 5 per problem.

| Stage | State |
|---|---|
| 1–2 literature + entity store | done (deterministic, re-runs instantly) |
| 3 problem discovery (4 candidates) | **done, cached** — do not regenerate |
| 4 selection gate | **done: P1 and P2** (`data/selected_problems.json`, decision log D-2026-10-08-a) |
| 5 ideas: 8 methods + 8 experiments per problem, 3 refinement rounds | **to do — this run.** 768 model calls |
| 6 rank + render top 5 per problem | automatic at the end |

**Why a local run.** The first attempt used free OpenRouter models. It was abandoned: free tier is
**50 requests/day per account** (OpenRouter's own `X-RateLimit-Limit` header), and free model slugs were
retired mid-run three times. On the workstation the whole idea stage is a few hours with no limits.

**Why the idea stage starts from zero.** The LLM cache is keyed by prompt text, *not* by model. The
OpenRouter idea-stage answers (P1, 114 steps) were moved to
`data/llm/archive/openrouter-2026-10/` so this run cannot silently reuse them: one generator and
one reviewer for every idea keeps the ranking a fair comparison. The archive is also the reference
data `probe_models.py` uses.

---

## 2. Models

Two models served at once on one 96 GB card: a **generator** (writes problems, methods, experiments)
and a **different reviewer** (the five ReviewingAgent criteria). The paper uses one model for both;
separating them avoids a model grading its own writing.

**Recommended (default in `serve_models.sh`):**

| Role | Model | Weights | Licence | Why |
|---|---|---|---|---|
| Generator | `Qwen/Qwen3.8-27B-FP8` | ~31 GB | Apache-2.0 | Newest strong dense model that fits beside a reviewer (Aug 2026); FP8 runs natively on Blackwell |
| Reviewer | `RedHatAI/gemma-4-31B-it-FP8-block` | ~33 GB | Apache-2.0 | Different family (Google vs Alibaba), so less tendency to prefer the generator's own style |

~64 GB of weights leaves ~25 GB for KV cache at `--gpu-memory-utilization 0.45` each.

**Alternatives,** all checked to exist on Hugging Face (2026-10-08), sizes are on-disk weights:

| Pairing | Fits? | Trade-off |
|---|---|---|
| `openai/gpt-oss-120b` (~61 GB MXFP4) + `openai/gpt-oss-20b` (~13 GB) | tight (~78 GB + KV) | bigger generator, but older (2025) and the same family for both roles |
| `nvidia/Gemma-4-31B-IT-NVFP4` instead of the FP8 reviewer | yes, smaller | NVFP4 is Blackwell-native; slightly more quantization |
| `google/gemma-4-26B-A4B-it` as reviewer | yes, fast | mixture-of-experts, ~4B active — faster, likely weaker judge |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (the OpenRouter run's generator) | **no** | ~80 GB alone; no room for a reviewer |
| `nvidia/Mistral-Medium-3.5-128B-NVFP4`, `zai-org/GLM-5.3-Flash`, `inclusionAI/Ling-3.0-flash` | **no** | 95–330 GB |

Quality claims above are not benchmarked by us. **The probe (§7) is the actual test**: it measures
whether the reviewer agrees with the original run's reviewer and is discerning. To try another
pairing, set `GEN_MODEL` / `REV_MODEL` (and `GEN_REASONING_PARSER` / `REV_REASONING_PARSER`) when
running `serve_models.sh`.

---

## 3. Software prerequisites

- **Python 3.11+** for the pipeline. It is stdlib-only: no `pip install` for the pipeline itself.
- **vLLM, recent enough for Blackwell (sm_120) and for these 2026 models** — use the latest release,
  with a CUDA 12.8+ PyTorch build. Install it in **its own virtualenv**, never into the repo's pinned
  `torch 2.1.1` environment:
  ```bash
  python3 -m venv ~/venvs/vllm && ~/venvs/vllm/bin/pip install -U vllm
  export VLLM=~/venvs/vllm/bin/vllm        # serve_models.sh uses $VLLM
  ```
  (or `uv venv` / `uv pip install vllm` if uv is available). Verify: `$VLLM --version`.
- **Disk:** ~65 GB for the two recommended models. Set `HF_HOME` to a disk with room; the repo's
  `AGENTS.md` asks for `export HF_HOME=/path-to-huggingface/cache/`.
- **Git:** push access to `jarjlol/diffullama` and a configured `user.name` / `user.email` — the run
  commits and pushes its progress.

---

## 4. Claim the GPU

The workstation is **shared**. Before starting, check `nvidia-smi`. The two servers hold ~86 GB for
the duration (~2–4 h). Tell whoever else uses the card. `serve_models.sh` refuses to start without
enough free memory and lists what is using it. **Never kill another person's process to free memory.**

---

## 5. Start the models

```bash
ideation/local/serve_models.sh
```

Starts the generator on `127.0.0.1:8000` and then the reviewer on `127.0.0.1:8001`, one after the
other: each profiles free memory at startup, so starting them together makes them mis-measure. Both
bind **localhost only**. Served names are fixed (`ideation-gen`, `ideation-rev`), so the env file never
changes when you switch models. First start downloads weights. Logs and PIDs:
`~/.local/state/ideation/vllm-*.{log,pid}`.

If vLLM rejects `--reasoning-parser qwen3` (flag names change between releases), restart with
`GEN_REASONING_PARSER=`. That is safe: `scripts/llm.py` strips `<think>…</think>` blocks itself,
so a rating written inside a model's private reasoning is never parsed as its answer.

---

## 6. Env file

```bash
cp ideation/local/local.env.example ~/.ideation-local-env
chmod 600 ~/.ideation-local-env
```

It contains no secrets: the servers run without `--api-key`, so the key is the placeholder `local`.
`daily_run.sh` still insists on mode 600, because the same script is used with real API keys.

---

## 7. Probe — gate before the full run

```bash
python3 ideation/local/probe_models.py
```

It writes nothing to the cache. It sends one real archived method-generation prompt to the generator,
and 12 real archived review prompts, spread across the original ratings, to the reviewer. **PASS**
requires all four:

1. the generator's output parses as `Method:` + `Rationale:`;
2. every reviewer output parses;
3. ≥70% of reviewer ratings are within 1 point of the original run's reviewer;
4. reviewer ratings are not all 4–5.

The result, including the real model ids and timings, is saved to `data/local_probe.json`. **Commit it**
— it is the evidence for the model choice.

**If it FAILS:** read which check failed. A parse failure usually means a new output format: extend the
parser in `scripts/research_agent.py` and keep `selftest.py` at 10/10. Do *not* loosen the rating
checks. Low agreement or all-high ratings means try another reviewer from §2. If the generator is very
slow (minutes per call), uncomment `LLM_EXTRA_BODY` in the env file to disable its thinking mode, and
re-probe.

---

## 8. Run

```bash
IDEATION_ENV_FILE=~/.ideation-local-env ideation/scripts/daily_run.sh
```

- Runs the pipeline to completion. Every finished call is cached (atomically) at once, so it can be
  stopped and re-run at any time and loses at most the call in flight.
- After each attempt it commits **only** `ideation/data` and `ideation/output` (after scanning for
  key-like strings) and pushes. It never force-pushes. Other staged work is left alone.
- Connection errors (server not up yet) are retried every 10 minutes.
- Expect **~2–4 hours** for 768 calls: ~128 generations and ~640 reviews. This is an estimate, not
  measured on this hardware; `probe_models.py` prints real seconds per call.
- Monitor: `tail -f ~/.local/state/ideation/run-$(date +%F).log` and
  `ls ideation/data/llm/responses | wc -l`, which climbs by 768 from its starting value.
- Exit 0 means done: `ideation/output/research_problems_and_ideas.md` exists.

Running `scripts/run_pipeline.py --backend openai` directly also works (after
`source ~/.ideation-local-env`), but it does not commit or push.

---

## 9. After it finishes

1. `ideation/local/stop_models.sh` — free the GPU.
2. `python3 ideation/scripts/selftest.py` — 10/10.
3. Sanity-check the deliverable: P1 and P2 each have 8 ranked ideas and 5 rendered in full; ratings
   spread across 1–5 rather than clustering at 4–5.
4. Update `REPORT.md` §4: the actual model ids (from `data/local_probe.json`), call count, wall-clock
   time, and anything that went wrong. Commit and push.
5. Tell the team. **Remaining human task:** a manual novelty check of every submitted idea. The
   pipeline's Originality and Innovativeness ratings are the reviewer's opinion, with no literature
   search. Start with the anchor paper's own authors' later work — both preemptions this project has
   hit came from there (e.g. DiffuCoder, arXiv:2506.20639, already measured AR-ness and pass@k on
   adapted diffusion models).

---

## 10. Failure modes

| Symptom | Meaning | Fix |
|---|---|---|
| `serve_models.sh`: "not enough free GPU memory" | someone else is using the card | wait or coordinate; or lower `GEN_UTIL`/`REV_UTIL`, which shrinks the KV cache |
| vLLM log: unknown argument `--reasoning-parser` | flag differs in this vLLM version | `GEN_REASONING_PARSER= ideation/local/serve_models.sh` |
| vLLM log: architecture not supported | vLLM too old for the model | upgrade vLLM in its venv |
| vLLM log: CUDA out of memory at startup | utilization fractions too high for what is free | lower `GEN_UTIL`/`REV_UTIL`, or `MAX_LEN=16384` (prompts are ≤8k tokens) |
| run log: `ParseError ... expected 'Method:'` | model used a new header format | extend the parser in `scripts/research_agent.py`; selftest must stay 10/10 |
| run log: `TimeoutError` / `timed out` | generation slower than `LLM_TIMEOUT` (900 s) | raise `LLM_TIMEOUT`, or disable thinking via `LLM_EXTRA_BODY` |
| `COMMIT FAILED` / `push failed` | git identity or remote moved | fix git, `git pull --rebase`, push by hand; progress is safe on disk |

---

## 11. Rules for agents working on this

- **Never delete anything in `ideation/data/llm/responses/`** during or after a run. Responses are
  content-addressed; regenerating one (temperature 0.7) changes every downstream prompt hash and forks
  the chain. Junk is moved to an archive with `git mv`, never deleted.
- **Do not edit** `scripts/templates.py` part 1 (the paper's verbatim prompts),
  `data/problems.json`, or `data/selected_problems.json`. Changing any of them invalidates the
  method's fidelity or the team's decisions. A change of models or settings goes in `REPORT.md`.
- **Do not change the model mid-run.** If you must, stop, archive the idea-stage cache as in §1, and
  restart, so one generator and one reviewer cover every ranked idea.
- **Never commit** model weights, `~/.ideation-local-env`, or any API key. `daily_run.sh` scans
  before committing; also run `grep -rE "sk-or-v1-[0-9a-f]{20,}" ideation/` before any manual commit.
- **Never force-push, hard-reset, or delete branches** (`AGENTS.md`).
- State plainly what you could not verify. Record deviations in `REPORT.md` and decisions in
  `project-docs/02-decision-log.md`.
