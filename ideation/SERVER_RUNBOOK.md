# SERVER RUNBOOK — ResearchAgent ideation run on OpenRouter (SUPERSEDED)

> ⚠️ **Superseded 2026-10-08 by [`WORKSTATION_RUN.md`](WORKSTATION_RUN.md)** — the run now uses
> locally hosted vLLM models. OpenRouter's free tier proved to be 50 requests/day per account, and
> free model slugs were retired mid-run three times. Kept for the record. The selection is now P1 + P2,
> and the OpenRouter idea-stage cache is archived under `data/llm/archive/openrouter-2026-10/`.

How to run the ideation pipeline on a server instead of a laptop. The run is
fully resumable: the server picks up exactly where the laptop stopped, with
zero re-spend, because every LLM response is cached in git.

## 1. What this is

`ideation/` implements **ResearchAgent** (Baek et al., arXiv:2404.07738) for the
course ideation assignment: 4 candidate research problems were generated from
the limitations assignment's gaps (stage 3, done), the team selected **P1, P2,
P3** at the selection gate (P4 dropped — MASK-token comparison, confounded),
and the pipeline is now generating 8 methods + 8 experiments per problem with
3 refinement rounds, then ranking a top 5 per problem (stages 5–6, in progress).

- Full pipeline map: [`README.md`](README.md). Methodology: [`REPORT.md`](REPORT.md).
- Call budget: **1,248 calls for P1+P2+P3** (384/problem) + 96 problem-stage = 1,344 total.
  Progress is `ls ideation/data/llm/responses/ | wc -l` (was ~308 at handover).
- Roles: **generator** = `nvidia/nemotron-3-super-120b-a12b:free` (slow, minutes/call),
  **reviewer** = `inclusionai/ling-3.0-flash-sante:free` (fast, ~10 s/call).
  Both free via OpenRouter. History: reviewer was Gemma-31B (congested) →
  Nemotron-Lightning (too slow) → Ling-fin (**went paid mid-run**) → Ling-sante.

## 2. Server prerequisites

- Any Linux server with **Python 3.11+**. No GPU needed. No `pip install` —
  the pipeline is **stdlib-only** (`urllib`, `json`, `re`).
- RAM/disk trivial (~50 MB + ~13 MB cache). Network egress to
  `https://openrouter.ai` required.
- 1–3 **OpenRouter API keys** (free tier: **50 requests/day per account** (corrected 2026-10-08 from OpenRouter's own `X-RateLimit-Limit: 50` header; the ~200 figure was an estimate) — the limit
  looks per-account, not per-key, so prefer keys from *different* accounts).

## 3. Setup

```bash
git clone https://github.com/jarjlol/diffullama
cd diffullama
git checkout assignment/research-ideation
git pull
python3 ideation/scripts/selftest.py   # expect 10/10; mock backend, no keys, ~1 min
```

Put keys in a root-owned env file (never in git, never on screen):

```bash
cat > ~/.ideation-env <<'EOF'
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_API_KEYS="KEY1,KEY2,KEY3"
export LLM_MODEL=nvidia/nemotron-3-super-120b-a12b:free
export REVIEWER_BASE_URL=https://openrouter.ai/api/v1
export REVIEWER_API_KEYS="KEY1,KEY2,KEY3"
export REVIEWER_MODEL=inclusionai/ling-3.0-flash-sante:free
EOF
chmod 600 ~/.ideation-env
source ~/.ideation-env
```

Notes: singular `LLM_API_KEY` also works; plural enables rotation.
`data/llm/key_state.json` persists rotation position (committed, no secrets in it).

## 4. Run (resumable loop)

```bash
source ~/.ideation-env
cd ~/diffullama   # or wherever you cloned
nohup bash -c 'for i in $(seq 1 200); do python3 ideation/scripts/run_pipeline.py --backend openai >> /tmp/ideation-run.log 2>&1; code=$?; echo "loop attempt $i exit=$code" >> /tmp/ideation-run.log; if [ $code -eq 0 ]; then break; fi; sleep 600; done' >/dev/null 2>&1 &
```

- Exit `0` = done → deliverable at `ideation/output/research_problems_and_ideas.md`.
- Anything else = crash/transient → loop sleeps 10 min and retries from cache.
  **Never delete files in `data/llm/responses/`**: responses are content-addressed;
  deleting + regenerating (temperature 0.7) forks the prompt chain and orphans
  downstream cache, burning quota. Only true junk (model refusals) may be removed,
  and that has already been done.
- Monitor: `ls ideation/data/llm/responses/ | wc -l` and `tail /tmp/ideation-run.log`.
- Quota wall looks like `KEY_EXHAUSTED (reviewer|generator): all N key(s)...`.
  If all keys 429 together, it is account-level or provider throttling — wait for
  the daily reset (~00:00 UTC); the loop covers ~33 h unattended.

## 4b. Alternative: run a little each day on a laptop (scheduled)

The free quota (**50 requests/day per account** (corrected 2026-10-08 from OpenRouter's own `X-RateLimit-Limit: 50` header; the ~200 figure was an estimate)) resets around 05:30 IST, so the run does not need a
machine on for days at a stretch — only for a few hours after each reset. Every finished call is
cached; switching off mid-run loses at most the call in progress.

```bash
ideation/scripts/daily_run.sh                    # run now: uses today's quota, commits + pushes, stops
ideation/scripts/install_daily_timer.sh          # schedule it at 05:45 IST daily (systemd user timer)
ideation/scripts/install_daily_timer.sh --uninstall
```

`daily_run.sh` refuses to run off `assignment/research-ideation` or with a key file that is not mode
600; reads keys from `~/.ideation-env` without printing them; commits **only** `ideation/data` and
`ideation/output`, after scanning the staged diff for key-like strings; pushes without ever forcing;
stops for the day on `KEY_EXHAUSTED`; retries transient errors every 10 min (up to 18 h); and stops with
a desktop notification when a human is needed (retired model, parse error, missing key). If the
laptop is off or asleep at 05:45, the timer fires when it is back. A closed lid still sleeps the
machine. Logs: `~/.local/state/ideation/run-YYYY-MM-DD.log`.

## 5. Known failure modes and fixes

| Symptom in log | Meaning | Fix |
|---|---|---|
| `LLM HTTP 429 (provider congested)` mentioning "upstream" | Free-model congestion, not your quota | Auto-retried in-code (6× backoff); no action |
| `KEY_EXHAUSTED (...)` | All keys throttled | Wait for reset / add a key from another account |
| `LLM HTTP 404 ... unavailable for free` naming the model | OpenRouter retired the free slug (happened to Ling-fin) | Pick a live `:free` model (`GET openrouter.ai/api/v1/models`, filter `id` ending `:free`), probe it with one real pending prompt, verify `parse_review`/`parse_artifact` accept the output, then set `REVIEWER_MODEL` (or `LLM_MODEL`) and relaunch. Document the swap in `REPORT.md` §4 |
| `ParseError: ... expected 'Method:' and 'Rationale:'` | New section-header variant | Parser already tolerates plain/bold/`##`/`### 7.`/titled headers and `:`-less labels; extend `_label()` in `scripts/research_agent.py`, verify `selftest.py` stays 10/10 |
| `ParseError: ... no 'Rating (1-5)...'` | New rating-line variant | Parser accepts `(1-5)`/`(1–5)`/bare `Rating: N`, `###`-headed lines; extend `_RATING` likewise |
| Duplicate `<tag>-<different-hash>.md` files | Orphaned variant chains from past regenerations | Harmless. The run follows exactly one chain; do not clean mid-run |

## 6. After exit 0

1. `python3 ideation/scripts/selftest.py` → 10/10.
2. Sanity: every selected problem has 8 ranked ideas, top 5 rendered;
   rating spread should look discerning (Sante gave roughly 1×1, 24×2, 144×3, 54×4 — not uniform 4–5).
3. **Manual novelty check per submitted idea** (the method does not check novelty —
   Originality ratings are reviewer opinion, no literature search).
4. Update `REPORT.md` §4 with the final call/token counts and any new model swaps,
   then commit the new cache + deliverable and push.

## 7. Security

- API keys live **only** in `~/.ideation-env` (mode 600) and process env. They are
  never written to the repo: `key_state.json` stores rotation indices only.
  Before any `git add`, run `grep -r "sk-or-v1" ideation/ --exclude-dir=.git` and
  expect no hits. Rotate any key that was ever pasted into chat.
