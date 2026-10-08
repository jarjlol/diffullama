#!/usr/bin/env bash
# Daily driver for the ideation run. Uses one day's free OpenRouter quota, saves progress,
# and stops. Safe to run by hand at any time; the systemd timer installed by
# install_daily_timer.sh runs it at 05:45 IST, just after the quota resets.
#
#   ideation/scripts/daily_run.sh            # run now
#
# What it does
#   - refuses to run unless the repo is on assignment/research-ideation (never switches branch)
#   - reads keys from ~/.ideation-env (must be mode 600); never prints them
#   - runs the pipeline; every finished call is cached, so stopping loses at most one call
#   - after each attempt, commits ONLY ideation/data and ideation/output, scans the staged diff
#     for API keys, and pushes (never force-pushes; a rejected push just stays local)
#   - stops for the day on KEY_EXHAUSTED; retries transient failures every 10 min;
#     stops and asks for a human on a retired model, a parse error, or a missing key
#   - holds an idle-sleep inhibitor while running (a closed lid still sleeps the machine)
#
# Logs: ~/.local/state/ideation/run-YYYY-MM-DD.log     Done marker: ~/.local/state/ideation/DONE
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH="assignment/research-ideation"
ENV_FILE="${IDEATION_ENV_FILE:-$HOME/.ideation-env}"
STATE="${IDEATION_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ideation}"
BACKEND="${IDEATION_BACKEND:-openai}"          # overridable only for testing (mock)
RETRY_SLEEP="${IDEATION_RETRY_SLEEP:-600}"
MAX_HOURS="${IDEATION_MAX_HOURS:-18}"         # give up on transient errors before tomorrow's run

mkdir -p "$STATE"
LOG="$STATE/run-$(date +%F).log"
exec > >(tee -a "$LOG") 2>&1
say()    { echo "[$(date '+%F %T')] $*"; }
notify() { command -v notify-send >/dev/null && notify-send "Ideation run" "$*" 2>/dev/null; say "$*"; }

exec 9>"$STATE/lock"
flock -n 9 || { say "another run is already in progress; exiting"; exit 0; }

if [[ -f "$STATE/DONE" ]]; then say "run already complete ($(cat "$STATE/DONE")); nothing to do"; exit 0; fi

cd "$REPO" || exit 1
find ideation/data -name "*.partial" -delete 2>/dev/null   # debris from a write interrupted by shutdown
cur="$(git rev-parse --abbrev-ref HEAD)"
[[ "$cur" == "$BRANCH" ]] || { notify "NOT RUN: repo is on '$cur', expected '$BRANCH'. Switch branch yourself, then re-run."; exit 1; }

if [[ "$BACKEND" != "mock" ]]; then
  [[ -f "$ENV_FILE" ]] || { notify "NOT RUN: $ENV_FILE is missing"; exit 1; }
  [[ "$(stat -c %a "$ENV_FILE")" == "600" ]] || { notify "NOT RUN: $ENV_FILE must be mode 600 (chmod 600 $ENV_FILE)"; exit 1; }
  set +x
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

commit_progress() {
  local paths=(ideation/data ideation/output)
  git add -- "${paths[@]}" 2>/dev/null
  git diff --cached --quiet -- "${paths[@]}" && return 0
  if git diff --cached -- "${paths[@]}" | grep -qE 'sk-or-v1-[0-9a-f]{20,}|sk-[A-Za-z0-9]{32,}'; then
    git reset -q -- "${paths[@]}"
    notify "SECRET-LIKE STRING in staged files; nothing committed. Inspect ideation/data before continuing."
    return 1
  fi
  local n; n=$(ls ideation/data/llm/responses 2>/dev/null | wc -l)
  # pathspec-limited commit: anything else you have staged is left alone
  git commit -q -m "chore(ideation): run progress, $n cached responses" \
                -m "Automated commit by ideation/scripts/daily_run.sh." -- "${paths[@]}" \
    && say "committed progress ($n cached responses)"
  [[ "$BACKEND" == "mock" ]] && return 0
  git push -q origin "$BRANCH" 2>/dev/null && say "pushed" \
    || say "push failed (offline, or remote has new commits) — progress is committed locally; pull and push by hand"
}

inhibit=()
command -v systemd-inhibit >/dev/null && systemd-inhibit --what=idle --who=ideation --why=test true 2>/dev/null \
  && inhibit=(systemd-inhibit --what=idle --who=ideation --why="ideation pipeline run")

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
attempt=0
say "=== start: backend=$BACKEND branch=$BRANCH ==="
while :; do
  attempt=$((attempt + 1))
  out="$(mktemp)"
  "${inhibit[@]}" python3 ideation/scripts/run_pipeline.py --backend "$BACKEND" 2>&1 | tee "$out"
  code=${PIPESTATUS[0]}
  commit_progress
  say "attempt $attempt exit=$code"

  if [[ $code -eq 0 ]]; then
    date '+%F %T' > "$STATE/DONE"
    notify "DONE — deliverable at ideation/output/research_problems_and_ideas.md"
    rm -f "$out"; exit 0
  elif grep -q "KEY_EXHAUSTED" "$out"; then
    notify "Today's quota is used up on all keys. Will resume at the next 05:45 IST run."
    rm -f "$out"; exit 0
  elif grep -qE "unavailable for free|HTTP 404|ParseError|is not set|HTTP 401|bad selection" "$out"; then
    notify "NEEDS A HUMAN: $(grep -m1 -oE 'unavailable for free.*|HTTP 40[14].*|ParseError.*|[A-Z_]+ is not set|bad selection.*' "$out" | cut -c1-160). See SERVER_RUNBOOK.md §5."
    rm -f "$out"; exit 1
  fi
  rm -f "$out"
  if (( $(date +%s) + RETRY_SLEEP > deadline )); then
    notify "Stopping after ${MAX_HOURS}h of transient errors; next scheduled run will resume."; exit 0
  fi
  say "transient failure; retrying in $((RETRY_SLEEP / 60)) min"
  sleep "$RETRY_SLEEP"
done
