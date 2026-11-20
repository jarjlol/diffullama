#!/usr/bin/env bash
# a3_window.sh <gpu> : the A3 (Arbor iteration 2) window. A queue line on EACH GPU queue:
#     SH /home/antpc/codify/iter2/a3/a3_window.sh 0      (in queue_gpu0.txt)
#     SH /home/antpc/codify/iter2/a3/a3_window.sh 1      (in queue_gpu1.txt)
# Arbor needs both GPUs (vLLM data-parallel, one replica per GPU), so the two queues rendezvous here:
#   - the first queue to arrive PARKS (holds no model, heartbeat file) for up to PARK_SECONDS (default 1500);
#   - if the other queue arrives in that time, the later one becomes the CONTROLLER and the parked one waits
#     until the window is over; both queues then continue with their next lines;
#   - if nobody arrives, the parked queue gives up and continues with its next line (a later A3 line retries).
# Every normal path exits 0, so queue.sh's 8-try retry never re-runs a half-done window. attempts are counted
# (max 3 windows), the final split is read once (state file final_eval.done).
#
# Controller steps: prerequisites (CPU export of the Arbor task, Aryan's prepare_job.py) -> both GPUs free ->
# start vLLM (~/codify/start_vllm.sh: DP=2, 131072, qwen3_xml) -> Arbor UNATTENDED (interaction_mode auto, red lines and
# protected paths come from the exported plugin) -> stop vLLM (only the process group this script started) ->
# final-split evaluation once (a3_final_eval.sh) -> markers.
# REVIEW MODE WAS NOT USED: nobody is attending, so every idea is auto-approved and no human feedback is given to Arbor.
# Dry run: A3_DRYRUN=1 (no vLLM, no Arbor, evaluates the GATE split of A3_READY); A3_STATE=<dir> for a separate state dir.
set -u
GPU=${1:?gpu 0 or 1}
OTHER=$((1 - GPU))
D=${A3_STATE:-$HOME/codify/iter2/a3}
DRY=${A3_DRYRUN:-0}
PARK=${A3_PARK_SECONDS:-1500}
mkdir -p "$D"
LOG=$D/a3.log
log() { echo "$(date '+%F %T') [gpu$GPU] $*" | tee -a "$LOG"; }
fresh() { [ -e "$1" ] && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -lt 90 ]; }
exec 7>"$D/.lock"

[ -e "$D/window.done" ] && { log "window already done, skip"; exit 0; }
[ -e "$D/window.gaveup" ] && { log "window gave up earlier (3 attempts), skip"; exit 0; }

follow() {   # wait until the controller is finished; hold no GPU memory
  log "follower: waiting for the controller to finish"
  local t0=$(date +%s)
  while [ ! -e "$D/window.done" ] && [ ! -e "$D/window.abort" ] && [ ! -e "$D/window.gaveup" ]; do
    touch "$D/parked.$GPU"; sleep 20
    [ $(( $(date +%s) - t0 )) -gt 21600 ] && { log "follower: 6 h limit, leaving"; break; }
  done
  rm -f "$D/parked.$GPU"; log "follower: released"; exit 0
}

flock 7
if [ -e "$D/window.running" ]; then flock -u 7; follow; fi
if fresh "$D/parked.$OTHER"; then
  echo "controller=$GPU $(date '+%F %T')" > "$D/window.running"; flock -u 7; ROLE=controller
else
  touch "$D/parked.$GPU"; flock -u 7; ROLE=parked
fi

if [ "$ROLE" = parked ]; then
  log "parked: waiting up to ${PARK}s for the other queue"
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$PARK" ]; do
    touch "$D/parked.$GPU"
    if [ -e "$D/window.running" ]; then follow; fi
    sleep 15
  done
  flock 7
  if [ -e "$D/window.running" ]; then flock -u 7; follow; fi
  rm -f "$D/parked.$GPU"; flock -u 7
  log "no partner within ${PARK}s: continuing with the next queue line (a later A3 line retries)"
  exit 0
fi

# ------------------------------------------------------------------ controller
ATT=$(( $(cat "$D/attempts" 2>/dev/null || echo 0) + 1 )); echo $ATT > "$D/attempts"
log "controller: attempt $ATT (dry run: $DRY)"
VLLM_PGID=""
cleanup() {
  if [ -n "$VLLM_PGID" ] && [ "$DRY" != 1 ]; then
    log "cleanup: stopping vLLM process group $VLLM_PGID (started by this script)"
    kill -- "-$VLLM_PGID" 2>/dev/null; sleep 20
    pgrep -g "$VLLM_PGID" >/dev/null 2>&1 && { kill -9 -- "-$VLLM_PGID" 2>/dev/null; sleep 5; }
  fi
  VLLM_PGID=""
}
finish() {   # finish <done|abort> <reason>
  cleanup
  log "controller: $1 ($2)"
  if [ "$1" = done ]; then touch "$D/window.done"; else echo "$2" > "$D/window.abort"; [ "$ATT" -ge 3 ] && touch "$D/window.gaveup"; fi
  sleep 60                       # let the follower see the marker
  rm -f "$D/window.running" "$D/parked.0" "$D/parked.1"; [ "$1" = abort ] && rm -f "$D/window.abort"
  exit 0
}
trap 'finish abort "controller interrupted"' INT TERM

[ "$ATT" -gt 3 ] && finish abort "more than 3 attempts"

# paper results freeze: Sun 22 Nov 2026 18:00 IST. A window that STARTS after 14:00 IST that day cannot finish in time (vLLM load,
# Arbor, evaluation take 2-4 h), so its outputs are labelled post-freeze, report only. The window still runs.
NOW=${A3_NOW_EPOCH:-$(date +%s)}
POST=""
if [ "$NOW" -ge "$(date -d '2026-11-22 14:00 IST' +%s)" ]; then
  POST="_POST_FREEZE_REPORT_ONLY"
  echo "window started $(date -d @$NOW '+%F %T') IST, after Sun 22 Nov 14:00 IST: results are POST-FREEZE, REPORT ONLY (not for the paper)" | tee "$D/POST_FREEZE_REPORT_ONLY.txt" | tee -a "$LOG"
fi

# 1. prerequisites: the Arbor task export (CPU; reuses a verified export, refuses incomplete features)
if [ "$DRY" = 1 ] && [ -n "${A3_READY:-}" ]; then READY=$A3_READY; log "dry run: using READY=$READY"
else
  python3 "$HOME/codify/iter2/aryan/codification/a3_support/prepare_job.py" >> "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && finish abort "prepare_job.py rc=$rc (features or export not ready)"
  READY=$(ls -dt "$HOME"/arbor-rerank-iter2-ready-* 2>/dev/null | head -1)
fi
for f in INSTRUCTION.md solution.py research_config.yaml eval.sh data/test/items.jsonl; do
  [ -e "$READY/$f" ] || finish abort "export $READY lacks $f"
done
[ "$DRY" != 1 ] && [ -n "$(git -C "$READY" status --porcelain)" ] && finish abort "export $READY has uncommitted changes"
log "export: $READY at $(git -C "$READY" rev-parse --short HEAD 2>/dev/null)"
cp "$READY/solution.py" "$D/seed_solution.py"

# 2. both GPUs free enough for vLLM (0.85 x memory each)
if [ "$DRY" != 1 ]; then
  t0=$(date +%s); ok=0
  while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do
    ok=1
    for g in 0 1; do
      s=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits -i $g | tr -d ' ')
      u=${s%,*}; fm=${s#*,}
      { [ "$u" -le 10 ] && [ "$fm" -ge 85000 ]; } || ok=0
    done
    [ $ok = 1 ] && break; sleep 30
  done
  [ $ok = 1 ] || finish abort "GPUs not free (util <= 10% and >= 85 GB free each) within 20 min"
  (ss -ltn 2>/dev/null | grep -q ':8000 ') && finish abort "port 8000 already in use (not ours)"
fi

# 3. vLLM (DP = 2, 131072, qwen3_xml), started here and stopped here
if [ "$DRY" = 1 ]; then log "dry run: would start ~/codify/start_vllm.sh and wait for /v1/models"
else
  log "starting vLLM"
  bash "$HOME/codify/start_vllm.sh"; sleep 5
  VPID=$(cat "$HOME/codify/vllm.pid" 2>/dev/null)
  grep -q "vllm" "/proc/$VPID/cmdline" 2>/dev/null || finish abort "vllm pid file does not point at a vllm process"
  VLLM_PGID=$(ps -o pgid= -p "$VPID" | tr -d ' ')
  t0=$(date +%s); up=0
  while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do
    curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$VPID" 2>/dev/null || break; sleep 20
  done
  [ $up = 1 ] || finish abort "vLLM did not come up in 20 min (see ~/codify/vllm.log)"
  log "vLLM up after $(( $(date +%s) - t0 )) s"
fi

# 4. Arbor, unattended: a COPY of the config with interaction_mode auto (the exported repo stays untouched)
sed 's/^\([[:space:]]*interaction_mode:\).*/\1 auto/' "$READY/research_config.yaml" > "$D/arbor_auto.yaml"
grep -n "interaction_mode" "$D/arbor_auto.yaml" | tee -a "$LOG"
echo "review mode not used: no human attends this run (decided by the senior session / Aaditya, 2026-11-21)" > "$D/REVIEW_MODE_NOT_USED.txt"
RUN="iter2-auto-$(date +%m%d-%H%M)"
if [ "$DRY" = 1 ]; then
  log "dry run: would run arbor run ... --run-name $RUN --config $D/arbor_auto.yaml (timeout 4h) in $READY"
else
  log "arbor run $RUN"
  ( cd "$READY" && HF_HUB_OFFLINE=1 OPENAI_API_KEY=dummy timeout 4h "$HOME/venvs/arbor/bin/arbor" run "$(cat INSTRUCTION.md)" \
      --yes --yes-cwd "$READY" --config "$D/arbor_auto.yaml" --run-name "$RUN" ) > "$D/arbor_$RUN.log" 2>&1
  log "arbor exit code $?  (log $D/arbor_$RUN.log)"
fi

# 5. stop vLLM now (do not hold the GPUs for the evaluation)
cleanup

# 6. did Arbor merge anything? then score it once on the final split (CPU); the final split is read only once
if cmp -s "$READY/solution.py" "$D/seed_solution.py" && [ "${A3_DRY_FORCE_EVAL:-0}" != 1 ]; then
  log "solution.py unchanged: nothing merged, final split NOT read"
  echo "no merge" > "$D/final_eval.skipped"
elif [ -e "$D/final_eval.done" ] && [ "$DRY" != 1 ]; then
  log "final evaluation already done earlier, not repeated"
else
  WHICH=final; [ "$DRY" = 1 ] && WHICH=gate
  OUTD="$HOME/codify/worker/codification/results/iter2/cpu/a3_${WHICH}_eval${POST}"
  [ "$DRY" = 1 ] && OUTD="$D/dry_eval"
  log "evaluating merged solution.py on the $WHICH split -> $OUTD"
  { [ -n "$POST" ] && echo "### POST-FREEZE, REPORT ONLY: this window started after Sun 22 Nov 2026 14:00 IST; do not put these numbers in the paper";     bash "$HOME/codify/iter2/a3/a3_final_eval.sh" "$READY" "$WHICH" "$OUTD"; } > "$OUTD.txt" 2>&1
  log "evaluation exit code $? (output $OUTD.txt)"
  [ "$DRY" != 1 ] && date > "$D/final_eval.done"
fi
finish done "attempt $ATT complete"
