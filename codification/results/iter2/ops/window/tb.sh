#!/usr/bin/env bash
# tb.sh <gpu> <venv> <python args...>   time-boxed, polite GPU job for the shared Blackwell box (queue line:  SH .../window/tb.sh 0 dream generate_candidates.py ...).
# Rules (agreed with the senior session / krispy's pause):
#  1. WAIT (hold no GPU memory) until  (a) the file $WINDOW_END exists and holds an epoch time at least MIN_LEFT s in the future (the end of the
#     window krispy gave us; nobody sets it => nothing starts), and  (b) this GPU has been idle for IDLE_SECONDS: no compute process of anyone,
#     memory.used < IDLE_MIB.
#  2. Run the job (the next queue lines see it as a normal SH line) under a hard stop at WINDOW_END: SIGTERM at the end (background jobs ignore SIGINT in a script), SIGKILL 30 s later.
#  3. YIELD: every 20 s, if a compute process that is NOT ours appears on this GPU, stop our job at once (SIGTERM, then SIGKILL) and exit 0.
#  The jobs used with it (generate_candidates.py, self_score.py, score_candidates.py) append one row per question and resume by uid, so a stop
#  loses at most the question in flight. Exit code: 0 after a timed stop or yield (the queue must not retry into someone's window); the job's own
#  code if it fails before the end (so the queue's retry and fault guard still work).
set -u
GPU=${1:?gpu}; VENV=${2:?venv}; shift 2
WINDOW_END=${WINDOW_END:-$HOME/codify/iter2/WINDOW_END}
STOPF=${TB_STOP:-$HOME/codify/iter2/STOP_gpu$GPU}
IDLE_SECONDS=${IDLE_SECONDS:-300}; IDLE_MIB=${IDLE_MIB:-1500}; MIN_LEFT=${MIN_LEFT:-1200}
NVSMI=${NVSMI:-nvidia-smi}; PYBIN=${PYBIN:-/home/antpc/venvs/$VENV/bin/python}
[ "$VENV" = SH ] && PYBIN=bash      # tb.sh <gpu> SH script.sh args...  runs a bash wrapper under the same rules
LOG=${TB_LOG:-$HOME/codify/iter2/logs/tb_gpu$GPU.log}
log() { echo "$(date '+%F %T') [tb gpu$GPU] $*" | tee -a "$LOG"; }
mkdir -p "$(dirname "$LOG")"
foreign() {   # prints pids of compute processes on this GPU that are not in our process group $1
  local pg=$1 p
  for p in $($NVSMI --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' '); do
    [ -n "$p" ] && ! ps -o pid= -g "$pg" 2>/dev/null | tr -d ' ' | grep -qx "$p" && echo "$p"
  done
}
used_mib() { $NVSMI --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' '; }
end_epoch() { local v; v=$([ -r "$WINDOW_END" ] && tr -dc '0-9' < "$WINDOW_END" | head -c 12); [ -n "$v" ] && echo $((v - ${MARGIN:-0})); }   # MARGIN: stop this many s BEFORE WINDOW_END (leaves time for the scoring lines)

# ---- 1. wait for the window and an idle GPU (holds nothing)
log "waiting: $* (WINDOW_END file $WINDOW_END, idle ${IDLE_SECONDS}s needed)"
idle=0
while :; do
  [ -e "$STOPF" ] && { log "STOP file $STOPF: leaving the line unrun (exit 1 so the queue keeps it)"; exit 1; }
  E=$(end_epoch); now=$(date +%s)
  [ -n "$E" ] && [ "$now" -ge "$E" ] && { log "window is over (now >= WINDOW_END - margin): line skipped, exit 0"; exit 0; }
  if [ -n "$E" ] && [ $((E - now)) -ge "$MIN_LEFT" ] && [ "$(used_mib)" -lt "$IDLE_MIB" ] && [ -z "$($NVSMI --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' \n')" ]; then
    idle=$((idle + 15))
  else idle=0; fi
  [ "$idle" -ge "$IDLE_SECONDS" ] && break
  sleep 15
done
E=$(end_epoch); now=$(date +%s); left=$((E - now))
log "window open: $((left / 60)) min until WINDOW_END ($(date -d @"$E" '+%F %T')); starting $VENV $*"

# ---- 2/3. run in its own process group, with the hard stop and the yield watchdog
setsid "$PYBIN" "$@" &
JP=$!
PG=$(ps -o pgid= -p "$JP" | tr -d ' ')
stopjob() { kill -TERM -- "-$PG" 2>/dev/null; for _ in 1 2 3 4 5 6; do sleep 5; kill -0 "$JP" 2>/dev/null || return 0; done; kill -KILL -- "-$PG" 2>/dev/null; }
REASON=""
while kill -0 "$JP" 2>/dev/null; do
  sleep 20
  now=$(date +%s)
  if [ "$now" -ge "$E" ]; then REASON="WINDOW_END reached"; stopjob; break; fi
  [ -e "$STOPF" ] && { REASON="STOP file"; stopjob; break; }
  f=$(foreign "$PG" | tr '\n' ' ')
  if [ -n "$f" ]; then REASON="YIELD: foreign GPU process(es) $f appeared"; stopjob; break; fi
done
wait "$JP"; rc=$?
if [ -n "$REASON" ]; then log "stopped: $REASON (job rc=$rc); exit 0"; exit 0; fi
log "job ended rc=$rc"
exit "$rc"
