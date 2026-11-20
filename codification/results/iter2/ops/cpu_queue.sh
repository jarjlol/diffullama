#!/usr/bin/env bash
# cpu_queue.sh [--once]: run CPU-only analysis jobs as soon as their inputs exist. Never touches a GPU.
#
#   queue file:  ~/codify/iter2/cpu_queue.txt   one job per line:   name|spec|command
#   spec         comma-separated conditions, ALL must hold ('-' = none). Paths are relative to the work dir
#                (~/codify/diffullama/codification/pipeline/work):
#                  file:<path>          the file exists
#                  lines:<path>:<N>     the file has at least N lines
#                  lane:<g>             ~/codify/iter2/logs/gpu<g>.log contains "lane gpu<g> DONE"
#   command      run with bash in ~/codify/worker/codification/pipeline, CODIFY_WORK_DIR=<work dir>,
#                CUDA_VISIBLE_DEVICES="" , under `nice -n 10`. $WORK and $OUT (the job's result file stem) are set.
#   output       ~/codify/worker/codification/results/iter2/cpu/<name>.txt : header, the exact command, raw output, exit code.
#   log          ~/codify/iter2/logs/cpu_queue.log      started jobs move to ~/codify/iter2/cpu_queue.done
#   failure      a failing job goes back to the end of the queue, up to 3 tries, then it is marked GAVE UP in the .done file.
#   anyone may add or edit PENDING lines at any time. Stop with: touch ~/codify/iter2/STOP_cpu
#   --once       one pass, print why each job waits (dry run), run nothing that is not ready, then exit.
set -u
D=~/codify/iter2; Q=$D/cpu_queue.txt; DONE=$D/cpu_queue.done; LOG=$D/logs/cpu_queue.log
W=~/codify/diffullama/codification/pipeline/work
P=~/codify/worker/codification/pipeline
RES=~/codify/worker/codification/results/iter2/cpu
mkdir -p "$RES" "$D/logs" "$D/cpu_tries"
ONCE=0; [ "${1:-}" = "--once" ] && ONCE=1
exec 9>"$D/.cpu_queue.lock"
flock -n 9 || { echo "cpu_queue.sh already running"; exit 0; }
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG" >/dev/null; [ $ONCE = 1 ] && echo "$*"; true; }

missing() {   # prints the first unmet condition, empty if all met
  local spec=$1 c
  [ "$spec" = "-" ] && return 0
  IFS=',' read -ra conds <<< "$spec"
  for c in "${conds[@]}"; do
    case "$c" in
      file:*)  [ -e "$W/${c#file:}" ] || { echo "$c"; return; } ;;
      lines:*) rest=${c#lines:}; f=${rest%:*}; n=${rest##*:}
               have=$( [ -f "$W/$f" ] && wc -l < "$W/$f" || echo 0 )
               [ "$have" -ge "$n" ] || { echo "$c (have $have)"; return; } ;;
      lane:*)  g=${c#lane:}; grep -q "lane gpu$g DONE" "$D/logs/gpu$g.log" 2>/dev/null || { echo "$c"; return; } ;;
      *)       echo "bad condition $c"; return ;;
    esac
  done
}

log "cpu_queue start (once=$ONCE) pid $$"
while true; do
  [ -e "$D/STOP_cpu" ] && { log "STOP_cpu present, exit"; exit 0; }
  started=0
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    name=${line%%|*}; rest=${line#*|}; spec=${rest%%|*}; cmd=${rest#*|}
    m=$(missing "$spec")
    if [ -n "$m" ]; then [ $ONCE = 1 ] && echo "WAIT  $name  <- $m"; continue; fi
    [ $ONCE = 1 ] && { echo "READY $name (dry run: not started)"; continue; }
    # take the line out of the queue, then run it
    python3 - "$Q" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
rows = open(q, encoding="utf-8").read().splitlines(keepends=True)
for i, r in enumerate(rows):
    if r.rstrip("\n") == line:
        del rows[i]; break
open(q, "w", encoding="utf-8").writelines(rows)
PY
    tries=$(( $(cat "$D/cpu_tries/$name" 2>/dev/null || echo 0) + 1 )); echo $tries > "$D/cpu_tries/$name"
    out="$RES/$name.txt"
    log "START $name (try $tries): $cmd"
    {
      echo "# job $name   started $(date '+%F %T')   worker commit $(git -C ~/codify/worker rev-parse --short HEAD)   try $tries"
      echo "# inputs: $spec"
      echo "\$ cd ~/codify/worker/codification/pipeline && $cmd"
      echo
    } > "$out"
    ( cd "$P" && export CODIFY_WORK_DIR="$W" CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 WORK="$W" OUT="$RES/$name"
      nice -n 10 bash -c "$cmd" ) >> "$out" 2>&1
    rc=$?
    { echo; echo "# exit code $rc, finished $(date '+%F %T')"; } >> "$out"
    if [ $rc -eq 0 ]; then
      log "OK    $name"; echo "$(date '+%F %T') rc=0 $name" >> "$DONE"
    elif [ $tries -lt 3 ]; then
      log "FAIL  $name rc=$rc (try $tries), re-queued"; echo "$line" >> "$Q"
    else
      log "GAVE UP $name rc=$rc after $tries tries"; echo "$(date '+%F %T') GAVE UP rc=$rc $name" >> "$DONE"
    fi
    started=1
  done < <(cat "$Q" 2>/dev/null)
  [ $ONCE = 1 ] && exit 0
  [ $started = 1 ] || sleep 120
done
