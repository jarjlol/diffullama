#!/usr/bin/env bash
# queue.sh <gpu>: keep GPU <gpu> busy. Waits for lane.sh <gpu> to end, then runs jobs from
# ~/codify/iter2/queue_gpu<gpu>.txt one at a time, first line first. Anyone may edit PENDING lines.
#   line format:  <venv> <script.py> <args...>     run in pipeline/ with ~/venvs/<venv>/bin/python
#                 SH <script.sh> <args...>         run with bash
#   '#' lines and blank lines are ignored.
# A started line is removed from the queue and logged to queue_gpu<gpu>.done with its result.
# Before each job it waits until no process is on this GPU (another lab user may have taken it).
# Holds no model while waiting. Stop with: touch ~/codify/iter2/STOP_gpu<gpu>
set -u
GPU=$1
D=~/codify/iter2; Q=$D/queue_gpu$GPU.txt; DONE=$D/queue_gpu$GPU.done; STOP=$D/STOP_gpu$GPU
P=~/codify/diffullama/codification/pipeline
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
busy() {  # "busy" = not idle enough for us: load above 10% or less than 50 GB free.
  # Other users may keep idle background processes on the card; those are fine to share with.
  local s; s=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits | awk -F", " -v g=$GPU '$1==g{print $2, $3}')
  set -- $s
  [ "${1:-100}" -gt 10 ] || [ "${2:-0}" -lt 51200 ]
}

{
echo "$(date '+%F %T') queue gpu$GPU waiting for lane $GPU to end"
while pgrep -f "lane.sh $GPU " >/dev/null; do sleep 60; done
echo "$(date '+%F %T') lane $GPU ended; queue starts"
while true; do
  [ -e "$STOP" ] && { echo "$(date '+%F %T') STOP file present; queue ends"; exit 0; }
  line=$(grep -v '^[[:space:]]*#' "$Q" 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
  [ -z "$line" ] && { sleep 60; continue; }
  while busy; do echo "$(date '+%F %T') GPU $GPU busy (another process); waiting"; sleep 120; done
  python3 - "$Q" "$line" <<'EOF'
import sys
q, line = sys.argv[1], sys.argv[2]
rows = open(q, encoding="utf-8").read().splitlines(keepends=True)
for i, r in enumerate(rows):
    if r.strip() == line.strip():
        del rows[i]; break
open(q, "w", encoding="utf-8").writelines(rows)
EOF
  set -- $line
  if [ "$1" = SH ]; then shift; cmd=(bash "$@"); else v=$1; shift; cmd=(~/venvs/$v/bin/python "$@"); fi
  rc=1
  for try in 1 2 3 4 5 6 7 8; do
    [ -e "$STOP" ] && break
    echo "$(date '+%F %T') START try $try: $line"
    (cd "$P" && "${cmd[@]}"); rc=$?
    [ $rc -eq 0 ] && { echo "$(date '+%F %T') OK: $line"; break; }
    echo "$(date '+%F %T') FAIL try $try (exit $rc): $line"; sleep 30
  done
  echo "$(date '+%F %T') rc=$rc  $line" >> "$DONE"
done
} >> "$D/logs/queue_gpu$GPU.log" 2>&1
