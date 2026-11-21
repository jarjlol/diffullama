#!/usr/bin/env bash
# queue_body_v3.sh: queue body with a GPU-fault guard (added 2026-11-21 after the 06:58 Xid 79 consumed both queues).
# Same behaviour as queue_body_pre_guard.sh (jobs from queue_gpu<N>.txt, one at a time, retried 8 times, .done log) PLUS:
#   after a job FAILS (and before the next retry), run a 5 s CUDA health check on this GPU (torch matmul, scorer venv).
#   If the check fails, or the job's output shows "CUDA unknown error|unspecified launch failure|Xid|fallen off the bus|Unable to
#   determine the device handle", the guard puts the job line BACK at the top of the queue, touches STOP_gpu<N>, writes
#   ~/codify/iter2/GPU_FAULT_gpu<N>.txt (time, job, last error lines) and the queue exits WITHOUT consuming more lines.
#   Also when nvidia-smi cannot see the GPU while waiting for it to be free.
# Test hooks (never set in production): QUEUE_DIR, QUEUE_SKIP_BUSY=1, QUEUE_FAKE_HEALTH=fail|ok, QUEUE_RETRY_SLEEP.
# usage (via queue.sh): queue.sh <gpu>
set -u
GPU=$1
D=${QUEUE_DIR:-/home/antpc/codify/iter2}; Q=$D/queue_gpu$GPU.txt; DONE=$D/queue_gpu$GPU.done; STOP=$D/STOP_gpu$GPU
FAULT=$D/GPU_FAULT_gpu$GPU.txt
P=/home/antpc/codify/diffullama/codification/pipeline
RS=${QUEUE_RETRY_SLEEP:-30}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v g=$GPU '$1==g{print $2}')
busy() {  # "busy" = not idle enough for us: load above 10% or less than 50 GB free.
  [ "${QUEUE_SKIP_BUSY:-0}" = 1 ] && return 1
  local s; s=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F", " -v g=$GPU '$1==g{print $2, $3}')
  [ -z "$s" ] && return 0
  set -- $s
  [ "${1:-100}" -gt 10 ] || [ "${2:-0}" -lt 51200 ]
}
gpu_visible() { [ "${QUEUE_SKIP_BUSY:-0}" = 1 ] && return 0; nvidia-smi -i "$GPU" >/dev/null 2>&1; }
health() {   # 0 = healthy. 5-second torch matmul on this GPU in the scorer venv.
  case "${QUEUE_FAKE_HEALTH:-}" in fail) echo "fake health check: fail"; return 1 ;; ok) return 0 ;; esac
  timeout 60 /home/antpc/venvs/scorer/bin/python -c "
import torch
x = torch.randn(2048, 2048, device='cuda'); y = (x @ x).sum().item(); torch.cuda.synchronize(); assert y == y
" 2>&1 | tail -n 2
  return ${PIPESTATUS[0]}
}
put_back_and_stop() {   # put_back_and_stop <line> <reason> <logfile-tail>
  python3 - "$Q" "$1" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
rows = open(q, encoding="utf-8").read().splitlines(keepends=True) if __import__("os").path.exists(q) else []
open(q, "w", encoding="utf-8").writelines([line + "\n"] + rows)
PY
  touch "$STOP"
  { echo "GPU fault on GPU $GPU detected $(date '+%F %T')"; echo "reason: $2"; echo "job (put back at the top of $Q): $1"; echo "--- last error lines"; printf '%s\n' "$3"; } > "$FAULT"
  echo "$(date '+%F %T') GPU FAULT: $2; job line put back at the top, $STOP touched, see $FAULT; queue ends"
}

{
echo "$(date '+%F %T') queue gpu$GPU (v3, fault guard) waiting for lane $GPU to end"
while pgrep -f "lane.sh $GPU " >/dev/null; do sleep 60; done
echo "$(date '+%F %T') lane $GPU ended; queue starts"
while true; do
  [ -e "$STOP" ] && { echo "$(date '+%F %T') STOP file present; queue ends"; exit 0; }
  line=$(grep -v '^[[:space:]]*#' "$Q" 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
  [ -z "$line" ] && { sleep 60; continue; }
  while busy; do
    gpu_visible || { put_back_and_stop "$line" "nvidia-smi cannot see GPU $GPU while waiting for it" "$(nvidia-smi -i $GPU 2>&1 | head -n 3)"; exit 0; }
    echo "$(date '+%F %T') GPU $GPU busy (another process); waiting"; sleep 120
  done
  python3 - "$Q" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
rows = open(q, encoding="utf-8").read().splitlines(keepends=True)
for i, r in enumerate(rows):
    if r.strip() == line.strip():
        del rows[i]; break
open(q, "w", encoding="utf-8").writelines(rows)
PY
  set -- $line
  if [ "$1" = SH ]; then shift; cmd=(bash "$@"); else v=$1; shift; cmd=(/home/antpc/venvs/$v/bin/python "$@"); fi
  rc=1; fault=0
  for try in 1 2 3 4 5 6 7 8; do
    [ -e "$STOP" ] && break
    echo "$(date '+%F %T') START try $try: $line"
    JOBOUT=$(mktemp)
    (cd "$P" && "${cmd[@]}") 2>&1 | tee "$JOBOUT"; rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && { echo "$(date '+%F %T') OK: $line"; rm -f "$JOBOUT"; break; }
    echo "$(date '+%F %T') FAIL try $try (exit $rc): $line"
    tailtxt=$(tail -n 40 "$JOBOUT"); rm -f "$JOBOUT"
    pat=$(printf '%s\n' "$tailtxt" | grep -E -m1 "CUDA unknown error|unspecified launch failure|Xid|fallen off the bus|Unable to determine the device handle" || true)
    h=$(health); hrc=$?
    if [ -n "$pat" ] || [ $hrc -ne 0 ]; then
      put_back_and_stop "$line" "job failed (exit $rc) and ${pat:+its output shows: $pat}${pat:+; }$([ $hrc -ne 0 ] && echo "health check failed: $h")" "$(printf '%s\n' "$tailtxt" | tail -n 8)"
      fault=1; break
    fi
    sleep "$RS"
  done
  [ $fault = 1 ] && exit 0
  echo "$(date '+%F %T') rc=$rc  $line" >> "$DONE"
done
} >> "$D/logs/queue_gpu$GPU.log" 2>&1
