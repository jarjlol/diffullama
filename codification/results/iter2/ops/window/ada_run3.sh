#!/usr/bin/env bash
# ada_run3.sh : LLaDA-8B-Base K=256 on Ada, walked from the BACK of the pre-declared random order (Blackwell took the front), until a fixed CLOCK
# time (ADA_UNTIL, default Tue 24 Nov 2026 06:00 IST) or a STOP file. Stopping is by clock only, never by results (k256_preregistration.md, Clarifications).
# Same rules as ada_run2.sh: starts only after the needed MiB stay free 10 minutes in a row, yields at once if others need the memory (< 1000 MiB free),
# holds no GPU memory while waiting, resumable by uid, touches nothing outside ~/codify-ada. Stop with: touch ~/codify-ada/ss/STOP
D=$HOME/codify-ada/ss; LOG=$D/logs/ada_run3.log
UNTIL=$(TZ=Asia/Kolkata date -d "${ADA_UNTIL:-2026-11-24 06:00}" +%s)
export HF_HOME=$HOME/codify-ada/hf HF_HUB_OFFLINE=1 CODIFY_WORK_DIR=$D/work PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 CODIFY_SAMPLE_CHUNK=${ADA_CHUNK:-2}
PY=$HOME/codify-ada/venv-llada/bin/python
M=llada-base; out=$D/work/candidates/$M@k256/test.jsonl
cd $D/codification/pipeline
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
rows() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
late() { [ "$(date +%s)" -ge "$UNTIL" ]; }
wait_room() {
  need=$1; ok=0
  while [ $ok -lt 10 ]; do
    [ -e $D/STOP ] && return 1
    late && return 1
    f=$(free_mib)
    if [ "${f:-0}" -ge "$need" ]; then ok=$((ok+1)); log "room: $f MiB free ($ok/10, need $need)"; else ok=0; log "waiting: only ${f:-?} MiB free (need $need)"; fi
    sleep 60
  done
}
guarded() {
  lf=$1; shift
  "$@" >> $lf 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    [ -e $D/STOP ] && { log "STOP file: stopping my process $pid"; kill $pid; wait $pid 2>/dev/null; return 1; }
    late && { log "clock cutoff reached: stopping my process $pid"; kill $pid; wait $pid 2>/dev/null; return 0; }
    f=$(free_mib)
    if [ "${f:-99999}" -lt 1000 ]; then log "YIELD: only $f MiB free, stopping my own process $pid"; kill $pid; wait $pid 2>/dev/null; return 2; fi
  done
  wait $pid; return $?
}
tries=0
while ! late; do
  [ -e $D/STOP ] && { log "STOP file, ending"; exit 0; }
  tries=$((tries+1)); [ $tries -gt 80 ] && { log "too many tries, giving up"; exit 0; }
  wait_room ${ADA_K256_MIB:-24000} || { log "no room / stop / cutoff, ending"; exit 0; }
  log "START generate $M K=256 --reverse chunk=$CODIFY_SAMPLE_CHUNK (have $(rows $out))"
  guarded $D/logs/gen_${M}_k256.log $PY generate_candidates.py --model $M --split test --steps 256 --order $D/k256_order.txt --reverse; log "job process ended rc=$?, rows now $(rows $out)"
  sleep 30
done
log "clock cutoff reached, ending with $(rows $out) rows"
