#!/usr/bin/env bash
# ada_run2.sh : replaces ada_run.sh (2026-11-22 revision, plan with Main). Runs on ada ONLY when the GPU has room, yields when another user needs the memory.
# Rules (unchanged, agreed with the senior session): start only after the needed MiB stay free for 10 consecutive minutes; while running, stop OUR
# process if less than 1000 MiB would remain for the others (they always win). Resumable by question id. Nothing outside ~/codify-ada.
# Stop with: touch ~/codify-ada/ss/STOP        Jobs, in priority order:
#   1. LLaDA self-score on test (500 rows; 273 came from Blackwell)                 -> completes test-500
#   2. Dream K = 256 generation, test, in the pre-declared random order (k256_order.txt) walked from the BACK, until ADA_K256_MAX rows or STOP.
#      Blackwell works the same split forwards; rows are merged by uid later (candidates/dream-base@k256/test.ext.jsonl there).
# The dev self-score jobs of the first runner are dropped (dev adds nothing to the paper tables).
D=$HOME/codify-ada/ss; LOG=$D/logs/ada_run.log
export HF_HOME=$HOME/codify-ada/hf HF_HUB_OFFLINE=1 CODIFY_WORK_DIR=$D/work PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 CODIFY_SAMPLE_CHUNK=${ADA_CHUNK:-5}
PY=$HOME/codify-ada/venv-dream/bin/python
MAXK=${ADA_K256_MAX:-60}
cd $D/codification/pipeline
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
rows() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
wait_room() {   # wait_room <MiB>: 10 consecutive 60 s checks with >= MiB free
  need=$1; ok=0
  while [ $ok -lt 10 ]; do
    [ -e $D/STOP ] && return 1
    f=$(free_mib)
    if [ "${f:-0}" -ge "$need" ]; then ok=$((ok+1)); log "room: $f MiB free ($ok/10, need $need)"; else ok=0; log "waiting: only ${f:-?} MiB free (need $need)"; fi
    sleep 60
  done
}
guarded() {   # guarded <logfile> <cmd...> : run in background, stop it when others need the memory
  lf=$1; shift
  "$@" >> $lf 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    [ -e $D/STOP ] && { log "STOP file: stopping my process $pid"; kill $pid; wait $pid 2>/dev/null; return 1; }
    [ -n "${CAPFILE:-}" ] && [ "$(rows $CAPFILE)" -ge "${CAPN:-0}" ] && { log "cap of $CAPN rows reached, stopping $pid"; kill $pid; wait $pid 2>/dev/null; return 0; }
    f=$(free_mib)
    if [ "${f:-99999}" -lt 1000 ]; then log "YIELD: only $f MiB free, stopping my own process $pid"; kill $pid; wait $pid 2>/dev/null; return 2; fi
  done
  wait $pid; return $?
}
run_selfscore() {   # run_selfscore <model> <split> <rows wanted>
  m=$1; s=$2; want=$3; out=$D/work/features/$m/$s.dlmself.jsonl; tries=0
  while [ "$(rows $out)" -lt "$want" ]; do
    [ -e $D/STOP ] && { log "STOP file, ending"; exit 0; }
    tries=$((tries+1)); [ $tries -gt 60 ] && { log "too many tries for $m $s, giving up on it"; return; }
    wait_room 20000 || exit 0
    log "START self_score $m $s (have $(rows $out) of $want)"
    guarded $D/logs/self_score_${m}_$s.log $PY self_score.py --model $m --split $s --chunk 2; log "job process ended rc=$?, rows now $(rows $out)"
    sleep 30
  done
  log "DONE self_score $m $s ($(rows $out) rows)"
}
run_k256() {   # run_k256 <model> : Dream K=256, reverse order, own work dir
  m=$1; out=$D/work/candidates/$m@k256/test.jsonl; tries=0
  while [ "$(rows $out)" -lt "$MAXK" ]; do
    [ -e $D/STOP ] && { log "STOP file, ending"; exit 0; }
    tries=$((tries+1)); [ $tries -gt 60 ] && { log "too many tries for K256 $m, giving up on it"; return; }
    wait_room ${ADA_K256_MIB:-22000} || exit 0
    log "START generate $m K=256 --reverse (have $(rows $out) of max $MAXK)"
    CAPFILE=$out CAPN=$MAXK guarded $D/logs/gen_${m}_k256.log $PY generate_candidates.py --model $m --split test --steps 256 --order $D/k256_order.txt --reverse; log "job process ended rc=$?, rows now $(rows $out)"
    sleep 30
  done
  log "DONE K256 $m ($(rows $out) rows)"
}
log "ada_run2 start (K256 max rows $MAXK)"
run_selfscore llada-base test 500
run_k256 dream-base
log "ALL DONE"
