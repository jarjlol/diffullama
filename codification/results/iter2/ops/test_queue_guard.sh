set -u
T=/tmp/qt; rm -rf $T; mkdir -p $T/logs; cd $T
printf '#!/usr/bin/env bash\necho "ordinary failure, nothing about CUDA"; exit 1\n' > fail.sh
printf '#!/usr/bin/env bash\necho "RuntimeError: CUDA unknown error - this may be due to ..."; exit 1\n' > cuda.sh
printf '#!/usr/bin/env bash\necho "job ok"; exit 0\n' > ok.sh
run() { # run <name> <gpu> <health> <queue lines...>
  name=$1; g=$2; h=$3; shift 3
  rm -f $T/queue_gpu$g.txt $T/queue_gpu$g.done $T/STOP_gpu$g $T/GPU_FAULT_gpu$g.txt $T/logs/queue_gpu$g.log
  printf '%s\n' "$@" > $T/queue_gpu$g.txt
  env QUEUE_DIR=$T QUEUE_PIPELINE_DIR=$T QUEUE_SKIP_BUSY=1 QUEUE_RETRY_SLEEP=1 ${h:+QUEUE_FAKE_HEALTH=$h} timeout 40 bash /tmp/queue_body_v3.sh $g
  echo "=== $name: exit $?"
  echo "-- queue file now:"; cat $T/queue_gpu$g.txt | cut -c1-80
  echo "-- done file:"; cat $T/queue_gpu$g.done 2>/dev/null | cut -c1-80
  echo "-- STOP exists: $([ -e $T/STOP_gpu$g ] && echo yes || echo no); fault file:"; cat $T/GPU_FAULT_gpu$g.txt 2>/dev/null | cut -c1-200
  echo "-- log (START/FAIL/FAULT/OK lines): $(grep -c START $T/logs/queue_gpu$g.log) starts"; grep -E "FAULT|OK:" $T/logs/queue_gpu$g.log | cut -c1-200
}
run "T1 ordinary failure, healthy GPU (fake ok)" 0 ok "SH $T/fail.sh" "SH $T/ok.sh"
run "T2 failure + FAILED health check (fake fail)" 0 fail "SH $T/fail.sh" "SH $T/ok.sh"
run "T3 job prints CUDA unknown error, health check ok" 0 ok "SH $T/cuda.sh" "SH $T/ok.sh"
echo; echo "######## T4 real health check on GPU 1 (no fake), ordinary failure"
run "T4 real health check" 1 "" "SH $T/fail.sh" "SH $T/ok.sh"

echo; echo "######## T5 queue killed mid-job: the running line must come back"
printf '#!/usr/bin/env bash
sleep 25
' > $T/slow.sh
rm -f $T/queue_gpu0.txt $T/queue_gpu0.done $T/queue_gpu0.running $T/STOP_gpu0 $T/logs/queue_gpu0.log
printf '%s
' "SH $T/slow.sh" "SH $T/ok.sh" > $T/queue_gpu0.txt
env QUEUE_DIR=$T QUEUE_PIPELINE_DIR=$T QUEUE_SKIP_BUSY=1 QUEUE_FAKE_HEALTH=ok bash /tmp/queue_body_v3.sh 0 &
QP=$!; sleep 4
echo "while running: running file = $(cat $T/queue_gpu0.running 2>/dev/null); queue file = $(tr '
' ' ' < $T/queue_gpu0.txt)"
pkill -KILL -P $QP; kill -KILL $QP; sleep 1       # what a pkill -KILL -u does: the queue and its job die
echo "after the kill: running file = $(cat $T/queue_gpu0.running 2>/dev/null); queue file = $(tr '
' ' ' < $T/queue_gpu0.txt); done file lines = $(cat $T/queue_gpu0.done 2>/dev/null | wc -l)"
env QUEUE_DIR=$T QUEUE_PIPELINE_DIR=$T QUEUE_SKIP_BUSY=1 QUEUE_FAKE_HEALTH=ok timeout 12 bash /tmp/queue_body_v3.sh 0
grep -E "restored|START" $T/logs/queue_gpu0.log | cut -c1-160
