set -u
T=/tmp/qt; rm -rf $T; mkdir -p $T/logs; cd $T
printf '#!/usr/bin/env bash\necho "ordinary failure, nothing about CUDA"; exit 1\n' > fail.sh
printf '#!/usr/bin/env bash\necho "RuntimeError: CUDA unknown error - this may be due to ..."; exit 1\n' > cuda.sh
printf '#!/usr/bin/env bash\necho "job ok"; exit 0\n' > ok.sh
run() { # run <name> <gpu> <health> <queue lines...>
  name=$1; g=$2; h=$3; shift 3
  rm -f $T/queue_gpu$g.txt $T/queue_gpu$g.done $T/STOP_gpu$g $T/GPU_FAULT_gpu$g.txt $T/logs/queue_gpu$g.log
  printf '%s\n' "$@" > $T/queue_gpu$g.txt
  env QUEUE_DIR=$T QUEUE_SKIP_BUSY=1 QUEUE_RETRY_SLEEP=1 ${h:+QUEUE_FAKE_HEALTH=$h} timeout 40 bash /tmp/queue_body_v3.sh $g
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
