#!/usr/bin/env bash
# start_window.sh <end "YYYY-MM-DD HH:MM" IST> : OPERATOR step, run only after Main's yes AND krispy's agreement. Writes WINDOW_END and starts the two
# time-boxed queues (they then wait, holding no GPU memory, until both GPUs are idle for 5 min). Remove the STOP files first.
set -eu
D=/home/antpc/codify/iter2
END=$(TZ=Asia/Kolkata date -d "$1" +%s)
[ "$END" -gt "$(date +%s)" ] || { echo "end time is in the past"; exit 1; }
echo "$END" > $D/WINDOW_END; echo "WINDOW_END = $(date -d @$END '+%F %T %Z')"
cd $D
for g in 0 1; do
  cp queue_gpu$g.txt queue_gpu$g.txt.before_window 2>/dev/null || true
  cp queue_gpu$g.window.txt queue_gpu$g.txt; rm -f STOP_gpu$g
done
echo "queues replaced by the window queues (old ones saved as .before_window); start queue.sh per GPU the way the earlier operator did:"
echo "  nohup bash $D/queue.sh 0 >> $D/logs/gpu0.log 2>&1 &   (and 1)"
