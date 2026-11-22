#!/usr/bin/env bash
# start_window.sh <hours> : OPERATOR step, run only after the go from Aaditya/Main. Declares a window of at most <hours> h, COUNTED FROM THE FIRST START
# of a window job (krispy's pause has no fixed end; it lasts as long as our run, and we cap it). Removes the STOP files and the old WINDOW_START.
# The window queues then wait, holding no GPU memory, until a GPU has been idle for 5 min (krispy's job finished), and stop at start+hours or at
# once if a foreign process appears on a GPU.
set -eu
D=/home/antpc/codify/iter2
H=${1:?hours}
rm -f $D/WINDOW_END $D/WINDOW_START; echo "$H" > $D/WINDOW_HOURS
cd $D
for g in 0 1; do
  cp queue_gpu$g.txt queue_gpu$g.txt.before_window 2>/dev/null || true
  cp queue_gpu$g.window.txt queue_gpu$g.txt; rm -f STOP_gpu$g
done
echo "window of $H h declared; queues replaced by the window queues (old ones saved as .before_window). Start queue.sh per GPU:"
echo "  nohup bash $D/queue.sh 0 >> $D/logs/gpu0.log 2>&1 &   (and 1)"
