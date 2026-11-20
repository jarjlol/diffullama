#!/usr/bin/env bash
# resume_at.sh <gpu> <model> <venv> <not-before "YYYY-MM-DD HH:MM">
# Agreed with the other lab user: he ends his job at 03:00 IST on 21 Nov. From that time on, as soon as
# GPU <gpu> is idle (load <= 10% and >= 50 GB free) for two checks in a row (2 min), remove our stop file and restart
# lane <gpu> (it resumes by uid), then its queue. Holds no GPU while waiting.
set -u
GPU=$1; MODEL=$2; VENV=$3; NOT_BEFORE=$4
D=~/codify/iter2; LOG=$D/logs/resume_gpu$GPU.log
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
busy() {  # "busy" = not idle enough for us: load above 10% or less than 50 GB free.
  # Other users may keep idle background processes on the card; those are fine to share with.
  local s; s=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits | awk -F", " -v g=$GPU '$1==g{print $2, $3}')
  set -- $s
  [ "${1:-100}" -gt 10 ] || [ "${2:-0}" -lt 51200 ]
}
{
echo "$(date '+%F %T') waiting for $NOT_BEFORE and a free GPU $GPU"
while [ "$(date +%s)" -lt "$(date -d "$NOT_BEFORE" +%s)" ]; do sleep 60; done
free=0
while [ $free -lt 2 ]; do
  if busy; then free=0; else free=$((free + 1)); fi
  sleep 60
done
rm -f "$D/STOP_gpu$GPU"
cd "$D"
setsid -f nohup ./lane.sh "$GPU" "$MODEL" "$VENV" > /dev/null 2>&1 < /dev/null
sleep 10
pgrep -f "lane.sh $GPU " > /dev/null || { echo "$(date '+%F %T') lane $GPU did not start"; exit 1; }
setsid -f nohup ./queue.sh "$GPU" > /dev/null 2>&1 < /dev/null
echo "$(date '+%F %T') GPU $GPU free: lane $GPU and queue $GPU restarted"
} >> "$LOG" 2>&1
