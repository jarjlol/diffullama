#!/usr/bin/env bash
# operator-start-guard: current consumers keep their old inode; future starts are singleton guarded.
set -u
GPU=${1:?GPU required}
D=/home/antpc/codify/iter2
exec 8>"$D/.lane_gpu$GPU.start.lock"
if ! flock -n 8; then exit 0; fi
python3 "$D/aryan/operator_start_guard.py" lane "$GPU" --exclude "$$" >> "$D/logs/lane_start_guard.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then exit "$([ "$rc" -eq 3 ] && echo 0 || echo "$rc")"; fi
source "$D/lane_body_pre_guard.sh" "$@"
