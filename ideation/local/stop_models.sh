#!/usr/bin/env bash
# Stop the two vLLM servers started by serve_models.sh (only those: matched by saved PIDs).
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/ideation"
for name in ideation-gen ideation-rev; do
  f="$STATE/vllm-$name.pid"
  [[ -f "$f" ]] || { echo "$name: not started by serve_models.sh"; continue; }
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"; for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "$name (pid $pid) did not stop after 30 s; stop it by hand"; continue; }
    echo "$name stopped"
  else echo "$name: not running"; fi
  unlink "$f"
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
