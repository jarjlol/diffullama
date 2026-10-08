#!/usr/bin/env bash
# Serve the ideation generator and reviewer on ONE GPU with vLLM, as two OpenAI-compatible
# servers bound to localhost only. Read ideation/WORKSTATION_RUN.md first.
#
#   ideation/local/serve_models.sh            # start both, wait until healthy
#   ideation/local/stop_models.sh             # stop both
#
# Overridable via environment (defaults = the recommended pair, see WORKSTATION_RUN.md §2):
#   GEN_MODEL  REV_MODEL           Hugging Face ids
#   GEN_UTIL   REV_UTIL            vLLM --gpu-memory-utilization per server (fraction of TOTAL GPU memory)
#   GEN_REASONING_PARSER / REV_REASONING_PARSER   vLLM --reasoning-parser ('' to omit)
#   MAX_LEN                        --max-model-len (prompts are <=8k tokens; 32k is ample)
#   VLLM                           vllm executable (e.g. ~/venvs/vllm/bin/vllm)
set -euo pipefail

GEN_MODEL="${GEN_MODEL:-Qwen/Qwen3.8-27B-FP8}"
REV_MODEL="${REV_MODEL:-RedHatAI/gemma-4-31B-it-FP8-block}"
GEN_UTIL="${GEN_UTIL:-0.45}"
REV_UTIL="${REV_UTIL:-0.45}"
GEN_REASONING_PARSER="${GEN_REASONING_PARSER-qwen3}"
REV_REASONING_PARSER="${REV_REASONING_PARSER-}"
MAX_LEN="${MAX_LEN:-32768}"
GEN_PORT=8000; REV_PORT=8001
VLLM="${VLLM:-vllm}"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/ideation"
mkdir -p "$STATE"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v "$VLLM" >/dev/null || die "vLLM not found ('$VLLM'). Install it: WORKSTATION_RUN.md §3"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"

# ---- the GPU is shared: refuse to start if there is not enough free memory -------------
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
free=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | head -1)
need=$(python3 -c "print(int($total*($GEN_UTIL+$REV_UTIL)))")
echo "GPU: ${free} MiB free of ${total} MiB; this setup needs ~${need} MiB"
if (( free < need )); then
  echo "Processes currently on the GPU:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  die "not enough free GPU memory. Ask whoever is using it, or lower GEN_UTIL/REV_UTIL. Do NOT kill others' jobs."
fi

running() { [[ -f "$STATE/vllm-$1.pid" ]] && kill -0 "$(cat "$STATE/vllm-$1.pid")" 2>/dev/null; }

launch() {   # name model port util reasoning_parser
  local name=$1 model=$2 port=$3 util=$4 parser=$5
  if running "$name"; then echo "$name already running (pid $(cat "$STATE/vllm-$name.pid"))"; return; fi
  local args=(serve "$model" --host 127.0.0.1 --port "$port" --served-model-name "$name"
              --gpu-memory-utilization "$util" --max-model-len "$MAX_LEN" --max-num-seqs 4)
  [[ -n "$parser" ]] && args+=(--reasoning-parser "$parser")
  echo "starting $name: $model on :$port (util $util${parser:+, reasoning-parser $parser})"
  nohup "$VLLM" "${args[@]}" > "$STATE/vllm-$name.log" 2>&1 &
  echo $! > "$STATE/vllm-$name.pid"
}

wait_healthy() {   # name port
  local name=$1 port=$2 t=0
  printf "waiting for %s (first start downloads weights; can take a while)" "$name"
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    running "$name" || { echo; tail -30 "$STATE/vllm-$name.log"; die "$name exited; log above ($STATE/vllm-$name.log)"; }
    (( t++ > 2700 )) && die "$name not healthy after 90 min; see $STATE/vllm-$name.log"
    printf "."; sleep 2
  done
  echo " ok"
}

# Sequential start: each server profiles GPU memory at startup, so starting both at once
# can make them mis-measure each other.
launch ideation-gen "$GEN_MODEL" "$GEN_PORT" "$GEN_UTIL" "$GEN_REASONING_PARSER"; wait_healthy ideation-gen "$GEN_PORT"
launch ideation-rev "$REV_MODEL" "$REV_PORT" "$REV_UTIL" "$REV_REASONING_PARSER"; wait_healthy ideation-rev "$REV_PORT"

echo
echo "Both servers healthy (localhost only). Logs: $STATE/vllm-*.log"
echo "Next: cp ideation/local/local.env.example ~/.ideation-local-env && chmod 600 ~/.ideation-local-env"
echo "      python3 ideation/local/probe_models.py      # must PASS before the full run"
