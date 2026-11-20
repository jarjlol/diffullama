#!/usr/bin/env bash
# Iteration 2, Tier 1: one lane = one GPU, a fixed list of steps run in order.
# Each step is retried up to 8 times (the display watchdog sometimes kills a kernel, I-18);
# every step resumes by uid, so a retry loses at most the question in flight.
# Stop between steps with: touch ~/codify/iter2/STOP_gpu<N>
#   usage: lane.sh <gpu> <model> <gen_venv>
set -u
GPU=$1; MODEL=$2; VENV=$3
P=~/codify/diffullama/codification/pipeline
LOG=~/codify/iter2/logs/gpu$GPU.log
STOP=~/codify/iter2/STOP_gpu$GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$P"

step() {   # step <venv> <args...>
  local py=~/venvs/$1/bin/python; shift
  for try in 1 2 3 4 5 6 7 8; do
    [ -e "$STOP" ] && { echo "$(date '+%F %T') STOP file present, lane ends before: $*"; exit 0; }
    echo "$(date '+%F %T') START try $try: $*"
    if "$py" "$@"; then echo "$(date '+%F %T') OK: $*"; return 0; fi
    echo "$(date '+%F %T') FAIL try $try: $*"; sleep 30
  done
  echo "$(date '+%F %T') GAVE UP after 8 tries: $*"; exit 1
}

{
echo "$(date '+%F %T') lane gpu$GPU model $MODEL begins (branch $(git -C ~/codify/diffullama rev-parse --short HEAD))"
if [ "$MODEL" = dream-base ]; then
  step "$VENV" fidelity_check.py --config dream-sampler --limit 100
  step "$VENV" fidelity_check.py --config dream-published --limit 100
else
  step "$VENV" fidelity_check.py --config llada-published --limit 100
fi
for K in 16 32; do
  step "$VENV" generate_candidates.py --model "$MODEL" --split test --limit 500 --steps $K
done
for K in 16 32; do
  for S in gpt2 qwen0.5b qwen7b; do
    step scorer score_candidates.py --model "$MODEL@k$K" --split test --scorer $S
    step scorer score_candidates.py --model "$MODEL@k$K" --split test --scorer $S --cut
  done
done
echo "$(date '+%F %T') lane gpu$GPU DONE"
} >> "$LOG" 2>&1
