#!/usr/bin/env bash
# Queue wrapper: per-question hidden states for T-b. usage: run_pq_extract.sh <dream|qwen7b> [split]
# Writes work/preservation_pq/<model>.<split>.pt (new directory). About 500 forwards of ~100 tokens: minutes.
set -euo pipefail
MODEL=$1; SPLIT=${2:-test}
case "$MODEL" in dream) VENV=dream;; qwen7b) VENV=scorer;; *) echo "bad model $MODEL" >&2; exit 2;; esac
cd /home/antpc/codify/worker/codification/pipeline
export CODIFY_WORK_DIR=/home/antpc/codify/diffullama/codification/pipeline/work PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
echo "$(date '+%F %T') per_question_preservation extract $MODEL $SPLIT  worker commit $(git rev-parse --short HEAD)  GPU=${CUDA_VISIBLE_DEVICES:-unset}"
exec /home/antpc/venvs/$VENV/bin/python per_question_preservation.py extract --model "$MODEL" --split "$SPLIT"
