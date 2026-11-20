#!/usr/bin/env bash
# Queue wrapper: DLM self-score (T-a). usage: run_selfscore.sh <dream-base|llada-base> <dev|test> <cut|trim>
# Runs the worker clone's self_score.py (the queue itself runs from ~/codify/diffullama, which we do not touch).
# Does NOT set CUDA_VISIBLE_DEVICES (queue.sh pins the GPU). Resumable by uid; non-zero exit on failure so the
# queue's 8-try retry works. Writes work/features/<model>/<split>.dlmself[trim].jsonl.
set -euo pipefail
MODEL=$1; SPLIT=$2; TEXT=${3:-cut}
case "$MODEL" in dream-base) VENV=dream;; llada-base) VENV=llada;; *) echo "bad model $MODEL" >&2; exit 2;; esac
cd /home/antpc/codify/worker/codification/pipeline
export CODIFY_WORK_DIR=/home/antpc/codify/diffullama/codification/pipeline/work PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
echo "$(date '+%F %T') self_score $MODEL $SPLIT $TEXT  worker commit $(git rev-parse --short HEAD)  GPU=${CUDA_VISIBLE_DEVICES:-unset}"
exec /home/antpc/venvs/$VENV/bin/python self_score.py --model "$MODEL" --split "$SPLIT" --text "$TEXT"
