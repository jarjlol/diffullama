#!/usr/bin/env bash
# selfscore_k.sh <dream|llada> <cut|trim> : DLM self-score of the K = 256 test candidates (candidates/<model>@k256/test.jsonl).
# features go to work/features/<model>-base@k256/test.dlmself[trim].jsonl, where cut_vs_trim.py / step_arm.py read them. Resumable by uid.
set -euo pipefail
B=$1; TEXT=${2:-cut}; M=$B-base
W=/home/antpc/codify/diffullama/codification/pipeline/work
SUF=""; [ "$TEXT" = trim ] && SUF=trim
cd /home/antpc/codify/worker/codification/pipeline
export CODIFY_WORK_DIR=$W PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p "$W/features/$M@k256"
exec /home/antpc/venvs/$B/bin/python self_score.py --model "$M" --candidates-from "$M@k256" --split test --text "$TEXT" --out "$W/features/$M@k256/test.dlmself$SUF.jsonl" ${SS_CHUNK:+--chunk $SS_CHUNK}
