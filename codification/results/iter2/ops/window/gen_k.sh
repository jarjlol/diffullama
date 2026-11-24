#!/usr/bin/env bash
# gen_k.sh <dream|llada> : K = 256 candidate generation on Blackwell, in the pre-declared random order (results/iter2/k256_order.txt), from the FRONT.
# Runs the worker clone's generate_candidates.py (the original checkout is not touched). Resumable by uid; rows carry host and gpu name.
set -euo pipefail
B=$1
W=/home/antpc/codify/diffullama/codification/pipeline/work
cd /home/antpc/codify/worker/codification/pipeline
export CODIFY_WORK_DIR=$W PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
bash /home/antpc/codify/iter2/window/prestart.sh $W/candidates/$B-base@k256/test.jsonl
exec /home/antpc/venvs/$B/bin/python generate_candidates.py --model $B-base --split test --steps 256 --order /home/antpc/codify/iter2/window/k256_order.txt
