#!/usr/bin/env bash
# greedy_k.sh <dream|llada> : greedy (T = 0), one sample, 256 steps, SAME prompt and sampler settings as the K = 256 rows except T = 0 and N = 1
# (fidelity_check.py configs dream-sampler / llada-published), on exactly the questions that have K = 256 rows (Blackwell test.jsonl + Ada's test.ext.jsonl).
# Rows already in work/fidelity/<config>.jsonl (the first-100 fidelity run, same config) are reused. Resumable by uid. Secondary/context comparison, not the test of record.
set -euo pipefail
B=$1
W=/home/antpc/codify/diffullama/codification/pipeline/work
case $B in dream) CONF=dream-sampler;; llada) CONF=llada-published;; *) echo "dream|llada"; exit 2;; esac
cd /home/antpc/codify/worker/codification/pipeline
export CODIFY_WORK_DIR=$W PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
C=$W/candidates/$B-base@k256
bash /home/antpc/codify/iter2/window/prestart.sh $W/fidelity/$CONF.jsonl
exec /home/antpc/venvs/$B/bin/python fidelity_check.py --config $CONF --from-candidates "$C/test.jsonl,$C/test.ext.jsonl"
