#!/usr/bin/env bash
# GSM8K decision-model scoring + table. Run from anywhere on blackwell.
set -euo pipefail
cd /home/antpc/codify/diffullama/codification/pipeline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
MODE="${1:-listwise}"

for m in dream-base llada-base; do
  python decision_rerank.py --model "$m" --split test --mode "$MODE" --backend pplx
done

if [ "$MODE" = listwise ]; then
  python decision_eval.py --split test --models dream-base llada-base \
    --json work/decision_test.json --dump work/decision_test_preds.jsonl
fi
