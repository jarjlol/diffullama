#!/usr/bin/env bash
# HumanEval: merge shards, execute, AR-score, decision-score, table.
set -euo pipefail
cd /home/antpc/codify/diffullama/codification/pipeline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python humaneval_merge.py --model dream-base --split test
python humaneval_exec.py --score dream-base --split test

CUDA_VISIBLE_DEVICES=1 ~/venvs/scorer/bin/python humaneval_ar.py --model dream-base --split test
CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm/bin/python humaneval_decision.py --model dream-base --split test --mode listwise
CUDA_VISIBLE_DEVICES=0 ~/venvs/vllm/bin/python humaneval_decision.py --model dream-base --split test --mode pointwise

python humaneval_eval.py --models dream-base \
  --json work/humaneval/result.json --dump work/humaneval/preds.jsonl
