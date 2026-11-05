#!/bin/bash
# Retry downloads of the campaign models until each succeeds (max 6 tries each). Network only, low priority.
LOG=~/codify/campaign/logs/prefetch-retry.log
for r in diffusionfamily/diffullama Dream-org/Dream-v0-Base-7B GSAI-ML/LLaDA-8B-Base NousResearch/Llama-2-7b-hf Qwen/Qwen2.5-7B Qwen/Qwen2.5-0.5B TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T; do
  for t in 1 2 3 4 5 6; do
    if nice -n 10 ~/venvs/scorer/bin/python -c "from huggingface_hub import snapshot_download as s; s(\"$r\")" >/dev/null 2>>$LOG; then
      echo "$(date -u +%FT%TZ) OK $r (try $t)" >> $LOG; break
    fi
    echo "$(date -u +%FT%TZ) FAIL $r (try $t), retrying in 60s" >> $LOG; sleep 60
  done
done
echo "$(date -u +%FT%TZ) PREFETCH DONE" >> $LOG
