#!/usr/bin/env bash
# after_campaign.sh -- chain the steps after the generation campaign, on GPU 1 only (GPU 0 is
# left to other users). Run under setsid; log in ~/codify/campaign/after.log.
#   1. wait for the running campaign to finish
#   2. rerun it with --gpus 1: the new scorer jobs (qwen7b, llama2-7b) and any left over
#   3. export the Arbor repo, with 200 test questions as its gate and 300 kept outside (final)
#   4. start vLLM for Arbor on GPU 1 and check that tool calling works
# Arbor itself is started by hand: review mode needs a person at the prompts.
set -uo pipefail
C=~/codify/campaign; REPO=~/codify/diffullama; LOG=$C/after.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
alive() { pgrep -f '^python3 campaign.py' > /dev/null; }

say "waiting for the campaign to finish"
while alive; do sleep 60; done
say "campaign exited: $(grep -E 'FINISHED|STOPPED' $C/campaign.log | tail -1)"

cd $REPO/codification/campaign
setsid python3 campaign.py --no-prefetch --warm-start --settle-min 3 --gpus 1 >> $C/supervisor.out 2>&1 < /dev/null &
sleep 30
while alive; do sleep 60; done
say "scoring rerun exited: $(grep -E 'FINISHED|STOPPED' $C/campaign.log | tail -1)"

rm -rf ~/arbor-rerank-final
FINAL_DIR=~/arbor-rerank-final bash $REPO/codification/make_arbor_repo.sh ~/arbor-rerank >> "$LOG" 2>&1 \
  || { say "make_arbor_repo failed"; exit 1; }
say "Arbor repo ready at ~/arbor-rerank; final split at ~/arbor-rerank-final"

# GPU 1 must be free of anyone else's processes before vLLM takes it
U1=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U1"; do
  say "GPU 1 busy, waiting"; sleep 120
done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
  setsid ~/venvs/vllm/bin/vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 4 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  > ~/codify/vllm.log 2>&1 < /dev/null &
echo $! > ~/codify/vllm.pid
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
  sleep 20
done
curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { say "vLLM did not come up; see ~/codify/vllm.log"; exit 1; }
say "vLLM up on GPU 1 (pid $(cat ~/codify/vllm.pid))"
reply=$(curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-27B-FP8", "max_tokens": 512,
  "messages": [{"role": "user", "content": "What is 17 * 23? Use the calculator tool."}],
  "tools": [{"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}],
  "tool_choice": "auto"}')
if echo "$reply" | grep -q '"tool_calls": *\[ *{'; then say "tool calling OK"; else say "tool calling FAILED: ${reply:0:400}"; fi
say "READY for Arbor"
