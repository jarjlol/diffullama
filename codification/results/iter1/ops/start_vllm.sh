#!/usr/bin/env bash
# vLLM for Arbor: Qwen3.8-27B-FP8, one data-parallel replica per GPU (0 and 1), 128k context. Arbor asks
# for 16k output tokens per call, so the 32k limit of the first launch overflowed on its 4th turn.
# Not tensor-parallel: TP=2 hung in the first all-reduce (P2P over PCIe does not work on this box).
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
  setsid ~/venvs/vllm/bin/vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 \
  --data-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 131072 --max-num-seqs 8 \
  --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  > ~/codify/vllm.log 2>&1 < /dev/null &
echo $! > ~/codify/vllm.pid
