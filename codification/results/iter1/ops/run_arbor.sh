#!/usr/bin/env bash
# Start Arbor iteration 1 on the rerank repo (vLLM already serving on GPU 1). Output also goes to arbor.log.
cd ~/arbor-rerank
OPENAI_API_KEY=dummy ~/venvs/arbor/bin/arbor run "$(cat ~/codify/diffullama/codification/arbor_instruction.txt)" \
  --yes --yes-cwd ~/arbor-rerank --run-name iter1b 2>&1 | tee -a ~/codify/arbor.log
