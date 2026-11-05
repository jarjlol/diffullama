#!/bin/bash
# Codification iteration 1: one venv per model family (README §B). CPU/network only, no GPU.
set -u
LOG=~/codify/setup-venvs.log; exec >>"$LOG" 2>&1
echo "=== start $(date -u)"
mk() {
  local name=$1; shift
  echo "== $name $(date -u)"
  if [ -x ~/venvs/$name/bin/python ]; then echo "$name exists, skipping create"; else python3 -m venv ~/venvs/$name || return 1; fi
  ~/venvs/$name/bin/pip install -q --upgrade pip
  ~/venvs/$name/bin/pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 || { echo "$name TORCH FAILED"; return 1; }
  ~/venvs/$name/bin/pip install -q "$@" || { echo "$name DEPS FAILED"; return 1; }
  ~/venvs/$name/bin/python -c "import torch, transformers; print('$name OK torch', torch.__version__, 'transformers', transformers.__version__, 'archs', torch.cuda.get_arch_list())"
}
mk anchor transformers==4.44.2 accelerate datasets sentencepiece protobuf
(cd ~/codify/diffullama && ~/venvs/anchor/bin/python -c "import model; print('anchor model.py imports OK')")
mk dream  transformers==4.46.2 accelerate datasets sentencepiece protobuf
mk llada  transformers==4.38.2 accelerate datasets sentencepiece protobuf
mk scorer "transformers>=4.46,<5" accelerate datasets sentencepiece protobuf
echo "=== done $(date -u)"
