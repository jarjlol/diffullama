#!/usr/bin/env bash
# eval_final.sh ARBOR_REPO FINAL_DIR [SOLUTION] -- score a solution and every baseline on the
# final split (the test questions Arbor never saw; see split_gate_final.py). Works on a temp
# copy, so the Arbor repo is left untouched.
set -euo pipefail
REPO="${1:?usage: eval_final.sh ARBOR_REPO FINAL_DIR [SOLUTION]}"
FINAL="${2:?usage: eval_final.sh ARBOR_REPO FINAL_DIR [SOLUTION]}"
SOL="$(realpath "${3:-$REPO/solution.py}")"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
( cd "$REPO" && tar --exclude=.git --exclude=.arbor --exclude=__pycache__ -cf - . ) | ( cd "$T" && tar -xf - )
cp "$FINAL/items.jsonl" "$FINAL/gold.jsonl" "$T/data/test/"
cp "$SOL" "$T/solution.py"
cd "$T"
echo "== solution: $SOL (final split)"
bash eval.sh test
echo "== baselines (final split)"
"${PYTHON:-python3}" baselines.py --split test
