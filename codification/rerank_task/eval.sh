#!/usr/bin/env bash
# eval.sh -- PROTECTED wrapper. Do not edit during a research run.
#   bash eval.sh        # dev split  (Arbor iterates on this)
#   bash eval.sh test   # held-out split (gates merges)
# Prints a single "score: <float>" line that Arbor reads as the metric.
set -euo pipefail
SPLIT="${1:-dev}"
export PYTHONHASHSEED=0
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
exec "${PYTHON:-python3}" "$HERE/eval.py" --split "$SPLIT"
