#!/usr/bin/env bash
# a3_final_eval.sh READY WHICH OUTDIR : score the exported Arbor repo's current solution.py once on the final (or gate) split.
# CPU only. Uses eval_final.sh (temp copy of the repo with the split in data/test) and paired_stats.py. WHICH = final | gate.
set -u
READY=$1; WHICH=$2; OUT=$3
P=~/codify/worker/codification
export CODIFY_WORK_DIR=~/codify/diffullama/codification/pipeline/work
mkdir -p "$OUT"
python3 "$P/pipeline/a3_final_items.py" --which "$WHICH" --out "$OUT/split" --like "$READY/data/test/items.jsonl" || exit $?
echo "== solution under test: $READY/solution.py  (git: $(git -C "$READY" log --oneline | head -3 | tr '\n' ';'))"
bash "$P/eval_final.sh" "$READY" "$OUT/split" "$READY/solution.py" || exit $?
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
( cd "$READY" && tar --exclude=.git --exclude=.arbor --exclude=__pycache__ -cf - . ) | ( cd "$T" && tar -xf - )
cp "$OUT/split/items.jsonl" "$OUT/split/gold.jsonl" "$T/data/test/"
echo "== paired stats vs majority and first ($WHICH split)"
python3 "$P/paired_stats.py" "$T" test "$READY/solution.py"
