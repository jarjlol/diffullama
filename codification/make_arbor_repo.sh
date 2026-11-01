#!/usr/bin/env bash
# make_arbor_repo.sh DEST -- copy the filled task pack into a fresh, clean git repo for Arbor.
#
# Arbor needs a clean git repo whose root is the task. The course repo is not that, so the
# pack is exported. Run after `pipeline/assemble.py` has written data/{dev,test}/items.jsonl.
set -euo pipefail
DEST="${1:?usage: make_arbor_repo.sh DEST}"
SRC="${CODIFY_TASK_DIR:-$(cd "$(dirname "$0")" && pwd)/rerank_task}"
[ -e "$DEST" ] && { echo "refusing: $DEST already exists" >&2; exit 1; }
for s in dev test; do
  for f in items gold; do
    [ -s "$SRC/data/$s/$f.jsonl" ] || { echo "missing $SRC/data/$s/$f.jsonl -- run the pipeline first" >&2; exit 1; }
  done
done
grep -q '"gold"' "$SRC"/data/*/items.jsonl && { echo "refusing: items.jsonl contains a gold field" >&2; exit 1; }
mkdir -p "$DEST"
( cd "$SRC" && tar --exclude='__pycache__' -cf - . ) | ( cd "$DEST" && tar -xf - )
cd "$DEST"
out="$(bash eval.sh dev)"; echo "$out"
echo "$out" | grep -q '^correctness: PASS' || { echo "seed solution fails eval -- not exporting" >&2; exit 1; }
git init -q
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-codify}" -c user.email="${GIT_AUTHOR_EMAIL:-codify@localhost}" \
    commit -qm "rerank_task: seed solution and cached candidates"
echo "Arbor repo ready: $DEST  (seed dev score above). Next: cd $DEST && arbor"
