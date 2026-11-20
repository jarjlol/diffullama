#!/usr/bin/env bash
# assist_reverse.sh <model> <venv>: while the other GPU's lane is generating <model>@k<K> on the test
# split, generate the same questions in REVERSE on this GPU, in a separate work dir. Every minute each
# side's finished rows are copied to the other as test.ext.jsonl (complete lines only, atomic rename);
# generate_candidates.py skips uids in its .ext.jsonl and the main lane merges them when it finishes.
# Repeats for each K the lane generates (16, then 32). Ends when the lane is no longer generating.
# Started by queue.sh with CUDA_VISIBLE_DEVICES already set to this GPU.
set -u
M=$1; VENV=$2
P=~/codify/diffullama/codification/pipeline
AW=~/codify/iter2/assist-work
mkdir -p "$AW"; [ -e "$AW/data" ] || ln -s "$P/work/data" "$AW/data"

sync_lines() {   # sync_lines <src> <dst>: copy complete JSON lines only, then rename into place
  python3 - "$1" "$2" <<'EOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
if not os.path.exists(src):
    sys.exit(0)
keep = []
with open(src, encoding="utf-8") as f:
    for line in f:
        if not line.endswith("\n"):
            break
        try:
            json.loads(line)
        except ValueError:
            break
        keep.append(line)
tmp = dst + ".part"
with open(tmp, "w", encoding="utf-8") as f:
    f.writelines(keep)
os.replace(tmp, dst)
EOF
}

main_k() {   # the K the main lane is generating for model M right now, or empty
  pgrep -af "generate_candidates.py --model $M --split test --limit 500 --steps" | grep -v assist-work \
    | grep -v -- "--reverse" | sed -n 's/.*--steps \([0-9]*\).*/\1/p' | head -1
}

while true; do
  K=$(main_k)
  [ -z "$K" ] && { echo "$(date '+%F %T') main lane not generating $M; assist ends"; exit 0; }
  TAG="$M@k$K"
  MAIN=$P/work/candidates/$TAG/test.jsonl
  MINE=$AW/candidates/$TAG/test.jsonl
  mkdir -p "$(dirname "$MINE")"
  sync_lines "$MAIN" "$(dirname "$MINE")/test.ext.jsonl"
  echo "$(date '+%F %T') assist $TAG in reverse (main has $(wc -l < "$MAIN" 2>/dev/null || echo 0) rows)"
  CODIFY_WORK_DIR=$AW ~/venvs/$VENV/bin/python "$P/generate_candidates.py" --model "$M" --split test \
    --limit 500 --steps "$K" --reverse &
  GEN=$!
  while kill -0 $GEN 2>/dev/null; do
    sync_lines "$MINE" "$(dirname "$MAIN")/test.ext.jsonl"
    sync_lines "$MAIN" "$(dirname "$MINE")/test.ext.jsonl"
    [ "$(main_k)" = "$K" ] || { echo "$(date '+%F %T') main lane left $TAG; stopping assist"; kill $GEN; wait $GEN; break; }
    sleep 60
  done
  wait $GEN 2>/dev/null; RC=$?
  sync_lines "$MINE" "$(dirname "$MAIN")/test.ext.jsonl"
  echo "$(date '+%F %T') assist $TAG exit $RC; assist rows $(wc -l < "$MINE" 2>/dev/null || echo 0)"
  # wait until the main lane moves on from this K before checking for the next one
  while [ "$(main_k)" = "$K" ]; do sleep 60; done
done
