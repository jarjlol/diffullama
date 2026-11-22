#!/usr/bin/env bash
# merge_ext.sh <model@k256> : rows another machine generated (candidates/<model>/test.ext.jsonl, copied in from ada) are merged into test.jsonl
# for questions of the pre-declared K=256 order; a uid already generated here keeps the Blackwell row; generate_candidates.py only does this when it exits normally, and our time-boxed runs are stopped.
set -euo pipefail
M=$1; D=/home/antpc/codify/diffullama/codification/pipeline/work
python3 - "$M" "$D" <<'PY'
import json, sys, pathlib
m, d = sys.argv[1], pathlib.Path(sys.argv[2])
cd = d / "candidates" / m; out, ext = cd / "test.jsonl", cd / "test.ext.jsonl"
rd = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
want = {l.strip() for l in open("/home/antpc/codify/iter2/window/k256_order.txt", encoding="utf-8") if l.strip()}
have = {r["uid"] for r in rd(out)}
extra = [r for r in rd(ext) if r["uid"] in want and r["uid"] not in have]
with open(out, "a", encoding="utf-8", newline="
") as f:
    for r in extra: f.write(json.dumps(r, ensure_ascii=False) + "
")
print(f"{m}: merged {len(extra)} rows from {ext.name}; test.jsonl now {len(rd(out))} rows")
PY
