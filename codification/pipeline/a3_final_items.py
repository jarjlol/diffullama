"""Assemble the items for the A3 final evaluation from work/ (cached candidates + features), CPU only.

    python3 a3_final_items.py --which final --out DIR --like READY/data/test/items.jsonl [--models dream-base llada-base]

--which final: test questions 201..500 in work/data/test.jsonl order (the 300 Arbor never saw).
--which gate : the first 200 (Arbor's gate; used by the dry run so the real final split is not read).
Writes DIR/items.jsonl and DIR/gold.jsonl in the layout eval_final.sh expects (like assemble.py: one record per
(model, question), features merged from every work/features/<model>/test.*.jsonl, plus len_chars). Only the feature
keys that appear in --like (the exported task's items) are kept, so the selector sees what it was developed on.
Exit 2 and a message if any selected question or any needed feature is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import WORK, read_jsonl, task, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["final", "gate"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--like", required=True, help="the exported task's data/test/items.jsonl (feature keys to keep)")
    ap.add_argument("--models", nargs="*", default=["dream-base", "llada-base"])
    a = ap.parse_args()
    order = [q["uid"] for q in read_jsonl(WORK / "data" / "test.jsonl")]
    gold = {q["uid"]: q["answer"] for q in read_jsonl(WORK / "data" / "test.jsonl")}
    uids = order[200:] if a.which == "final" else order[:200]
    keys = set()
    for it in read_jsonl(a.like):
        for c in it["candidates"]:
            keys |= set(c["features"])
    keys.add("len_chars")
    items, problems = [], []
    for m in a.models:
        cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / m / "test.jsonl")}
        feats = {}
        for f in sorted((WORK / "features" / m).glob("test.*.jsonl")):
            for r in read_jsonl(f):
                feats.setdefault(r["uid"], []).append(r["features"])
        for u in uids:
            if u not in cands:
                problems.append(f"{m}: candidates missing for {u}"); continue
            r = cands[u]; cs = []
            for j, cd in enumerate(r["candidates"]):
                f = {"len_chars": len(task.trim(cd["text"]))}
                for per in feats.get(u, []):
                    f.update(per[j])
                missing = keys - set(f)
                if missing:
                    problems.append(f"{m} {u}: features missing {sorted(missing)[:3]}"); break
                cs.append({"text": cd["text"], "answer": cd["answer"], "features": {k: f[k] for k in keys}})
            else:
                items.append({"uid": u, "model": m, "question": r["question"], "candidates": cs})
    if problems:
        print(f"NOT READY: {len(problems)} problems, first: {problems[:3]}", file=sys.stderr)
        return 2
    assert all("gold" not in it for it in items)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "items.jsonl", items)
    write_jsonl(out / "gold.jsonl", [{"uid": u, "answer": gold[u]} for u in uids])
    print(f"{a.which}: {len(items)} items ({len(a.models)} models x {len(uids)} questions), {len(keys)} feature keys -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
