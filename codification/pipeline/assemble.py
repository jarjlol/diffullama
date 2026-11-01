"""Step 4 -- merge candidates + every cached feature file into the Arbor task's data.

    python3 assemble.py --split dev
    python3 assemble.py --split test

Writes ../rerank_task/data/<split>/items.jsonl: one record per (model, question),
with NO gold (gold.jsonl is written separately by prepare_data.py). Every model and
every feature file present under work/ is included; questions missing from any
model are reported, never silently dropped.
"""
from __future__ import annotations

import argparse

from common import TASK, WORK, read_jsonl, task, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--models", nargs="*", help="restrict to these models (default: all generated)")
    a = ap.parse_args()
    gold = {r["uid"] for r in read_jsonl(TASK / "data" / a.split / "gold.jsonl")}
    models = a.models or sorted(p.name for p in (WORK / "candidates").iterdir() if (p / f"{a.split}.jsonl").exists())
    items = []
    for m in models:
        cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / m / f"{a.split}.jsonl")}
        feats = {}
        for f in sorted((WORK / "features" / m).glob(f"{a.split}.*.jsonl")) if (WORK / "features" / m).exists() else []:
            for r in read_jsonl(f):
                feats.setdefault(r["uid"], []).append(r["features"])
        missing = gold - cands.keys()
        if missing:
            print(f"  {m}: {len(missing)} of {len(gold)} questions not generated yet -- excluded from this split")
        for uid in sorted(cands.keys() & gold):
            r = cands[uid]
            cs = []
            for j, c in enumerate(r["candidates"]):
                f = {"len_chars": len(task.trim(c["text"]))}
                for per_scorer in feats.get(uid, []):
                    f.update(per_scorer[j])
                cs.append({"text": c["text"], "answer": c["answer"], "features": f})
            items.append({"uid": uid, "model": m, "question": r["question"], "candidates": cs})
    assert all("gold" not in it and "answer" not in it for it in items), "gold leaked into items"
    write_jsonl(TASK / "data" / a.split / "items.jsonl", items)
    print(f"{a.split}: {len(items)} items from {len(models)} model(s): {models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
