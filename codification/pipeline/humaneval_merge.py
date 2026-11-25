"""Merge a reverse generation shard into the main HumanEval candidate file.

    python3 humaneval_merge.py --model dream-base --split test

The two shards write `<split>.jsonl` and `<split>.rev.jsonl`; this keeps the
first row per uid (the main shard) and appends only uids the main shard lacks.
Idempotent. The `.rev.jsonl` file is left in place for provenance; nothing
downstream reads it.
"""
from __future__ import annotations

import argparse

from common import WORK, read_jsonl, write_jsonl

HERE = WORK / "humaneval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()

    out = HERE / "candidates" / a.model / f"{a.split}.jsonl"
    rev = out.with_name(f"{a.split}.rev.jsonl")
    rows = read_jsonl(out)
    have = {r["uid"] for r in rows}
    add = [r for r in read_jsonl(rev)] if rev.exists() else []
    new = [r for r in add if r["uid"] not in have]
    if new:
        rows.extend(new)
        write_jsonl(out, rows)
    print(f"{a.model}/{a.split}: {len(have)} main + {len(new)} from rev = {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
