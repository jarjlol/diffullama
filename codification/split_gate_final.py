"""Split an exported Arbor repo's test data into a gate split and a final split.

    python3 split_gate_final.py ARBOR_REPO FINAL_DIR QUESTIONS_JSONL --gate 200

Arbor optimises on data/dev and gates merges on data/test, so a test split Arbor gates on
is no longer clean for the final number. This keeps the first --gate questions (in
QUESTIONS_JSONL order, i.e. pipeline/work/data/test.jsonl) as ARBOR_REPO/data/test, the gate,
and moves the rest to FINAL_DIR/{items,gold}.jsonl, outside the repo Arbor can read. The
final comparison (Arbor's best solution against every baseline) is made on FINAL_DIR only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(path: Path, rs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("final", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--gate", type=int, default=200)
    a = ap.parse_args()
    order = [r["uid"] for r in rows(a.questions)]
    test = a.repo / "data" / "test"
    items, gold = rows(test / "items.jsonl"), rows(test / "gold.jsonl")
    present = {r["uid"] for r in items}                  # gold.jsonl lists every test question; items only those generated
    order = [u for u in order if u in present]
    gate = set(order[:a.gate])
    assert len(gate) == a.gate, f"only {len(gate)} test questions present, need {a.gate}"
    write(test / "items.jsonl", [r for r in items if r["uid"] in gate])
    write(test / "gold.jsonl", [r for r in gold if r["uid"] in gate])
    write(a.final / "items.jsonl", [r for r in items if r["uid"] not in gate])
    write(a.final / "gold.jsonl", [r for r in gold if r["uid"] in present - gate])
    n_final = len(present) - len(gate)
    print(f"gate: {len(gate)} questions in {test}; final: {n_final} questions in {a.final} (not in the repo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
