"""External R7 review check. Reads item features only; leaves official eval.py unchanged.

Run before accepting any selector: order_check.py SOLUTION ITEMS_JSONL.
Fresh namespaces are essential: reversing calls on a warm cache misses the old leak.
This is a regression check, not a proof that arbitrary Python is pure or a sandbox.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def selector(source):
    namespace = {"__name__": "review_selector"}
    exec(compile(source, "<review selector>", "exec"), namespace)
    return namespace["select"]


def choice(select, item):
    view = copy.deepcopy(item)
    index = select(view)
    if view != item:
        raise ValueError("select mutated its input")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(item["candidates"]):
        raise ValueError("invalid candidate index")
    return index


def check(source, items):
    keys = [(r["model"], r["uid"]) for r in items]
    if len(keys) != len(set(keys)) or not items:
        raise ValueError("empty items or duplicate model/uid")
    if any("gold" in r or "answer" in r for r in items):
        raise ValueError("item-level gold is forbidden")
    forward_select, reverse_select = selector(source), selector(source)
    forward = {key: choice(forward_select, row) for key, row in zip(keys, items)}
    reverse = {key: choice(reverse_select, row) for key, row in reversed(list(zip(keys, items)))}
    for key, row in zip(keys, items):
        alone = selector(source)
        isolated = choice(alone, row)
        repeated = choice(alone, row)
        if not forward[key] == reverse[key] == isolated == repeated:
            raise ValueError(f"order-dependent select: {key[0]}/{key[1]}")
    return {"correctness": "PASS", "items": len(items), "checks": [
        "fresh forward vs fresh reverse", "isolated item", "repeated item", "input mutation", "index bounds"],
        "solution_sha256": hashlib.sha256(source.encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("solution", type=Path)
    ap.add_argument("items", type=Path)
    args = ap.parse_args()
    try:
        items = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = check(args.solution.read_text(encoding="utf-8"), items)
    except Exception as exc:
        print(json.dumps({"correctness": "FAIL", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
