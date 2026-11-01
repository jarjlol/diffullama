"""eval.py -- PROTECTED evaluation harness. Do not edit during a research run.

Scores solution.select() on one split and prints exactly one machine-readable line

    score: <float>

score = macro-average accuracy over the diffusion LMs present in the split (each
model's accuracy is the fraction of questions whose selected candidate is correct),
so a selector cannot win by excelling on one model only. Higher is better.

Correctness gate: select() must return a valid integer index for every item, and
solution.py must pass a static check (no file/OS access, imports only from a small
stdlib whitelist -- so it cannot read gold from data/); otherwise the score is 0.0. dev and test are disjoint (see README / PROVENANCE):
dev comes from the GSM8K *train* file, test is the official GSM8K test set.
"""
from __future__ import annotations

import argparse
import ast
import collections
import sys
import time
from pathlib import Path

import task

ALLOWED_IMPORTS = {"math", "re", "collections", "statistics", "itertools", "functools", "heapq", "bisect", "typing", "__future__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "vars", "input", "breakpoint"}


def static_check(path: Path) -> None:
    """Reject a solution that could reach the gold files or the network."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                raise ValueError(f"solution.py imports {bad}; allowed: {sorted(ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"solution.py uses forbidden name {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            raise ValueError(f"solution.py uses dunder attribute {node.attr!r}")


def evaluate(split: str) -> tuple[float, dict]:
    static_check(Path(__file__).resolve().parent / "solution.py")
    import solution
    items, gold = task.load_items(split), task.load_gold(split)
    per = collections.defaultdict(lambda: [0, 0, 0])          # model -> [correct, n, oracle]
    t0 = time.perf_counter()
    for it in items:
        view = {k: v for k, v in it.items() if k != "gold"}  # belt and braces: never pass gold
        idx = solution.select(view)
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(it["candidates"]):
            raise ValueError(f"select() returned {idx!r} for {it['uid']} ({len(it['candidates'])} candidates)")
        g = gold[it["uid"]]
        rec = per[it["model"]]
        rec[0] += task.is_correct(it["candidates"][idx].get("answer"), g)
        rec[1] += 1
        rec[2] += any(task.is_correct(c.get("answer"), g) for c in it["candidates"])
    report = {m: {"acc": c / n, "n": n, "oracle": o / n} for m, (c, n, o) in sorted(per.items())}
    report["_seconds"] = time.perf_counter() - t0
    accs = [r["acc"] for k, r in report.items() if not k.startswith("_")]
    return (sum(accs) / len(accs) if accs else 0.0), report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    print(f"split: {args.split}")
    try:
        score, rep = evaluate(args.split)
    except Exception as e:                       # any failure still yields a parseable score
        print(f"correctness: FAIL ({type(e).__name__}: {e})")
        print("score: 0.0")
        return
    print("correctness: PASS")
    for m, r in rep.items():
        if m.startswith("_"):
            continue
        flag = "  WARNING: within 1% of oracle -- check for gold leakage" if r["n"] >= 50 and r["acc"] >= 0.99 * r["oracle"] > 0 else ""
        print(f"model {m}: acc {r['acc']:.4f}  oracle(hit@N) {r['oracle']:.4f}  n {r['n']}{flag}")
    print(f"seconds: {rep['_seconds']:.2f}")
    print(f"score: {score:.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
