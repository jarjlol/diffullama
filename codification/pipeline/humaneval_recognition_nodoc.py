"""Dev-only control for the HumanEval recognition probe: strip the docstring.

The main probe shows the problem prompt (which contains the doctest examples)
and two complete programs that also embed that prompt. A 20/20 dev score could
therefore be docstring-example matching rather than code discrimination. This
control removes the entire triple-quoted docstring from the problem and from
both options, leaving only the signature and the two bodies. The mutation is
unchanged and still verified failing; removing a docstring is semantics-
preserving, so the canonical/mutant labels are unaffected.

    python3 humaneval_recognition_nodoc.py --make
    python3 humaneval_recognition_nodoc.py --eval --results nodoc.results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from common import read_jsonl, write_jsonl

RESULTS = Path(__file__).resolve().parent.parent / "results"
DOC = re.compile(r'""".*?"""', re.S)


def strip_doc(text: str) -> str:
    return DOC.sub('""""""', text)


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def make() -> int:
    reqs = [r for r in read_jsonl(RESULTS / "humaneval_recognition_requests.jsonl")
            if r["split"] == "dev"]
    out = []
    for q in reqs:
        q = json.loads(json.dumps(q))
        q["uid"] += "-nodoc"
        q["state"]["problem"] = strip_doc(q["state"]["problem"])
        for k in ("A", "B"):
            q["state"]["options"][k] = strip_doc(q["state"]["options"][k])
        out.append(q)
    write_jsonl(RESULTS / "humaneval_recognition_nodoc_requests.jsonl", out)
    print(f"wrote {len(out)} requests -> {RESULTS / 'humaneval_recognition_nodoc_requests.jsonl'}")
    return 0


def evaluate(results: Path) -> int:
    reqs = read_jsonl(RESULTS / "humaneval_recognition_nodoc_requests.jsonl")
    res = {r["uid"]: r for r in read_jsonl(results)}
    rows = []
    for q in reqs:
        r = res.get(q["uid"])
        if not r or "probs" not in r:
            continue
        p = r["probs"]
        pick = max(q["state"]["options"], key=lambda k: (p.get(k, 0.0), k))
        rows.append({"task_id": q["task_id"], "canonical_option": q["canonical_option"],
                     "pick": pick, "correct": pick == q["canonical_option"],
                     "p_canonical": p.get(q["canonical_option"], 0.0)})
    k, n = sum(r["correct"] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    out = {"variant": "no-docstring control", "n": n, "correct": k,
           "rate": k / n if n else None, "wilson95": [lo, hi],
           "mean_p_canonical": sum(r["p_canonical"] for r in rows) / n if n else None,
           "items": rows}
    dest = RESULTS / "humaneval_recognition_nodoc_result.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"no-doc control: {k}/{n} = {k/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"wrote {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--results", default=str(RESULTS / "humaneval_recognition_nodoc_results.jsonl"))
    a = ap.parse_args()
    if a.make:
        return make()
    if a.eval:
        return evaluate(Path(a.results))
    ap.error("pass --make or --eval")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
