"""MBPP execution: official test_list assertions in the same sandbox as HumanEval.

    python3 mbpp_exec.py --check-canonical
    python3 mbpp_exec.py --score dream-base
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import WORK, read_jsonl
from humaneval_exec import PASS_SENTINEL, _run_source, extract_code, parsable_prefix

HERE = WORK / "mbpp"


def problems() -> dict[str, dict]:
    return {str(r["task_id"]): r for r in json.loads((HERE / "mbpp.json").read_text(encoding="utf-8"))}


def passes(code: str, p: dict) -> tuple[bool, str]:
    imports = "\n".join(p.get("test_imports") or [])
    tests = "\n".join(p["test_list"])
    src = f"{imports}\n{code}\n\n{tests}\nprint({PASS_SENTINEL!r})\n"
    ok, out = _run_source(src)
    return ok and PASS_SENTINEL in out, out


def check_canonical() -> int:
    ps = problems()
    ok = 0
    for uid, p in sorted(ps.items()):
        good, out = passes(p["code"], p)
        if not good:
            print(f"FAIL {uid}\n{out[-400:]}")
        ok += good
    print(f"canonical: {ok}/{len(ps)}")
    return 0 if ok == len(ps) else 1


def score(model: str) -> Path:
    ps = problems()
    rows = read_jsonl(HERE / "candidates" / model / "test.jsonl")
    out = HERE / "exec" / model / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{model}: {len(done)} done, {len(todo)} to score")
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        p = ps[r["uid"]]
        passed = []
        for c in r["candidates"]:
            code = parsable_prefix(extract_code(c["text"]))
            ok, _ = passes(code, p)
            passed.append(ok)
        rec = {"uid": r["uid"], "passed": passed}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 5 == 0:
            print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-canonical", action="store_true")
    ap.add_argument("--score", default=None)
    a = ap.parse_args()
    if a.check_canonical:
        return check_canonical()
    if a.score:
        print(score(a.score))
        return 0
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
