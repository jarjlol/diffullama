"""HumanEval execution: the only ground truth in the code arm.

    python3 humaneval_exec.py --check-canonical      # must print 164/164
    python3 humaneval_exec.py --score dream-base     # writes exec/<model>/test.jsonl

A candidate passes iff its extracted code, plus the official test, defines the
entry point and `check(entry_point)` exits 0 inside a fresh subprocess:

  * 5 s CPU limit, 2 GB address space, 5 s wall clock;
  * isolated interpreter (`-I`), temp cwd, env scrubbed;
  * no network is attempted by the official tests, and the subprocess has no
    credentials in its environment.

This is a fault isolator, not a security boundary; state that in the report.
Execution results are cached by (model, split, candidate text hash), so the
selectors can be re-evaluated without re-running 1,440 programs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import WORK, read_jsonl

HERE = WORK / "humaneval"


def problems() -> dict[str, dict]:
    return {r["task_id"]: r for r in read_jsonl(HERE / "HumanEval.jsonl")}


def extract_code(text: str) -> str:
    """Drop markdown fences and any prose before the first Python definition."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    lines = t.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^(def |async def |from [A-Za-z_]|import [A-Za-z_]|@)", line):
            return "\n".join(lines[i:])
    return t


def candidate_code(text: str, prompt: str, entry: str) -> str:
    """The model's completion, completed with the prompt when it is body-only.

    Base completion models continue the signature+docstring instead of repeating
    it, so the candidate is prompt + continuation. Models that do repeat the
    definition are used as returned.
    """
    code = extract_code(text)
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    entry = m.group(1) if m else entry
    if entry and f"def {entry}" not in code:
        code = prompt.rstrip("\n") + "\n" + text.strip("\n")
    return code


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))


def _run_source(source: str, timeout: float = 5.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="he_") as d:
        p = Path(d) / "candidate.py"
        p.write_text(source, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8"}
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, preexec_fn=_limits)
            return r.returncode == 0, r.stdout.decode("utf-8", "replace")[-2000:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:                                      # OSError, memory, ...
            return False, f"{type(e).__name__}: {e}"


def passes(code: str, problem: dict) -> tuple[bool, str]:
    src = f"{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    return _run_source(src)


def check_canonical() -> int:
    ps = problems()
    ok = 0
    for tid, p in sorted(ps.items()):
        good, out = passes(p["prompt"] + p["canonical_solution"], p)
        if not good:
            print(f"FAIL {tid}\n{out[-500:]}")
        ok += good
    print(f"canonical: {ok}/{len(ps)}")
    return 0 if ok == len(ps) else 1


EXAMPLES_DRIVER = r'''
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
ns = {"__name__": "__main__"}
exec(payload["code"], ns)
fn = ns.get(payload["entry"])
if fn is None:
    print("missing entry point"); sys.exit(1)
bad = 0
for source, want in payload["examples"]:
    try:
        got = eval(source, ns)
        if repr(got) != want:
            bad += 1
    except Exception:
        bad += 1
sys.exit(0 if bad == 0 else 1)
'''


def passes_examples(code: str, examples: list[tuple[str, str]], entry: str,
                    timeout: float = 3.0) -> bool:
    if not examples:
        return False
    with tempfile.TemporaryDirectory(prefix="he_ex_") as d:
        payload = Path(d) / "payload.json"
        payload.write_text(json.dumps({"code": code, "entry": entry,
                                       "examples": [list(e) for e in examples]}), encoding="utf-8")
        driver = Path(d) / "driver.py"
        driver.write_text(EXAMPLES_DRIVER, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8"}
        try:
            r = subprocess.run([sys.executable, "-I", str(driver), str(payload)], cwd=d, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, preexec_fn=_limits)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def examples_for(problem: dict) -> list[tuple[str, str]]:
    import doctest
    return [(e.source, e.want) for e in doctest.DocTestParser().get_examples(problem["prompt"])]


def score(model: str, split: str = "test") -> Path:
    ps = problems()
    rows = read_jsonl(HERE / "candidates" / model / f"{split}.jsonl")
    out = HERE / "exec" / model / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"]: r for r in read_jsonl(out)} if out.exists() else {}
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{model}/{split}: {len(done)} done, {len(todo)} to score")
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        p = ps[r["uid"]]
        passed, examples = [], []
        for j, c in enumerate(r["candidates"]):
            code = candidate_code(c["text"], p["prompt"], p["entry_point"])
            ok, _ = passes(code, p)
            passed.append(ok)
            examples.append(passes_examples(code, examples_for(p), p["entry_point"]))
        rec = {"uid": r["uid"], "passed": passed, "example_passed": examples}
        done[r["uid"]] = rec
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 5 == 0:
            print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-canonical", action="store_true")
    ap.add_argument("--score", default=None, help="model name")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()
    if a.check_canonical:
        return check_canonical()
    if a.score:
        print(score(a.score, a.split))
        return 0
    ap.error("nothing to do: pass --check-canonical or --score MODEL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
