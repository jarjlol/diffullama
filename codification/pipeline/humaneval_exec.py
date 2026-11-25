"""HumanEval execution: the only ground truth in the code arm.

    python3 humaneval_exec.py --check-canonical      # must print 164/164
    python3 humaneval_exec.py --score dream-base     # writes exec/<model>/test.jsonl

A candidate passes iff its code, plus the official test, defines the entry point
and `check(entry_point)` runs to completion inside a fresh subprocess. A sentinel
printed only after `check` returns is required, so `sys.exit(0)` at import or
inside `check` cannot fake a pass.

Sandbox: 5 s CPU limit, 2 GB address space, 5 s wall clock, isolated interpreter
(`-I`), temp cwd, env scrubbed. This is a fault isolator, not a security
boundary; state that in the report. Results are cached by uid.
"""
from __future__ import annotations

import argparse
import ast
import doctest
import json
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import WORK, read_jsonl

HERE = WORK / "humaneval"
PASS_SENTINEL = "__HE_PASS__"


def problems() -> dict[str, dict]:
    return {r["task_id"]: r for r in read_jsonl(HERE / "HumanEval.jsonl")}


def clean_completion(text: str) -> str:
    """Drop markdown fences, preserving the continuation's own indentation."""
    t = text.strip("\n")
    t = re.sub(r"^\s*```[a-zA-Z]*[ \t]*\n", "", t)
    t = re.sub(r"\n```[ \t]*$", "", t)
    return t


def extract_code(text: str) -> str:
    """From the first Python definition/import onward; body-only text is returned as-is."""
    t = clean_completion(text)
    lines = t.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^(def |async def |from [A-Za-z_]|import [A-Za-z_]|@)", line):
            return "\n".join(lines[i:])
    return t


def parsable_prefix(code: str) -> str:
    """Longest line prefix that parses, so trailing prose is dropped, not executed."""
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return code


def defines(code: str, entry: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry
               for n in tree.body)


def candidate_code(text: str, prompt: str, entry: str) -> str:
    """Assemble the candidate into a complete function.

    Base completion models continue after the signature+docstring, so the body is
    appended to the prompt. If the model repeats the whole function but omits the
    prompt's imports, the header is prepended. Detection uses a top-level
    FunctionDef, not a substring.
    """
    body = clean_completion(text)
    code = parsable_prefix(extract_code(body))
    if defines(code, entry):
        header = prompt.split(f"def {entry}", 1)[0].rstrip()
        if header and "import" in header and "import" not in code:
            code = header + "\n\n" + code
        return parsable_prefix(code)
    return parsable_prefix(prompt.rstrip("\n") + "\n" + body)


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))


def _run_source(source: str, timeout: float = 5.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="he_") as d:
        p = Path(d) / "candidate.py"
        p.write_text(source, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "LC_ALL": "C.UTF-8"}
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
    src = (f"{code}\n\n{problem['test']}\n\n"
           f"try:\n    check({problem['entry_point']})\nexcept BaseException:\n    raise\n"
           f"print({PASS_SENTINEL!r})\n")
    ok, out = _run_source(src)
    return ok and PASS_SENTINEL in out, out


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
    sys.exit(1)
bad = 0
for source, want in payload["examples"]:
    try:
        got = eval(source, ns)
    except Exception:
        bad += 1
        continue
    if want and repr(got) != want:
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
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "LC_ALL": "C.UTF-8"}
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
    """Parse only the docstring, so the closing triple quote is not part of `want`."""
    m = re.search(r'"""(.*?)"""', problem["prompt"], re.S)
    doc = m.group(1) if m else problem["prompt"]
    return [(e.source, e.want.strip("\n")) for e in doctest.DocTestParser().get_examples(doc)]


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
        examples = examples_for(p)
        passed, example_passed = [], []
        for c in r["candidates"]:
            code = candidate_code(c["text"], p["prompt"], p["entry_point"])
            ok, _ = passes(code, p)
            passed.append(ok)
            example_passed.append(passes_examples(code, examples, p["entry_point"]))
        rec = {"uid": r["uid"], "passed": passed, "example_passed": example_passed}
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
