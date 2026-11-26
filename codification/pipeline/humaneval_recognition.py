"""HumanEval recognition probe (issue #81, part a).

One canonical solution and one *verified-failing* single-token mutant are shown
as options A/B; the decision model is asked which one passes the hidden tests.
This is a **recognition** probe: both programs are in context, so it measures
whether the model can discriminate a correct implementation from a subtly wrong
one, not whether it can recall a solution from the docstring alone.

    python3 humaneval_recognition.py --make --splits dev test --n-test 40
    # score requests on the host with network access, then:
    python3 humaneval_recognition.py --eval --results recognition.results.jsonl

The mutation search changes exactly one operator or numeric constant in the
function body and keeps the change only if the official HumanEval test passes on
the canonical program and fails on the mutant. Preference is given to mutants
that fail an assertion (wrong output) over mutants that raise, so the distractor
is not obviously broken. Every attempt is recorded in the metadata.

The executor is the sentinel subprocess from humaneval_exec.py, reimplemented
here with per-limit error handling so the same script runs on macOS as well as
Linux (macOS rejects RLIMIT_AS below the current process size).
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import random
import re
import resource
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

from common import WORK, read_jsonl, write_jsonl

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PASS_SENTINEL = "__HE_PASS__"
INSTRUCTIONS = "Which option correctly implements the function and will pass the hidden tests?"

# One-token substitutions. Structural tokens (brackets, commas, dots, ...) are
# deliberately absent.
OP_SUB = {
    "+": "-", "-": "+", "*": "//", "//": "*", "/": "//", "%": "*", "**": "*",
    "<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==",
    "and": "or", "or": "and",
    "+=": "-=", "-=": "+=", "*=": "//=", "//=": "*=", "/=": "//=", "%=": "*=",
    "&": "|", "|": "&", "^": "&", "<<": ">>", ">>": "<<",
}
CMP_SUB = {"in": "not in", "is": "is not"}
# Fallback only for one-line bodies with no operator or constant at all.
# Reported separately as kind="call"; never silently mixed into the main class.
CALL_SUB = {
    "swapcase": "lower", "lower": "upper",
    "startswith": "endswith", "endswith": "startswith",
    "max": "min", "min": "max", "sorted": "reversed",
    "len": "id", "md5": "sha1", "sha1": "md5",
    "join": "split", "set": "list", "list": "set",
}


def problems(path: Path | None = None) -> dict[str, dict]:
    p = path or (WORK / "humaneval" / "HumanEval.jsonl")
    return {r["task_id"]: r for r in read_jsonl(p)}


def splits() -> dict[str, str]:
    return json.loads((RESULTS / "humaneval_splits.json").read_text(encoding="utf-8"))


def _limits() -> None:
    """Best-effort resource caps; a platform that rejects one still runs the rest."""
    for what, value in ((resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_CPU, 5),
                        (resource.RLIMIT_FSIZE, 8 << 20)):
        try:
            resource.setrlimit(what, (value, value))
        except (OSError, ValueError):
            pass


def _run(source: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="he_rec_") as d:
        p = Path(d) / "candidate.py"
        p.write_text(source, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "LC_ALL": "C.UTF-8"}
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, preexec_fn=_limits)
            return r.returncode == 0, r.stdout.decode("utf-8", "replace")[-4000:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:  # noqa: BLE001 - surface, never fake a pass
            return False, f"{type(e).__name__}: {e}"


def passes(code: str, problem: dict, timeout: float = 5.0) -> tuple[bool, str]:
    src = (f"{code}\n\n{problem['test']}\n\n"
           f"try:\n    check({problem['entry_point']})\nexcept BaseException:\n    raise\n"
           f"print({PASS_SENTINEL!r})\n")
    ok, out = _run(src, timeout)
    return ok and PASS_SENTINEL in out, out


def _body_start_line(full_code: str, entry: str) -> int:
    """First line of the function body (excludes signature, docstring and defaults)."""
    tree = ast.parse(full_code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            if not node.body:
                return node.lineno + 1
            return node.body[0].lineno
    raise ValueError(f"{entry} not found")


def mutation_sites(full_code: str, entry: str) -> list[dict]:
    """Every single-token operator/constant substitution in the function body."""
    start = _body_start_line(full_code, entry)
    sites: list[dict] = []
    lines = full_code.splitlines(keepends=True)
    for tok in tokenize.generate_tokens(io.StringIO(full_code).readline):
        if tok.start[0] < start:
            continue
        kind = old = new = None
        if tok.type == tokenize.OP and tok.string in OP_SUB:
            kind, old, new = "op", tok.string, OP_SUB[tok.string]
        elif tok.type == tokenize.NAME and tok.string in ("and", "or"):
            kind, old, new = "bool", tok.string, OP_SUB[tok.string]
        elif tok.type == tokenize.NAME and tok.string in CMP_SUB:
            kind, old, new = "cmp", tok.string, CMP_SUB[tok.string]
        elif tok.type == tokenize.NAME and tok.string in ("None", "True", "False"):
            kind, old = "const", tok.string
            new = {"None": "0", "True": "False", "False": "True"}[tok.string]
        elif tok.type == tokenize.NAME and tok.string in CALL_SUB:
            kind, old, new = "call", tok.string, CALL_SUB[tok.string]
        elif tok.type == tokenize.NUMBER:
            try:
                val = ast.literal_eval(tok.string)
            except Exception:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            if isinstance(val, int):
                new = str(1 if val == 0 else val + 1)
            else:
                new = repr(val + 1.0)
            kind, old = "const", tok.string
        elif tok.type == tokenize.STRING:
            try:
                val = ast.literal_eval(tok.string)
            except Exception:
                continue
            if not isinstance(val, str):
                continue
            if val == "":
                new = repr(" ")
            elif val == " ":
                new = repr("")
            elif val.lower() in ("ascii", "utf-8", "utf8"):
                new = repr("utf-8" if val.lower() == "ascii" else "ascii")
            else:
                new = repr(val + "x")
            kind, old = "string", tok.string
        if kind is None:
            continue
        row, col = tok.start
        end_row, end_col = tok.end
        if row != end_row:  # multi-line tokens are not single substitutions
            continue
        line = lines[row - 1]
        mutated = line[:col] + new + line[end_col:]
        sites.append({"kind": kind, "old": old, "new": new, "line": row, "col": col,
                      "line_before": line.strip(), "line_after": mutated.strip(),
                      "source": "".join(lines[: row - 1]) + mutated + "".join(lines[row:])})
    return sites


def _rank(site: dict) -> int:
    """Prefer comparisons, then arithmetic, then booleans, constants, call names."""
    if site["kind"] == "cmp":
        return 0
    if site["kind"] == "op":
        return 0 if site["old"] in ("<", "<=", ">", ">=", "==", "!=") else 1
    if site["kind"] == "bool":
        return 2
    if site["kind"] in ("const", "string"):
        return 3
    return 4  # call fallback


def make_mutant(problem: dict, seed: int, attempts: int, timeout: float) -> dict | None:
    """Return a verified canonical/mutant pair, or None if no mutation worked."""
    entry = problem["entry_point"]
    canonical = problem["prompt"] + problem["canonical_solution"]
    ok, out = passes(canonical, problem, timeout)
    if not ok:
        return {"task_id": problem["task_id"], "error": "canonical failed", "output": out[-1000:]}

    try:
        sites = mutation_sites(canonical, entry)
    except (SyntaxError, ValueError) as e:
        return {"task_id": problem["task_id"], "error": f"no mutation sites: {e}"}

    rng = random.Random(f"mut|{seed}|{problem['task_id']}")
    rng.shuffle(sites)
    sites.sort(key=_rank)  # stable: rank first, seed order within a rank
    tried = []
    fallback = None
    for site in sites[:attempts]:
        try:
            ast.parse(site["source"])
        except SyntaxError:
            continue
        good, mout = passes(site["source"], problem, timeout)
        rec = {k: site[k] for k in ("kind", "old", "new", "line", "col",
                                    "line_before", "line_after")}
        if good:
            tried.append({**rec, "result": "still passes"})
            continue
        assert_fail = "AssertionError" in mout
        rec["result"] = "assert-fail" if assert_fail else "raises/timeout"
        rec["failure_tail"] = mout[-400:]
        tried.append(rec)
        if assert_fail:
            return {**rec, "task_id": problem["task_id"], "canonical": canonical,
                    "mutant": site["source"], "attempts": tried, "fallback": False,
                    "canonical_passes": True, "mutant_passes": False,
                    "mutant_raised": False}
        if fallback is None:
            fallback = {**rec, "task_id": problem["task_id"], "canonical": canonical,
                        "mutant": site["source"], "attempts": tried, "fallback": True,
                        "canonical_passes": True, "mutant_passes": False,
                        "mutant_raised": "AssertionError" not in mout}
    if fallback is not None:
        return fallback
    return {"task_id": problem["task_id"], "error": "no failing mutation found",
            "attempts": tried}


def request_for(rec: dict, problem: dict, split: str, seed: int) -> dict:
    flip = random.Random(f"order|{seed}|{problem['task_id']}").random() < 0.5
    canonical_letter = "B" if flip else "A"
    mutant_letter = "A" if flip else "B"
    options = {canonical_letter: rec["canonical"], mutant_letter: rec["mutant"]}
    return {
        "uid": f"he-recog-{split}-{problem['task_id'].replace('/', '-')}",
        "model": "humaneval-recognition",
        "split": split,
        "task_id": problem["task_id"],
        "canonical_option": canonical_letter,
        "mutant_option": mutant_letter,
        "state": {
            "problem": problem["prompt"],
            # Serialize in A/B order, not canonical-first: otherwise the JSON
            # order leaks which option is canonical regardless of the label.
            "options": {k: options[k] for k in ("A", "B")},
        },
        "questions": {"best": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {"A": "Option A", "B": "Option B"},
        }},
    }


def make(args: argparse.Namespace) -> int:
    ps = problems()
    sp = splits()
    chosen: list[tuple[str, str]] = []
    for split in args.splits:
        ids = sorted(t for t, s in sp.items() if s == split)
        n = args.n_dev if split == "dev" else args.n_test
        if n and n < len(ids):
            ids = random.Random(args.seed).sample(ids, n)
        chosen += [(split, t) for t in ids]
    if args.limit:
        chosen = chosen[: args.limit]

    requests, meta = [], []
    for k, (split, tid) in enumerate(chosen, 1):
        p = ps[tid]
        rec = make_mutant(p, args.seed, args.attempts, args.timeout)
        if rec is None or "error" in rec:
            print(f"  SKIP {tid}: {rec.get('error') if rec else 'unknown'}", flush=True)
            meta.append(rec or {"task_id": tid, "error": "unknown"})
            continue
        req = request_for(rec, p, split, args.seed)
        requests.append(req)
        meta.append({"task_id": tid, "split": split,
                     "canonical_option": req["canonical_option"],
                     "mutation": {kk: rec[kk] for kk in ("kind", "old", "new", "line", "col",
                                                          "line_before", "line_after")},
                     "mutation_result": rec["result"], "fallback": rec["fallback"],
                     "canonical_passes": rec["canonical_passes"],
                     "mutant_passes": rec["mutant_passes"], "mutant_raised": rec["mutant_raised"],
                     "attempts": rec["attempts"]})
        print(f"  {k}/{len(chosen)} {tid} {split}: {rec['kind']} {rec['old']}->{rec['new']} "
              f"line {rec['line']} ({rec['result']})", flush=True)

    req_path = RESULTS / "humaneval_recognition_requests.jsonl"
    meta_path = RESULTS / "humaneval_recognition_mutants.json"
    write_jsonl(req_path, requests)
    meta_path.write_text(json.dumps({"seed": args.seed, "instructions": INSTRUCTIONS,
                                     "problems": meta}, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"wrote {len(requests)} requests -> {req_path}")
    print(f"wrote metadata -> {meta_path}")
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def eval_results(args: argparse.Namespace) -> int:
    reqs = read_jsonl(RESULTS / "humaneval_recognition_requests.jsonl")
    res = {r["uid"]: r for r in read_jsonl(args.results)}
    meta = json.loads((RESULTS / "humaneval_recognition_mutants.json").read_text(encoding="utf-8"))
    mut = {m["task_id"]: m for m in meta["problems"] if "error" not in m}

    rows = []
    for q in reqs:
        r = res.get(q["uid"])
        if not r or "probs" not in r:
            continue
        probs = r["probs"]
        allowed = {q["canonical_option"], q["mutant_option"]}
        valid = {k: float(v) for k, v in probs.items() if k in allowed}
        if not valid:
            continue
        pick = max(valid, key=lambda k: (valid[k], k))
        canonical = q["canonical_option"]
        rows.append({"uid": q["uid"], "task_id": q["task_id"], "split": q["split"],
                     "canonical_option": canonical, "pick": pick,
                     "p_canonical": valid[canonical],
                     "p_mutant": valid[q["mutant_option"]],
                     "correct": pick == canonical,
                     "mutation_kind": mut.get(q["task_id"], {}).get("mutation", {}).get("kind"),
                     "latency_s": r.get("latency_s"), "usage": r.get("usage")})

    out = {"n_requests": len(reqs), "n_scored": len(rows), "instructions": INSTRUCTIONS,
           "model": args.model, "splits": {}}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        k = sum(r["correct"] for r in rs)
        n = len(rs)
        lo, hi = wilson(k, n)
        at_a = [r for r in rs if r["canonical_option"] == "A"]
        at_b = [r for r in rs if r["canonical_option"] == "B"]
        by_kind = {}
        for kind in sorted({r["mutation_kind"] for r in rs if r["mutation_kind"]}):
            kr = [r for r in rs if r["mutation_kind"] == kind]
            by_kind[kind] = {"n": len(kr), "correct": sum(r["correct"] for r in kr),
                             "rate": sum(r["correct"] for r in kr) / len(kr)}
        out["splits"][split] = {
            "n": n, "correct": k, "rate": k / n if n else None,
            "wilson95": [lo, hi],
            "mean_p_canonical": sum(r["p_canonical"] for r in rs) / n if n else None,
            "canonical_at_A": {"n": len(at_a), "correct": sum(r["correct"] for r in at_a),
                               "rate": sum(r["correct"] for r in at_a) / len(at_a) if at_a else None},
            "canonical_at_B": {"n": len(at_b), "correct": sum(r["correct"] for r in at_b),
                               "rate": sum(r["correct"] for r in at_b) / len(at_b) if at_b else None},
            "by_mutation_kind": by_kind,
            "items": rs,
        }
    out["pooled"] = {
        "n": len(rows), "correct": sum(r["correct"] for r in rows),
        "rate": sum(r["correct"] for r in rows) / len(rows) if rows else None,
    }
    dest = RESULTS / "humaneval_recognition_result.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for split, s in out["splits"].items():
        print(f"{split}: {s['correct']}/{s['n']} = {s['rate']:.3f}  "
              f"95% CI [{s['wilson95'][0]:.3f}, {s['wilson95'][1]:.3f}]  "
              f"A={s['canonical_at_A']['rate']:.3f} B={s['canonical_at_B']['rate']:.3f}")
    print(f"pooled: {out['pooled']['correct']}/{out['pooled']['n']} = {out['pooled']['rate']:.3f}")
    print(f"wrote {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--splits", nargs="*", default=["dev", "test"], choices=["dev", "test"])
    ap.add_argument("--n-dev", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--attempts", type=int, default=30,
                    help="max mutation sites tried per problem")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="first N problems (smoke test)")
    ap.add_argument("--results", default=str(RESULTS / "humaneval_recognition_results.jsonl"))
    ap.add_argument("--model", default="perplexity/pplx-decider-flash-v1.3:free")
    a = ap.parse_args()
    if a.make:
        return make(a)
    if a.eval:
        return eval_results(a)
    ap.error("pass --make or --eval")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
