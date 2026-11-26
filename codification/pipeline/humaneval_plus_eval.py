"""Re-execute the existing HumanEval candidates against HumanEval+ (issue #84).

The candidates, AR features and decision scores are unchanged. Only the test
suite changes: a candidate "passes" iff the HumanEval+ `test` runs to completion
with the same sentinel executor used for standard HumanEval, after the same
longest-parsable-prefix assembly. MBR-exec choices are also test-independent
(they cluster docstring-example outputs), so they are reused.

    python3 humaneval_plus_eval.py --plus work/humaneval/HumanEvalPlus.test.jsonl

Problems whose HumanEval+ canonical solution does not pass are excluded from
every denominator and reported, so standard and Plus are compared on the same n.

The Plus tests import numpy for some problems; run under an interpreter that has
it (on blackwell: /home/antpc/venvs/llada/bin/python).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import WORK, read_jsonl
from humaneval_exec import _limits, candidate_code, problems as standard_problems

PASS_SENTINEL = "__HE_PASS__"

HERE = WORK / "humaneval"
RESULTS = Path(__file__).resolve().parent.parent / "results"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def plus_problems(path: Path) -> dict[str, dict]:
    return {r["task_id"]: r for r in read_jsonl(path)}


def _run_plus(source: str, timeout: float = 5.0) -> tuple[bool, str]:
    """The humaneval_exec sandbox plus single-threaded BLAS.

    HumanEval+ tests import numpy; on a 128-core host OpenBLAS tries to reserve
    per-thread buffers and dies inside the 2 GB address-space limit before the
    candidate is exercised. Thread counts are a resource knob, not a test
    change, so they are pinned to 1 here; every other limit is unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="he_plus_") as d:
        p = Path(d) / "candidate.py"
        p.write_text(source, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "LC_ALL": "C.UTF-8",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, preexec_fn=_limits)
            return r.returncode == 0, r.stdout.decode("utf-8", "replace")[-4000:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"


def plus_passes(code: str, problem: dict, timeout: float = 5.0) -> tuple[bool, str]:
    src = (f"{code}\n\n{problem['test']}\n\n"
           f"try:\n    check({problem['entry_point']})\nexcept BaseException:\n    raise\n"
           f"print({PASS_SENTINEL!r})\n")
    ok, out = _run_plus(src, timeout)
    return ok and PASS_SENTINEL in out, out


def load_jsonl_map(path: Path, key: str = "uid") -> dict[str, dict]:
    return {r[key]: r for r in read_jsonl(path)} if path.exists() else {}


def feature_rows(model: str, split: str = "test") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    d = HERE / "features" / model
    for f in sorted(d.glob(f"{split}.*.jsonl")):
        for r in read_jsonl(f):
            merged = out.setdefault(r["uid"], [{} for _ in r["features"]])
            for j, fs in enumerate(r["features"]):
                merged[j].update(fs)
    return out


def best_index(feats: list[dict], key: str, normalise: bool = False) -> int | None:
    vals = []
    for i, f in enumerate(feats):
        v = f.get(key)
        n = f.get(key.replace("nll_sum", "n_tokens"))
        if v is None or (normalise and not n):
            continue
        vals.append((v / n if normalise else v, i))
    return min(vals)[1] if vals else None


def top_index(feats: list[dict], key: str) -> int | None:
    vals = [(f.get(key), i) for i, f in enumerate(feats)]
    vals = [(v, i) for v, i in vals if v is not None]
    return max(vals, key=lambda x: (x[0], -x[1]))[1] if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--plus", default=str(HERE / "HumanEvalPlus.test.jsonl"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--out", default=str(RESULTS / "humaneval_plus_result.json"))
    a = ap.parse_args()

    sp = standard_problems()
    pp = plus_problems(Path(a.plus))
    print(f"standard {len(sp)} problems; plus {len(pp)} problems", flush=True)

    # 1. Canonical gate: the Plus tests must pass the Plus reference solution.
    def canonical_one(tid: str) -> tuple[str, bool, str]:
        p = pp[tid]
        code = p["prompt"] + p["canonical_solution"]
        ok, out = plus_passes(code, p, timeout=a.timeout)
        return tid, ok, out

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        canon = list(ex.map(canonical_one, sorted(pp)))
    bad = [(tid, out) for tid, ok, out in canon if not ok]
    print(f"plus canonical: {len(canon)-len(bad)}/{len(canon)}", flush=True)
    for tid, out in bad[:5]:
        print(f"  FAIL {tid}: {out[-300:]}", flush=True)
    keep = {tid for tid, ok, _ in canon if ok}

    # 2. Run every candidate on the Plus test.
    rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
    rows = [r for r in rows if r["uid"] in keep]
    jobs = []
    for r in rows:
        p_std = sp[r["uid"]]
        p_plus = pp[r["uid"]]
        for j, c in enumerate(r["candidates"]):
            code = candidate_code(c["text"], p_std["prompt"], p_std["entry_point"])
            jobs.append((r["uid"], j, code, p_plus))

    def one(job):
        uid, j, code, p = job
        ok, _ = plus_passes(code, p, timeout=a.timeout)
        return uid, j, ok

    print(f"{len(rows)} items, {len(jobs)} candidate programs", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        got = list(ex.map(one, jobs))
    plus_pass: dict[str, dict[int, bool]] = {}
    for uid, j, ok in got:
        plus_pass.setdefault(uid, {})[j] = ok

    # 3. Selections are unchanged; recompute their correctness under Plus.
    std_exec = load_jsonl_map(HERE / "exec" / a.model / f"{a.split}.jsonl")
    feats = feature_rows(a.model, a.split)
    mbr = {r["uid"]: r["choice"] for r in
           read_jsonl(HERE / "mbr" / a.model / f"{a.split}.jsonl")} if (
        HERE / "mbr" / a.model / f"{a.split}.jsonl").exists() else {}

    per_item = []
    for r in rows:
        uid = r["uid"]
        passed = [bool(plus_pass[uid][j]) for j in range(len(r["candidates"]))]
        std_passed = std_exec.get(uid, {}).get("passed", [])
        examples = std_exec.get(uid, {}).get("example_passed", [])
        n = len(passed)
        rec = {"uid": uid, "n": n, "plus_passed": passed, "std_passed": std_passed,
               "oracle": any(passed), "std_oracle": any(std_passed),
               "random(exp.)": sum(passed) / n, "correct": {}, "choice": {}}

        def put(name: str, j: int | None) -> None:
            rec["choice"][name] = j
            rec["correct"][name] = bool(passed[j]) if j is not None and j < n else False

        put("first", 0)
        if examples:
            j = next((k for k, ok in enumerate(examples) if ok), 0)
        else:
            j = 0
        put("example-filter", j)
        if uid in mbr:
            put("mbr-exec", mbr[uid])
        fs = feats.get(uid)
        if fs is not None and len(fs) != n:
            raise SystemExit(f"{uid}: {len(fs)} feature rows vs {n} candidates")
        fs = fs if fs is not None else [{} for _ in range(n)]
        keys = {k for f in fs for k in f}
        for k in sorted(keys):
            if k.startswith("ar.") and k.endswith(".nll_sum"):
                s = k[len("ar."):-len(".nll_sum")]
                put(f"ar.{s}.sum", best_index(fs, k, False))
                put(f"ar.{s}.mean", best_index(fs, k, True))
            elif k == "decision.listwise.p":
                put("decision.listwise", top_index(fs, k))
            elif k == "decision.pointwise.p_yes":
                put("decision.pointwise", top_index(fs, k))
        per_item.append(rec)

    names = sorted({n for r in per_item for n in r["correct"]})
    table = {n: sum(r["correct"][n] for r in per_item) / len(per_item) for n in names}
    table["random(exp.)"] = sum(r["random(exp.)"] for r in per_item) / len(per_item)
    table["oracle"] = sum(r["oracle"] for r in per_item) / len(per_item)
    table["n"] = len(per_item)

    std_table = {n: sum(bool(r["std_passed"][r["choice"][n]]) for r in per_item
                        if r["choice"].get(n) is not None and r["std_passed"]) / len(per_item)
                 for n in names}
    std_table["random(exp.)"] = sum(sum(r["std_passed"]) / len(r["std_passed"])
                                    for r in per_item if r["std_passed"]) / len(per_item)
    std_table["oracle"] = sum(bool(r["std_oracle"]) for r in per_item) / len(per_item)
    std_table["n"] = len(per_item)

    # Conditional view: on Plus-solvable items, how often does each selector pick a pass?
    solvable = [r for r in per_item if r["oracle"]]
    conditional = {n: sum(r["correct"][n] for r in solvable) / len(solvable) for n in names}
    conditional["n"] = len(solvable)
    conditional["first"] = sum(r["correct"]["first"] for r in solvable) / len(solvable)

    out = {
        "split": a.split, "model": a.model, "n_canonical_plus": len(keep),
        "excluded_canonical_fail": [{"task_id": tid, "tail": o[-500:]} for tid, o in bad],
        "n_items": len(per_item), "plus": table, "standard": std_table,
        "drop_plus_minus_standard": {n: table[n] - std_table[n] for n in names},
        "plus_solvable": conditional,
        "n_plus_solvable": len(solvable),
        "items": per_item,
        "note": "Selections are unchanged from standard HumanEval; only the test "
                "suite changes. MBR-exec choices cluster docstring examples and are "
                "therefore test-independent. HumanEval+ is stronger tests, not fresh "
                "problems, so a memorisation caveat still applies.",
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n{'selector':24s} {'standard':>9s} {'plus':>9s} {'drop':>9s}")
    for n in names:
        print(f"{n:24s} {std_table[n]:9.3f} {table[n]:9.3f} {table[n]-std_table[n]:+9.3f}")
    print(f"{'oracle':24s} {std_table['oracle']:9.3f} {table['oracle']:9.3f}")
    print(f"n={len(per_item)}  plus-solvable={len(solvable)}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
