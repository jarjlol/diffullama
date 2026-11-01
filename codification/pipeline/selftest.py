"""CPU-only wiring test of the whole harness. Touches nothing in the repo.

    python3 selftest.py

Copies rerank_task/ to a temp dir (CODIFY_TASK_DIR), uses a temp work dir
(CODIFY_WORK_DIR), writes synthetic GSM8K-style questions, then runs the real
scripts end to end with the mock generator and mock scorer:
  generate -> score -> assemble -> eval.sh dev/test -> baselines -> analyze,
plus CKA property tests and a check that eval.py zeroes a broken solution.
The accuracy numbers it prints are meaningless; only pass/fail matters.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "rerank_task"
PY = sys.executable


def run(args, env, cwd=HERE):
    r = subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAIL {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def cka_tests():
    from cka import cka_reference, cka_torch
    rng = random.Random(0); n, d = 40, 6
    X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    R = [[rng.gauss(0, 1) for _ in range(d + 2)] for _ in range(n)]
    S = [[2.5 * v + 1.0 for v in row] for row in X]          # scale + shift
    assert abs(cka_reference(X, X) - 1) < 1e-9
    assert abs(cka_reference(X, S) - 1) < 1e-9
    assert cka_reference(X, R) < 0.5
    try:
        import torch
    except ImportError:
        return "CKA reference OK (torch absent: torch version not cross-checked)"
    for A, B in ((X, R), (X, S)):
        t, ref = cka_torch(torch.tensor(A), torch.tensor(B)), cka_reference(A, B)
        assert math.isclose(t, ref, rel_tol=1e-9), (t, ref)
    return "CKA reference and torch agree; identity/scale invariance OK"


def main() -> int:
    print(cka_tests())
    tmp = Path(tempfile.mkdtemp(prefix="codify-selftest-"))
    try:
        task_dir, work = tmp / "rerank_task", tmp / "work"
        shutil.copytree(PACK, task_dir, ignore=shutil.ignore_patterns("items.jsonl", "__pycache__"))
        env = dict(os.environ, CODIFY_TASK_DIR=str(task_dir), CODIFY_WORK_DIR=str(work), PYTHONHASHSEED="0")
        rng = random.Random(1)
        for split, n in (("dev", 40), ("test", 60)):
            rows = []
            for i in range(n):
                a, b = rng.randint(2, 50), rng.randint(2, 50)
                rows.append({"uid": f"syn-{split}-{i:03d}", "question": f"Tom has {a} apples and buys {b}. How many?", "answer": str(a + b)})
            (work / "data").mkdir(parents=True, exist_ok=True)
            (work / "data" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (task_dir / "data" / split).mkdir(parents=True, exist_ok=True)
            (task_dir / "data" / split / "gold.jsonl").write_text(
                "".join(json.dumps({"uid": r["uid"], "answer": r["answer"]}) + "\n" for r in rows), encoding="utf-8")
        (work / "data" / "shots.json").write_text(json.dumps([{"question": "1+1?", "answer": "1+1=2\n#### 2"}]), encoding="utf-8")

        for split in ("dev", "test"):
            run([PY, "generate_candidates.py", "--model", "mock", "--split", split], env)
            run([PY, "score_candidates.py", "--model", "mock", "--split", split, "--scorer", "mock"], env)
            run([PY, "assemble.py", "--split", split], env)
            out = run(["bash", str(task_dir / "eval.sh"), split], env)
            assert sum(l.startswith("score: ") for l in out.splitlines()) == 1, out
            print(f"eval.sh {split}: {[l for l in out.splitlines() if l.startswith('score: ')][0]}")
        bl = json.loads(run([PY, str(task_dir / "baselines.py"), "--split", "test", "--json"], env, cwd=task_dir))
        r = bl["mock"]
        assert r["oracle"] >= max(v for k, v in r.items() if not k.startswith("_") and k != "n") - 1e-12, "a selector beat the oracle"
        print(f"baselines: first {r['first']:.3f}  majority {r['majority']:.3f}  oracle {r['oracle']:.3f}")

        (work / "preservation.json").write_text(json.dumps({"mock": {"mean": 0.8, "per_layer": [0.8]}}), encoding="utf-8")
        an = run([PY, "analyze.py", "--split", "test", "--json"], env)
        assert json.loads(an)["per_scorer"]["ar.mock.sum"]["n_models"] == 1
        from analyze import spearman
        assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12 and abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
        print("analyze: join + Spearman OK")

        exp = run(["bash", str(HERE.parent / "make_arbor_repo.sh"), str(tmp / "arbor_repo")], env)
        assert (tmp / "arbor_repo" / ".git").is_dir() and "correctness: PASS" in exp
        st = subprocess.run(["git", "status", "--porcelain"], cwd=tmp / "arbor_repo", capture_output=True, text=True).stdout
        assert st == "", f"exported repo not clean: {st}"
        print("make_arbor_repo.sh: clean git repo, seed passes eval -- OK")

        good = (task_dir / "solution.py").read_text(encoding="utf-8")
        (task_dir / "solution.py").write_text("import os\n" + good, encoding="utf-8")
        assert "score: 0.0" in run(["bash", str(task_dir / "eval.sh"), "dev"], env)
        print("leakage gate: 'import os' in solution.py scores 0.0 -- OK")
        (task_dir / "solution.py").write_text("def select(item):\n    return 99\n", encoding="utf-8")
        out = run(["bash", str(task_dir / "eval.sh"), "dev"], env)
        assert "score: 0.0" in out, out
        print("correctness gate: out-of-range index scores 0.0 -- OK")
        assert not list(PACK.glob("data/*/items.jsonl")), "selftest must not write into the real pack"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
