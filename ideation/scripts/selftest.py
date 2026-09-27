"""Wiring test: run the whole pipeline on the mock backend in a scratch directory.

Mock output is placeholder text, so this proves only that every stage runs, every
prompt formats, every response parses, the gate and resume logic work, and a
deliverable renders. It says nothing about research quality. Stdlib only:

    python3 ideation/scripts/selftest.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
RUN = [sys.executable, str(HERE / "run_pipeline.py")]


def run(env, *extra):
    return subprocess.run(RUN + list(extra), env=env, capture_output=True, text=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ideation-selftest-"))
    env = {**os.environ, "IDEATION_DATA_DIR": str(tmp / "data"), "IDEATION_OUT_DIR": str(tmp / "out"),
           "PYTHONIOENCODING": "utf-8"}
    checks = []
    try:
        # 1. manual backend with no responses: must stop cleanly, exit 2, and write requests
        r = run(env)
        reqs = list((tmp / "data/llm/requests").glob("*.md"))
        checks.append(("manual backend stops with exit 2", r.returncode == 2))
        checks.append(("manual backend writes request files", len(reqs) >= 1))

        # 2. mock backend without a selection: must stop at the gate, exit 3
        r = run(env, "--backend", "mock")
        checks.append(("mock run stops at selection gate (exit 3)", r.returncode == 3))
        checks.append(("problem menu rendered", (tmp / "out/problem_candidates.md").exists()))

        # 3. supply a selection, run to completion
        (tmp / "data/selected_problems.json").write_text(
            json.dumps({"selected": ["P1", "P2"], "selected_by": "selftest", "reason": ""}), encoding="utf-8")
        r = run(env, "--backend", "mock")
        checks.append(("full mock run exits 0", r.returncode == 0))
        ranking = json.loads((tmp / "out/ranking.json").read_text(encoding="utf-8")) if r.returncode == 0 else {}
        cfg = json.loads((HERE.parent / "config.json").read_text(encoding="utf-8"))
        checks.append(("every selected problem ranked", sorted(ranking) == ["P1", "P2"]))
        checks.append((f"{cfg['n_ideas_per_problem']} ideas ranked per problem",
                       all(len(v) == cfg["n_ideas_per_problem"] for v in ranking.values())))
        doc = (tmp / "out/research_problems_and_ideas.md").read_text(encoding="utf-8") if r.returncode == 0 else ""
        checks.append((f"top {cfg['top_k_ideas']} ideas rendered per problem",
                       doc.count("### Idea ") == 2 * cfg["top_k_ideas"]))

        # 4. resume: a second run must make zero new model calls
        r = run(env, "--backend", "mock")
        checks.append(("re-run is fully cached (0 new calls)", "(0 new model calls)" in r.stdout))

        n_calls = len(list((tmp / "data/llm/responses").glob("*.md")))
        est = subprocess.run(RUN + ["--estimate"], capture_output=True, text=True).stdout
        expected = int(est.strip().splitlines()[-1].split(":")[1])
        checks.append((f"call count matches estimate ({n_calls} vs {expected})", n_calls == expected))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = sum(not ok for _, ok in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
