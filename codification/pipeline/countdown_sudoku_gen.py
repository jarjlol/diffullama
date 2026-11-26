"""Generate N=10 candidates for the Countdown/Sudoku feasibility run. Needs a GPU.

    python3 countdown_sudoku.py --make --n 100 --seed 0      # CPU, once
    python3 countdown_sudoku_gen.py --model dream-base --limit 10
    python3 countdown_sudoku_gen.py --model llada-base --limit 10

Same protocol as the GSM8K / GSM-Symbolic runs: N=10, 64 denoising steps,
256 new tokens, one model per process in its own virtualenv. Writes
work/countdown_sudoku/candidates/<model>/<task>.jsonl, resumable by uid, and
prints the pass@10 oracle at the end. Candidates are graded with the exact
checkers in countdown_sudoku.py; the gold solution never enters the prompt.
"""
from __future__ import annotations

import argparse
import time

import countdown_sudoku as cs
from common import WORK, append_jsonl, cfg, read_jsonl
from generate_candidates import BACKENDS

HERE = WORK / "countdown_sudoku"
TASKS = ("countdown", "sudoku")


def prompt_for(task: str, row: dict) -> str:
    if task == "countdown":
        return cs.countdown_prompt(row["numbers"], row["target"])
    return cs.sudoku_prompt(row["puzzle"])


def grade(task: str, text: str, row: dict) -> tuple[str | None, bool, bool, str]:
    """Return (normalised answer, parsed-as-the-right-shape, exact-correct, reason)."""
    if task == "countdown":
        answer = cs.normalise_countdown(text)
        parsed = False
        if answer:
            try:
                cs.parse_expression(answer)
                parsed = True
            except (ValueError, ZeroDivisionError):
                pass
        ok, why = cs.check_countdown(answer, row)
    else:
        answer = cs.normalise_sudoku(text)
        parsed = answer is not None
        ok, why = cs.check_sudoku(answer, row)
    return answer, parsed, ok, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key of config.json 'models', or 'mock' for a CPU wiring test")
    ap.add_argument("--task", choices=[*TASKS, "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles per task (10 for the 20-puzzle gate)")
    a = ap.parse_args()

    c = cfg()
    spec = {"backend": "mock"} if a.model == "mock" else c["models"][a.model]
    c = {**c, "gen_tokens": spec.get("gen_tokens", c["gen_tokens"])}
    print(f"{a.model}: N={c['n_candidates']} steps={spec.get('steps', '-')} tokens={c['gen_tokens']}")
    backend = BACKENDS[spec["backend"]](a.model, spec, c)
    tasks = TASKS if a.task == "both" else (a.task,)
    for task in tasks:
        path = HERE / f"{task}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}; run: python3 countdown_sudoku.py --make --n 100 --seed 0")
        rows = read_jsonl(path)
        rows = rows[: a.limit] if a.limit else rows
        out = HERE / "candidates" / a.model / f"{task}.jsonl"
        done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
        todo = [r for r in rows if r["uid"] not in done]
        print(f"{task}: {len(done)} done, {len(todo)} to go -> {out}", flush=True)
        t0 = time.time()
        for k, row in enumerate(todo, 1):
            tq = time.time()
            texts = backend.sample(row["uid"], prompt_for(task, row), c["n_candidates"])
            cands = []
            for text in texts:
                answer, parsed, ok, why = grade(task, text, row)
                cands.append({"text": text, "answer": answer, "parsed": parsed, "ok": ok, "why": why})
            append_jsonl(out, {"uid": row["uid"], "task": task, "model": a.model,
                               "gen_seconds": round(time.time() - tq, 3), "candidates": cands})
            if k % 5 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el / k:.1f}s/question", flush=True)

    for task in tasks:
        out = HERE / "candidates" / a.model / f"{task}.jsonl"
        rows = read_jsonl(out) if out.exists() else []
        if not rows:
            continue
        oracle = sum(any(x["ok"] for x in r["candidates"]) for r in rows)
        parsed = sum(x["parsed"] for r in rows for x in r["candidates"])
        total = sum(len(r["candidates"]) for r in rows)
        print(f"{task}: oracle pass@10 = {oracle}/{len(rows)}; "
              f"right-shaped = {parsed}/{total} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
