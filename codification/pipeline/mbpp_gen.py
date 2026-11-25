"""MBPP candidate generation -- queued after HumanEval.

    python3 mbpp_gen.py --model dream-base [--reverse --out-tag .rev]

N=10, 64 denoising steps, T=0.7, 512 new tokens, zero-shot. The prompt is the
MBPP task text; the tests are never shown. Writes
work/mbpp/candidates/<model>/test.jsonl, resumable by task_id. The split is all
427 sanitized problems (no dev); if time cuts the run short, the exact n is
reported and no comparison is made to a full-set number.
"""
from __future__ import annotations

import argparse
import json
import time

from common import WORK, cfg, read_jsonl
from generate_candidates import BACKENDS

HERE = WORK / "mbpp"
TEMPLATE = """Write a Python function to solve the following task. Return only the code,
no tests and no explanation.

{task}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()

    c = cfg()
    spec = c["models"][a.model]
    if spec["backend"] != "dream":
        raise SystemExit("MBPP generation is registered for Dream only")
    c = {**c, "gen_tokens": 512}
    rows = json.loads((HERE / "mbpp.json").read_text(encoding="utf-8"))
    rows = rows[: a.limit] if a.limit else rows
    rows = rows[::-1] if a.reverse else rows
    out = HERE / "candidates" / a.model / f"test{a.out_tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if str(r["task_id"]) not in done]
    print(f"{a.model}: {len(done)} done, {len(todo)} to go, N=10, steps={spec['steps']} -> {out}")

    backend = BACKENDS[spec["backend"]](a.model, spec, c)
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        uid = str(r["task_id"])
        tq = time.time()
        texts = backend.sample(uid, TEMPLATE.format(task=r["prompt"]), c["n_candidates"])
        rec = {"uid": uid, "model": a.model, "question": r["prompt"],
               "gen_seconds": round(time.time() - tq, 3),
               "candidates": [{"text": t} for t in texts]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k % 5 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el/k:.1f}s/problem  eta {el/k*(len(todo)-k)/60:.0f} min",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
