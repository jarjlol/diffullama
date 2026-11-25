"""GSM-Symbolic candidate generation (fresh, non-memorised GSM8K-style questions).

    python3 gsm_symbolic_gen.py --model dream-base
    python3 gsm_symbolic_gen.py --model llada-base

Same protocol as the GSM8K run: N=10, 64 denoising steps, 4-shot prompt, 256 new
tokens, temperature per model config. Writes work/gsm_symbolic/candidates/<model>/
test.jsonl, resumable by uid. The question file is work/gsm_symbolic/questions.jsonl
(125 P1 + 125 P2, seed 0).
"""
from __future__ import annotations

import argparse
import json
import time

from common import WORK, build_prompt, cfg, read_jsonl, task
from generate_candidates import BACKENDS

HERE = WORK / "gsm_symbolic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()

    c = cfg()
    spec = c["models"][a.model]
    c = {**c, "gen_tokens": spec.get("gen_tokens", c["gen_tokens"])}
    shots = json.loads((WORK / "data" / "shots.json").read_text(encoding="utf-8"))
    shots = shots[: spec.get("n_shots", c["n_shots"])]
    rows = read_jsonl(HERE / "questions.jsonl")
    rows = rows[: a.limit] if a.limit else rows
    rows = rows[::-1] if a.reverse else rows
    out = HERE / "candidates" / a.model / f"test{a.out_tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{a.model}: {len(done)} done, {len(todo)} to go, N=10, steps={spec['steps']} -> {out}")

    backend = BACKENDS[spec["backend"]](a.model, spec, c)
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        tq = time.time()
        texts = backend.sample(r["uid"], build_prompt(r["question"], shots), c["n_candidates"])
        rec = {"uid": r["uid"], "model": a.model, "question": r["question"],
               "gen_seconds": round(time.time() - tq, 3),
               "candidates": [{"text": t, "answer": task.extract_answer(t)} for t in texts]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k % 5 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el/k:.1f}s/question  eta {el/k*(len(todo)-k)/60:.0f} min",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
