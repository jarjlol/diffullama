"""HumanEval candidate generation -- the only arm that needs new samples.

    python3 humaneval_gen.py --model dream-base --split test
    python3 humaneval_gen.py --model llada-base --split test --limit 2   # smoke

Run ONE model per process, in that model's own venv, exactly like
generate_candidates.py (attention_patch and the three model families' pinned
transformers versions forbid sharing a process):

    CUDA_VISIBLE_DEVICES=1 ~/venvs/dream/bin/python humaneval_gen.py --model dream-base --split test
    CUDA_VISIBLE_DEVICES=0 ~/venvs/llada/bin/python humaneval_gen.py --model llada-base --split test

N=10, 64 denoising steps, T=0.7, max_new_tokens=512, zero-shot. Writes
work/humaneval/candidates/<model>/<split>.jsonl, resumable by task_id. The
dev/test split is a seeded shuffle of the 164 problems, frozen in splits.json
before the first sample, so every model and selector sees the same split.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from common import WORK, cfg, read_jsonl
from generate_candidates import BACKENDS

HERE = WORK / "humaneval"
TEMPLATE = """Complete the following Python function. Return only the complete function
(signature, docstring and body). Do not write tests and do not explain.

{prompt}"""


def problems() -> list[dict]:
    return read_jsonl(HERE / "HumanEval.jsonl")


def splits() -> dict[str, str]:
    """Frozen dev(20)/test(144) assignment, seed 0, written once."""
    p = HERE / "splits.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    rows = problems()
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    out = {rows[i]["task_id"]: ("dev" if k < 20 else "test") for k, i in enumerate(idx)}
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key of config.json 'models'")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out-tag", default="", help="write <split><tag>.jsonl, for parallel shards")
    a = ap.parse_args()

    c = cfg()
    spec = c["models"][a.model]
    if spec["backend"] not in ("dream", "llada"):
        raise SystemExit(f"{a.model}: HumanEval generation is registered for Dream and LLaDA only")
    gen_tokens = 512
    c = {**c, "gen_tokens": gen_tokens}

    which = splits()
    rows = [r for r in problems() if which[r["task_id"]] == a.split]
    rows = rows[: a.limit] if a.limit else rows
    rows = rows[::-1] if a.reverse else rows
    out = HERE / "candidates" / a.model / f"{a.split}{a.out_tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["task_id"] not in done]
    print(f"{a.model}/{a.split}: {len(done)} done, {len(todo)} to go, N=10, steps={spec['steps']}, "
          f"gen_tokens={gen_tokens} -> {out}")

    backend = BACKENDS[spec["backend"]](a.model, spec, c)
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        tq = time.time()
        texts = backend.sample(r["task_id"], TEMPLATE.format(prompt=r["prompt"]), c["n_candidates"])
        assert len(texts) == c["n_candidates"]
        append = {"uid": r["task_id"], "model": a.model, "question": r["prompt"],
                  "gen_seconds": round(time.time() - tq, 3),
                  "candidates": [{"text": t} for t in texts]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(append, ensure_ascii=False) + "\n")
        if k % 5 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el / k:.1f}s/problem  eta {el / k * (len(todo) - k) / 60:.0f} min",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
