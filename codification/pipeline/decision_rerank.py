"""Step 5 -- decision-model scores for cached candidates. Needs a GPU.

    python3 decision_rerank.py --model dream-base --split test --mode listwise
    python3 decision_rerank.py --model dream-base --split dev  --mode pointwise --limit 20
    python3 decision_rerank.py --model mock --split dev --mode listwise --limit 10   # CPU wiring test

Writes work/features/<model>/<split>.nimble-<mode>.jsonl, one line per question,
`features` aligned 1:1 with the candidates in work/candidates/<model>/<split>.jsonl:

    decision.listwise.p      P(this candidate is the correct one), softmax over A..J
    decision.pointwise.p_yes P(this candidate's final answer is correct), yes/no pass

The feature keys are backend-neutral: the same file layout works for pplx-decider
(the registered primary), Bespoke-Nimble, and the mock.

The candidate text is cut exactly as the AR cut scores were cut
(`score_candidates.cut_after_answer` on `task.trim`), so every selector sees the
same input text. No gold, no answer key, nothing but the question and the
candidates enters the prompt.

Resumable by uid. Runs one model per process. The decision model is loaded once
and reused; on a 96 GB card that is one 9B model, so do not run two copies on the
same GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from common import WORK, append_jsonl, read_jsonl, task
from score_candidates import cut_after_answer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CONFIG = Path("/home/antpc/models/nimble-model.json")


def candidate_text(raw: str) -> str:
    return cut_after_answer(task.trim(raw))


def listwise_context(question: str, texts: list[str]) -> str:
    parts = [
        "Math word problem:",
        question.strip(),
        "",
        f"{len(texts)} candidate solutions, each ending with its final answer:",
        "",
    ]
    for i, t in enumerate(texts):
        parts += [f"Candidate {LETTERS[i]}:", t.strip(), ""]
    parts.append("Pick the one candidate whose final answer is correct.")
    return "\n".join(parts)


def listwise_schema(n: int) -> dict:
    return {
        "best": {
            "type": "enum",
            "choices": list(LETTERS[:n]),
            "description": "The candidate solution that reaches the correct final answer to the math problem.",
            "choice_descriptions": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(n)},
        }
    }


def pointwise_context(question: str, text: str) -> str:
    return "\n".join([
        "Math word problem:",
        question.strip(),
        "",
        "Candidate solution:",
        text.strip(),
        "",
        "Does this candidate's final answer agree with the correct answer to the problem?",
    ])


POINTWISE_SCHEMA = {
    "correct": {
        "type": "boolean",
        "description": "Whether the candidate solution reaches the correct final answer to the math problem.",
    }
}


class Nimble:
    """Bespoke-Nimble-9B through its CUDA scorer; one forward pass per field."""

    def __init__(self, context_path: Path = CONFIG):
        from nimble.scoring.cuda_scorer import CudaCandidateScorer

        c = json.loads(context_path.read_text(encoding="utf-8"))
        self.scorer = CudaCandidateScorer(**c)

    def listwise(self, question: str, texts: list[str]) -> tuple[list[float], int]:
        r = self.scorer.score(listwise_context(question, texts), listwise_schema(len(texts)))
        f = r["fields"]["best"]
        return [float(f["scores"][LETTERS[i]]) for i in range(len(texts))], f["prompt_token_count"]

    def pointwise(self, question: str, text: str) -> tuple[float, int]:
        r = self.scorer.score(pointwise_context(question, text), POINTWISE_SCHEMA)
        f = r["fields"]["correct"]
        return float(f["scores"]["true"]), f["prompt_token_count"]


class Pplx:
    """pplx-decider-flash-v1.3 through its native local package.

    4B, Apache-2.0, 30,720-token training length (262K window). `choice` returns
    a probability per option; `noul` returns P(true). One sequence per question,
    no chain of thought.
    """

    def __init__(self, path: Path = Path("/home/antpc/models/pplx-decider-flash-v1.3")):
        import sys

        sys.path.insert(0, str(path / "source" / "src"))
        from pplx_decider import Decider

        self.decider = Decider.from_pretrained(path, device="cuda")

    def listwise(self, question: str, texts: list[str]) -> tuple[list[float], int]:
        criteria = {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(texts))}
        state = {"problem": question,
                 "candidates": {LETTERS[i]: t for i, t in enumerate(texts)}}
        r = self.decider.decide({
            "state": state,
            "questions": {"best": {
                "type": "choice",
                "instructions": "Which candidate solution reaches the correct final answer to the math problem?",
                "criteria": criteria,
            }},
        })
        ans = r["answers"]["best"]
        return [float(ans["probabilities"][LETTERS[i]]) for i in range(len(texts))], r["usage"]["input_tokens"]

    def pointwise(self, question: str, text: str) -> tuple[float, int]:
        r = self.decider.decide({
            "state": {"problem": question, "candidate": text},
            "questions": {"correct": {
                "type": "noul",
                "instructions": "Does this candidate's final answer agree with the correct answer to the math problem?",
            }},
        })
        return float(r["answers"]["correct"]["noul"]), r["usage"]["input_tokens"]


class Mock:
    """Deterministic stand-in with no model: checks the plumbing, not the science."""

    def _p(self, *parts: str) -> float:
        h = hashlib.sha256("|".join(parts).encode()).digest()[:8]
        return 0.05 + 0.9 * (int.from_bytes(h, "big") / 0xFFFFFFFFFFFFFFFF)

    def listwise(self, question: str, texts: list[str]) -> tuple[list[float], int]:
        p = [self._p(question, t) for t in texts]
        s = sum(p)
        return [x / s for x in p], sum(len(t.split()) for t in texts)

    def pointwise(self, question: str, text: str) -> tuple[float, int]:
        return self._p(question, text), len(text.split())


def out_path(model: str, split: str, mode: str) -> Path:
    return WORK / "features" / model / f"{split}.nimble-{mode}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key of config.json 'models', or 'mock'")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--mode", choices=["listwise", "pointwise"], required=True)
    ap.add_argument("--backend", choices=["pplx", "nimble", "mock"], default="pplx",
                    help="decision model; pplx is the registered primary")
    ap.add_argument("--limit", type=int, default=0, help="first N questions (smoke tests)")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="present candidates in a seeded permuted order, map scores back")
    ap.add_argument("--max-prompts", type=int, default=0)
    a = ap.parse_args()

    cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / a.model / f"{a.split}.jsonl")}
    uids = sorted(cands)
    uids = uids[: a.limit] if a.limit else uids
    uids = uids[::-1] if a.reverse else uids

    out = out_path(a.model, a.split, a.mode)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [u for u in uids if u not in done]
    print(f"{a.model}/{a.split}/{a.mode}: {len(done)} done, {len(todo)} to go -> {out}")

    scorer = {"pplx": Pplx, "nimble": Nimble, "mock": Mock}[a.backend]()
    t0, seen = time.time(), 0
    for k, uid in enumerate(todo, 1):
        r = cands[uid]
        texts = [candidate_text(c["text"]) for c in r["candidates"]]
        perm = list(range(len(texts)))
        if a.shuffle_seed is not None:
            random.Random(f"{a.shuffle_seed}|{uid}").shuffle(perm)
        shown = [texts[i] for i in perm]

        if a.mode == "listwise":
            try:
                probs, ntok = scorer.listwise(r["question"], shown)
            except Exception as e:
                if "Length" not in type(e).__name__:
                    raise
                # Pre-registered fallback: score this item pointwise, never truncate.
                probs, ntok = [scorer.pointwise(r["question"], t)[0] for t in shown], 0
                row_extra = {"fallback": "pointwise"}
            else:
                row_extra = {}
            back = [0.0] * len(texts)
            for pos, orig in enumerate(perm):
                back[orig] = probs[pos]
            row = {"uid": uid, "features": [{"decision.listwise.p": p} for p in back],
                   "prompt_tokens": ntok, "shuffle_seed": a.shuffle_seed, **row_extra}
        else:
            probs, ntok = [], 0
            for t in shown:
                p, n = scorer.pointwise(r["question"], t)
                probs.append(p)
                ntok = max(ntok, n)
            back = [0.0] * len(texts)
            for pos, orig in enumerate(perm):
                back[orig] = probs[pos]
            row = {"uid": uid, "features": [{"decision.pointwise.p_yes": p} for p in back],
                   "prompt_tokens": ntok, "shuffle_seed": a.shuffle_seed}

        append_jsonl(out, row)
        seen += 1
        if k % 10 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el / k:.2f}s/question  eta {el / k * (len(todo) - k) / 60:.1f} min",
                  flush=True)
        if a.max_prompts and seen >= a.max_prompts:
            print(f"stopped after {seen} prompts (--max-prompts)")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
