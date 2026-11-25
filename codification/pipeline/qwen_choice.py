"""Fallback selector: a cached AR model asked for a typed letter / yes-no.

This is NOT the registered System-1 model. It exists so the pipeline produces a
complete table when the decision-model weights cannot be staged (the campus link
is capped and the 9.6 GB download cannot finish). It is reported as
`qwen.choice` / `qwen.yes`, never as `decision.listwise`.

    CUDA_VISIBLE_DEVICES=1 ~/venvs/scorer/bin/python qwen_choice.py --model dream-base --mode listwise
    ... --limit 5 --dump      # print choices for a smoke test

One forward pass per question (listwise) or per candidate (pointwise); the
choice is read from the logits of the single-token letters A..J (or yes/no), no
chain of thought and no sampling.
"""
from __future__ import annotations

import argparse
import json
import time

from common import WORK, read_jsonl
from decision_rerank import candidate_text, listwise_context, pointwise_context
from score_candidates import HFScorer

LETTERS = "ABCDEFGHIJ"


class QwenChoice:
    def __init__(self, hf_id: str = "Qwen/Qwen2.5-7B"):
        s = HFScorer(hf_id)
        self.tok, self.model, self.torch = s.tok, s.model, s.torch
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(" " + L, add_special_tokens=False)
            assert len(ids) == 1, f"letter {L} is {len(ids)} tokens"
            self.letter_ids.append(ids[0])
        self.yes = self.tok.encode(" yes", add_special_tokens=False)
        self.no = self.tok.encode(" no", add_special_tokens=False)
        assert len(self.yes) == 1 and len(self.no) == 1

    def _probs(self, prompt: str, ids: list[int]) -> list[float]:
        torch = self.torch
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        sub = logits[torch.tensor(ids, device=logits.device)]
        return torch.softmax(sub, dim=-1).tolist()

    def listwise(self, question: str, texts: list[str]) -> tuple[list[float], int]:
        p = listwise_context(question, texts) + "\nThe correct candidate is:"
        return self._probs(p, self.letter_ids), len(self.tok(p).input_ids)

    def pointwise(self, question: str, text: str) -> tuple[float, int]:
        p = pointwise_context(question, text) + "\nAnswer yes or no:"
        probs = self._probs(p, self.yes + self.no)
        return probs[0], len(self.tok(p).input_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--mode", choices=["listwise", "pointwise"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()

    rows = read_jsonl(WORK / "candidates" / a.model / f"{a.split}.jsonl")
    rows = rows[: a.limit] if a.limit else rows
    out = WORK / "features" / a.model / f"{a.split}.qwen-{a.mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{a.model}/{a.split}/qwen-{a.mode}: {len(done)} done, {len(todo)} to go -> {out}")

    q = QwenChoice()
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        texts = [candidate_text(c["text"]) for c in r["candidates"]]
        if a.mode == "listwise":
            probs, ntok = q.listwise(r["question"], texts)
            feats = [{"qwen.choice.p": p} for p in probs]
            if a.dump:
                i = max(range(len(probs)), key=lambda j: probs[j])
                print(r["uid"], LETTERS[i], f"{probs[i]:.3f}", flush=True)
        else:
            feats = []
            for t in texts:
                p, ntok = q.pointwise(r["question"], t)
                feats.append({"qwen.yes.p": p})
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"uid": r["uid"], "features": feats, "prompt_tokens": ntok}) + "\n")
        if k % 20 == 0:
            print(f"  {k}/{len(todo)}  {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
