"""Step 3 -- AR-scorer features for every candidate. Needs a GPU (small models).

    python3 score_candidates.py --model diffullama --split test --scorer gpt2
    python3 score_candidates.py --model mock --split dev --scorer mock     # CPU wiring test

For each candidate, the scorer's negative log-likelihood of the candidate's answer
text given the question:

    ar.<scorer>.nll_sum    sum of -log p(answer token | "Question: q\\nAnswer:", earlier tokens)
    ar.<scorer>.n_tokens   number of answer tokens scored

RESOLVED AMBIGUITY in idea 1.1: it says "NLL conditioned on the prompt" without
saying which prompt. The 4-shot generation prompt alone nearly fills GPT-2's
1024-token window, so the scorer conditions on the question only. Recorded in
../IDEA_SELECTION.md as a specification fix.

MUST run in a process where attention_patch has NOT been applied -- the patch
replaces LlamaModel/GPT2Model.forward process-wide and would silently make the
scorer bidirectional. A causality self-check runs at load and aborts if a future
token can change an earlier token's prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time

from common import WORK, append_jsonl, cfg, read_jsonl, task


class MockScorer:
    """Deterministic pseudo-NLL for wiring tests; not a language model."""
    def nll(self, context, texts):
        out = []
        for t in texts:
            n = max(1, len(t.split()))
            h = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            out.append((n * (2.0 + h), n))
        return out


class HFScorer:
    def __init__(self, hf_id):
        import torch
        import transformers
        for cls in ("LlamaModel", "GPT2Model"):
            mod = transformers.models.llama.modeling_llama if cls == "LlamaModel" else transformers.models.gpt2.modeling_gpt2
            fwd = getattr(mod, cls).forward
            assert "attention_patch" not in getattr(fwd, "__module__", ""), \
                f"{cls}.forward is patched by attention_patch -- run the scorer in a separate process"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.max_len = min(getattr(self.model.config, "max_position_embeddings", 2048), 4096)
        self._check_causal()

    def _check_causal(self):
        torch = self.torch
        ids = self.tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.to("cuda")
        alt = ids.clone(); alt[0, -1] = (alt[0, -1] + 1) % self.model.config.vocab_size
        with torch.no_grad():
            a = self.model(ids).logits[0, :-1].float(); b = self.model(alt).logits[0, :-1].float()
        assert torch.allclose(a, b, atol=1e-2), "scorer is NOT causal: changing the last token changed earlier logits"

    def nll(self, context, texts):
        torch = self.torch
        ctx = self.tok(context, add_special_tokens=True).input_ids
        rows, spans = [], []
        for t in texts:
            ans = self.tok(" " + t, add_special_tokens=False).input_ids
            ids = (ctx + ans)[: self.max_len]
            rows.append(ids); spans.append((len(ctx), len(ids)))
        width = max(len(r) for r in rows)
        x = torch.full((len(rows), width), self.tok.pad_token_id)
        att = torch.zeros_like(x)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r); att[i, :len(r)] = 1
        x, att = x.to("cuda"), att.to("cuda")
        with torch.no_grad():
            logp = torch.log_softmax(self.model(x, attention_mask=att).logits.float(), dim=-1)
        out = []
        for i, (s, e) in enumerate(spans):
            if e <= s:
                out.append((None, 0)); continue
            tgt = x[i, s:e]
            lp = logp[i, s - 1:e - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # token t predicted at t-1
            out.append((float(-lp.sum()), e - s))
        return out


_ANSWER_LINE = re.compile(r"####[^\n]*")


def cut_after_answer(text: str) -> str:
    """Iteration 2: the candidate up to the end of its '#### N' line.

    Most 64-step candidates keep writing after the answer line (repeated words, unrelated text):
    753 of 1000 Dream and 581 of 1000 LLaDA candidates on the first 100 test questions carry more
    than 20 characters after '#### N' (median about 450). task.trim keeps that tail, so the
    iteration-1 AR scores mostly scored junk. Candidates without '####' are left as task.trim gives them.
    """
    body = task.trim(text)
    m = _ANSWER_LINE.search(body)
    return body[:m.end()] if m else body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--scorer", required=True, help="a key of config.json 'scorers', or 'mock'")
    ap.add_argument("--cut", action="store_true",
                    help="score the text only up to the '#### N' line; features are named ar.<scorer>-cut.*, "
                         "so baselines.py reports them as a separate scorer")
    a = ap.parse_args()
    c = cfg()
    name = f"{a.scorer}-cut" if a.cut else a.scorer
    src = WORK / "candidates" / a.model / f"{a.split}.jsonl"
    out = WORK / "features" / a.model / f"{a.split}.{name}.jsonl"
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in read_jsonl(src) if r["uid"] not in done]
    print(f"{a.model}/{a.split} x {name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return 0
    sc = MockScorer() if a.scorer == "mock" else HFScorer(c["scorers"][a.scorer])
    prep = cut_after_answer if a.cut else task.trim
    for r in todo:
        t0 = time.time()
        texts = [prep(cd["text"]) for cd in r["candidates"]]
        res = sc.nll(f"Question: {r['question']}\nAnswer:", texts)
        append_jsonl(out, {"uid": r["uid"], "score_seconds": round(time.time() - t0, 3), "features": [
            {f"ar.{name}.nll_sum": s, f"ar.{name}.n_tokens": n} for s, n in res]})
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
