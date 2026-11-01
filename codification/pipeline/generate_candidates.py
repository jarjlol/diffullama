"""Step 2 -- sample N candidate answers per question from ONE diffusion LM. Needs a GPU.

    python3 generate_candidates.py --model diffullama --split test
    python3 generate_candidates.py --model mock --split dev        # CPU wiring test

Writes work/candidates/<model>/<split>.jsonl, one line per question, appended as it
goes and resumed by uid -- a killed run loses at most the question in flight.

Run ONE model per process, in that model's own virtualenv (config.json 'venv'):
  * the anchor code's attention_patch replaces LlamaModel/GPT2Model.forward for the
    WHOLE process, so no AR model may be loaded alongside DiffuGPT/DiffuLLaMA;
  * the three model families pin different transformers versions.

Backends and the traps each one guards against:
  diffugpt / diffullama  the paper's own sampler (model.generate_samples). It returns
                         the canvas with its FIRST POSITION DROPPED (the shift
                         operation, project fact F-3), so the answer starts at
                         len(prefix) - 1. Asserted at run time: the returned prompt
                         region must equal the prompt.
  dream                  alg MUST be 'entropy': the default 'origin' is random-order
                         decoding (fact F-10). temperature > 0 for N distinct samples.
  llada                  block_length = gen_tokens (smaller = semi-autoregressive,
                         fact F-11); temperature > 0 (its default 0 makes all N
                         identical). Uses LLaDA's generate.py at a pinned commit,
                         fetched at run time: the LLaDA repo has no licence, so it is
                         not vendored.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
import urllib.request

from common import HERE, ROOT, WORK, append_jsonl, build_prompt, cfg, read_jsonl, task

# ------------------------------------------------------------------ backends

class Mock:
    """CPU stand-in: deterministic per (uid, seed) candidates, some right, some wrong."""
    def __init__(self, name, spec, c):
        self.n, self.name = c["n_candidates"], name

    def sample(self, uid, prompt, n, gold_hint=None):
        rng = random.Random(f"{self.name}|{uid}")
        out = []
        for j in range(n):
            g = int(gold_hint) if gold_hint and gold_hint.lstrip("-").isdigit() else 7
            ans = g if rng.random() < 0.35 else g + rng.choice([-2, -1, 1, 3, 10])
            filler = " ".join(["step"] * rng.randint(3, 40))
            out.append(f" Let us think. {filler} so the result is {ans}.\n#### {ans}\n\nQuestion: next one #### 999")
        return out


class Anchor:
    """DiffuGPT / DiffuLLaMA through the paper's own sampler."""
    def __init__(self, name, spec, c):
        sys.path.insert(0, str(ROOT))
        if spec["backend"] == "diffullama":
            from attention_patch import replace_attention_mask
            replace_attention_mask()                 # patches the whole process -- see module docstring
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from model import DiscreteDiffusionModel, generate_samples
        self.torch, self.gen = torch, generate_samples
        self.tok = AutoTokenizer.from_pretrained(spec["hf"])
        conf = AutoConfig.from_pretrained(spec["hf"])
        if spec["backend"] == "diffullama":
            from transformers import LlamaForCausalLM
            base = LlamaForCausalLM.from_pretrained(spec["hf"], device_map="auto", torch_dtype=torch.bfloat16,
                                                    _attn_implementation="eager")
            self.model = DiscreteDiffusionModel(model=base, config=conf, tokenizer=self.tok, device="cuda").to("cuda")
            self.max_len = 2048
        else:
            self.model = DiscreteDiffusionModel.from_pretrained(spec["hf"], model=spec["base"].split("/")[-1],
                                                                config=conf, tokenizer=self.tok, device="cuda").to("cuda")
            self.max_len = conf.task_specific_params["text-generation"]["max_length"]
        self.args = types.SimpleNamespace(diffusion_steps=spec["steps"], logits_temp=spec["logits_temp"],
                                          topp_temp=spec["top_p"], shift=True, verbose=False)
        self.gen_tokens = c["gen_tokens"]

    def sample(self, uid, prompt, n, gold_hint=None):
        torch = self.torch
        prefix = [self.tok.bos_token_id] + self.tok.encode(prompt)
        L = len(prefix) + self.gen_tokens
        if L > self.max_len:
            raise ValueError(f"{uid}: prompt+generation {L} > model max {self.max_len}")
        x0 = torch.tensor([prefix + [0] * self.gen_tokens] * n)
        src = torch.tensor([[1] * len(prefix) + [0] * self.gen_tokens] * n)
        with torch.no_grad():
            out = self.gen(self.model, self.args, self.tok, {"input_ids": x0, "src_mask": src})
        # F-3: shift drops canvas position 0, so returned[k] is canvas[k + 1].
        assert out.shape[1] == L - 1, f"expected {L - 1} returned positions, got {out.shape[1]}"
        assert torch.equal(out[:, :len(prefix) - 1].cpu(), x0[:, 1:len(prefix)]), "prompt region not preserved: shift offset wrong"
        return [self.tok.decode(row[len(prefix) - 1:].tolist(), skip_special_tokens=True) for row in out]


class Dream:
    def __init__(self, name, spec, c):
        import torch
        from transformers import AutoModel, AutoTokenizer
        assert spec.get("alg") == "entropy", "Dream's default alg='origin' is random-order decoding (F-10); set 'entropy'"
        assert spec["temperature"] > 0, "temperature 0 makes all N candidates identical"
        self.torch, self.spec, self.gen_tokens = torch, spec, c["gen_tokens"]
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    def sample(self, uid, prompt, n, gold_hint=None):
        enc = self.tok([prompt] * n, return_tensors="pt")
        ids, att = enc.input_ids.to("cuda"), enc.attention_mask.to("cuda")
        assert ids.shape[1] + self.gen_tokens <= 2048, f"{uid}: exceeds Dream's 2048 context"
        with self.torch.no_grad():
            o = self.model.diffusion_generate(ids, attention_mask=att, max_new_tokens=self.gen_tokens,
                                              output_history=False, return_dict_in_generate=True,
                                              steps=self.spec["steps"], temperature=self.spec["temperature"],
                                              top_p=self.spec["top_p"], alg=self.spec["alg"], alg_temp=0.0)
        return [self.tok.decode(g[ids.shape[1]:].tolist(), skip_special_tokens=True) for g in o.sequences]


class LLaDA:
    def __init__(self, name, spec, c):
        import importlib.util
        import torch
        from transformers import AutoModel, AutoTokenizer
        assert spec["temperature"] > 0, "LLaDA's default temperature 0 makes all N candidates identical"
        path = HERE / "third_party" / f"llada_generate_{c['llada_generate_commit'][:7]}.py"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/ML-GSAI/LLaDA/{c['llada_generate_commit']}/generate.py"
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        mod = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("llada_generate", path)); s.loader.exec_module(mod)
        self.generate, self.torch, self.spec, self.gen_tokens = mod.generate, torch, spec, c["gen_tokens"]
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    def sample(self, uid, prompt, n, gold_hint=None):
        ids = self.tok([prompt] * n, return_tensors="pt").input_ids.to("cuda")
        with self.torch.no_grad():
            out = self.generate(self.model, ids, steps=self.spec["steps"], gen_length=self.gen_tokens,
                                block_length=self.gen_tokens,             # F-11: no semi-AR blocks
                                temperature=self.spec["temperature"], cfg_scale=0.0, remasking=self.spec["remasking"])
        return [self.tok.decode(g[ids.shape[1]:].tolist(), skip_special_tokens=True) for g in out]


BACKENDS = {"mock": Mock, "diffugpt": Anchor, "diffullama": Anchor, "dream": Dream, "llada": LLaDA}

# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key of config.json 'models', or 'mock'")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke tests)")
    a = ap.parse_args()
    c = cfg()
    spec = {"backend": "mock"} if a.model == "mock" else c["models"][a.model]
    shots = json.loads((WORK / "data" / "shots.json").read_text(encoding="utf-8"))
    qs = read_jsonl(WORK / "data" / f"{a.split}.jsonl")
    qs = qs[:a.limit] if a.limit else qs
    out = WORK / "candidates" / a.model / f"{a.split}.jsonl"
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [q for q in qs if q["uid"] not in done]
    print(f"{a.model}/{a.split}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return 0
    backend = BACKENDS[spec["backend"]](a.model, spec, c)
    t0 = time.time()
    for k, q in enumerate(todo, 1):
        prompt = build_prompt(q["question"], shots)
        hint = q["answer"] if spec["backend"] == "mock" else None     # only the mock may see gold
        texts = backend.sample(q["uid"], prompt, c["n_candidates"], gold_hint=hint)
        assert len(texts) == c["n_candidates"]
        append_jsonl(out, {"uid": q["uid"], "model": a.model, "question": q["question"], "prompt": prompt,
                           "candidates": [{"text": t, "answer": task.extract_answer(t)} for t in texts]})
        if k % 10 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el / k:.1f}s/question  eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
