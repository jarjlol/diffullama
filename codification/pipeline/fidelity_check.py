"""Iteration 2, prompt A2 step 3 -- fidelity check. Needs a GPU.

Iteration 1's first-sample accuracy is far below the published numbers (Dream 41.0 vs 77.2,
LLaDA 48.7 vs 70.3; ITERATION_1.md "Published numbers"). This runs ONE greedy answer per
question at each model's published settings, so we can tell a sampler effect from a bug.

    python3 fidelity_check.py --config dream-sampler --limit 100
    python3 fidelity_check.py --config dream-published --limit 100
    python3 fidelity_check.py --config llada-published --limit 100
    python3 fidelity_check.py --report            # table vs iteration 1 on the same questions

Configs (settings read 2026-11-20 from the authors' own eval scripts):
  dream-sampler    our 4-shot prompt; Dream's published sampler: 256 new tokens, 256 steps,
                   temperature 0, alg entropy. Changes only the sampler vs iteration 1.
  dream-published  as above, plus Dream's published prompt: lm-eval gsm8k_cot, 8 fixed
                   chain-of-thought exemplars ("Q: ... A: ... The answer is N."), BOS token
                   prepended. Source: DreamLM/Dream eval/eval_dream_gen.sh (gsm8k_cot, 8-shot,
                   max_new_tokens = diffusion_steps = 256, temperature 0, add_bos_token=true)
                   and eval/eval.py (alg default "entropy"); exemplars from
                   EleutherAI/lm-evaluation-harness lm_eval/tasks/gsm8k/gsm8k-cot.yaml.
  llada-published  our 4-shot prompt (same "Question:/Answer:" format as lm-eval gsm8k);
                   LLaDA's published sampler: gen_length = steps = block_length = 256,
                   temperature 0, remasking low_confidence, no CFG. Source: ML-GSAI/LLaDA
                   EVAL.md (GSM8K 70.0 at the 256 setting) and eval_llada.py.

Grading is task.extract_answer + task.is_correct, the same code as iteration 1. Writes
work/fidelity/<config>.jsonl, appended and resumed by uid.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time

from common import HERE, WORK, append_jsonl, build_prompt, cfg, read_jsonl, task

COT_SHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]

CONFIGS = {
    "dream-sampler":   {"model": "dream-base", "prompt": "ours",  "steps": 256, "gen_tokens": 256, "bos": False},
    "dream-published": {"model": "dream-base", "prompt": "cot8",  "steps": 256, "gen_tokens": 256, "bos": True},
    "llada-published": {"model": "llada-base", "prompt": "ours",  "steps": 256, "gen_tokens": 256, "bos": False},
}
PUBLISHED = {"dream-base": "77.2 (8-shot gsm8k_cot; Dream paper Table 1)",
             "llada-base": "70.3 (4-shot; LLaDA Table 1) / 70.0 (lm-eval, 256 setting; LLaDA EVAL.md)"}


def cot_prompt(question: str) -> str:
    # lm-eval: doc_to_text "Q: {q}\nA:", target_delimiter " ", fewshot_delimiter "\n\n"
    shots = [f"Q: {q}\nA: {a}" for q, a in COT_SHOTS]
    return "\n\n".join(shots + [f"Q: {question}\nA:"])


class DreamGreedy:
    def __init__(self, spec, conf):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.conf = torch, conf
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    def answer(self, prompt):
        if self.conf["bos"] and self.tok.bos_token:
            prompt = self.tok.bos_token + prompt
        enc = self.tok([prompt], return_tensors="pt")
        ids, att = enc.input_ids.to("cuda"), enc.attention_mask.to("cuda")
        with self.torch.no_grad():
            o = self.model.diffusion_generate(ids, attention_mask=att, max_new_tokens=self.conf["gen_tokens"],
                                              output_history=False, return_dict_in_generate=True,
                                              steps=self.conf["steps"], temperature=0.0, top_p=0.95,
                                              alg="entropy", alg_temp=0.0)
        return self.tok.decode(o.sequences[0][ids.shape[1]:].tolist(), skip_special_tokens=True)


class LLaDAGreedy:
    def __init__(self, spec, conf, c):
        import torch
        from transformers import AutoModel, AutoTokenizer
        path = HERE / "third_party" / f"llada_generate_{c['llada_generate_commit'][:7]}.py"
        assert path.exists(), f"{path} missing: run generate_candidates.py for llada-base once, it fetches it"
        mod = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("llada_generate", path)); s.loader.exec_module(mod)
        self.generate, self.torch, self.conf = mod.generate, torch, conf
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    def answer(self, prompt):
        ids = self.tok([prompt], return_tensors="pt").input_ids.to("cuda")
        n = self.conf["gen_tokens"]
        with self.torch.no_grad():
            out = self.generate(self.model, ids, steps=self.conf["steps"], gen_length=n, block_length=n,
                                temperature=0.0, cfg_scale=0.0, remasking="low_confidence")
        return self.tok.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True)


def run(name: str, limit: int) -> None:
    c, conf = cfg(), CONFIGS[name]
    spec = c["models"][conf["model"]]
    shots = json.loads((WORK / "data" / "shots.json").read_text(encoding="utf-8"))[:spec.get("n_shots", c["n_shots"])]
    qs = read_jsonl(WORK / "data" / "test.jsonl")[:limit]
    out = WORK / "fidelity" / f"{name}.jsonl"
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [q for q in qs if q["uid"] not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    gen = DreamGreedy(spec, conf) if conf["model"] == "dream-base" else LLaDAGreedy(spec, conf, c)
    t0 = time.time()
    for k, q in enumerate(todo, 1):
        prompt = cot_prompt(q["question"]) if conf["prompt"] == "cot8" else build_prompt(q["question"], shots)
        tq = time.time()
        text = gen.answer(prompt)
        pred = task.extract_answer(text)
        append_jsonl(out, {"uid": q["uid"], "config": name, "gen_seconds": round(time.time() - tq, 3),
                           "text": text, "answer": pred, "correct": task.is_correct(pred, q["answer"])})
        if k % 10 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el / k:.1f}s/question", flush=True)


def report() -> None:
    """Greedy accuracy per config next to iteration 1's first sample and majority on the same uids."""
    for name, conf in CONFIGS.items():
        p = WORK / "fidelity" / f"{name}.jsonl"
        if not p.exists():
            print(f"{name:16s} not run"); continue
        rows = read_jsonl(p)
        uids = {r["uid"] for r in rows}
        gold = {q["uid"]: q["answer"] for q in read_jsonl(WORK / "data" / "test.jsonl")}
        cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / conf["model"] / "test.jsonl") if r["uid"] in uids}
        first = sum(task.is_correct(cands[u]["candidates"][0]["answer"], gold[u]) for u in cands)
        acc = sum(r["correct"] for r in rows) / len(rows)
        none = sum(r["answer"] is None for r in rows)
        secs = sorted(r["gen_seconds"] for r in rows)[len(rows) // 2]
        print(f"{name:16s} n={len(rows):3d}  greedy {acc:.3f}  | iter-1 first sample on same {len(cands)} Qs "
              f"{first / max(1, len(cands)):.3f}  | no answer {none}  | median {secs:.1f}s/q  | published {PUBLISHED[conf['model']]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return 0
    if not a.config:
        ap.error("--config or --report")
    run(a.config, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
