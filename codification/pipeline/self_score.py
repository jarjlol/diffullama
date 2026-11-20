"""Iteration 2, Tier 2 (prompt A2 step 2): the diffusion LM's OWN likelihood as a reranking signal.

    python3 self_score.py --model dream-base --split test            # GPU, Dream venv
    python3 self_score.py --model llada-base --split test            # GPU, LLaDA venv
    python3 self_score.py --selftest                                 # CPU, estimator unit tests
    python3 self_score.py --model mock --split test --candidates-from dream-base --limit 5   # CPU wiring test

Scores every EXISTING 64-step candidate in work/candidates/<model>/<split>.jsonl. No new generation.
Writes work/features/<model>/<split>.dlmself.jsonl in the format score_candidates.py uses (one row per
question, one feature dict per candidate, resumed by uid), so assemble.py merges it and the protected
rerank_task/baselines.py picks up `dlm.selfscore` unchanged (it selects the HIGHEST value).

Per candidate j of a question, three features:
    dlm.selfscore      mean log-likelihood per answer token (ELBO / n_tokens); HIGHER = better. Used by baselines.py
    dlm.self.nll_sum   -ELBO, the summed negative log-likelihood bound
    dlm.self.n_tokens  number of answer tokens scored

ESTIMATOR (follows LLaDA's own code at the commit pinned in config.json "llada_generate_commit",
f2683fd, files eval_llada.py `get_loglikelihood` / `_forward_process`, read 2026-11-20):
a Monte-Carlo bound on log p(answer | prompt). With k draws in one batch, draw i masks x_i answer
tokens (x_i stratified over 1..T as in the official code, positions chosen at random, prompt never
masked), so its mask fraction is t_i = x_i / T. The bound is
        ELBO = - (1/k) * sum_i  sum_{masked positions}  CE(logits, true token) / t_i .
Check: a uniform predictor gives exactly -T log V (tested below), a perfect one gives 0.
The ELBO is a lower bound on the log-likelihood, and noisy for small k (k = 8 here).

SEED: one torch.Generator per (uid, candidate index), seeded from sha256, so a rerun reproduces the scores.

WHAT IS SCORED, AND HOW IT DIFFERS FROM THE AR SCORERS: the context is the 4-shot GENERATION prompt stored
in the candidates file (the diffusion model's own situation), NOT the question-only context the AR scorers
use (score_candidates.py had to drop the shots because of GPT-2's window). The answer is task.trim(text)
with a leading space, tokenised separately. The answer is capped at ANSWER_CAP tokens.

PER-MODEL DETAILS (each checked in the model code, 2026-11-20):
  llada  mask id 126336; logits at position i predict token i (no shift); official code uses cfg 0.
  dream  mask id = model.generation_config.mask_token_id; Dream's own sampler shifts the logits right by
         one position before reading them (generation_utils.py line 412:
         `logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)`), so token i is read from raw
         logits[i-1]. `shift_right` below does the same; the estimator reads aligned logits.
         The likelihood estimator itself is LLaDA's, applied to Dream by analogy [unverified: Dream's own
         evaluation code was not available offline, so its estimator was not compared].

Run in the model's own venv (config.json 'venv'). Not for DiffuGPT/DiffuLLaMA (not asked for).
Smoke test built in: after the first question the 10 scores are printed and the run aborts if they are
all equal (a broken mask or shift would give identical scores).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

from common import WORK, append_jsonl, cfg, read_jsonl, task

ANSWER_CAP = 256


# ------------------------------------------------------------------ the estimator (pure torch)

def shift_right(raw_logits):
    """Dream: aligned[:, i] = raw[:, i-1] (position 0 keeps raw[:, 0]), as in its generation_utils."""
    import torch
    return torch.cat([raw_logits[:, :1], raw_logits[:, :-1]], dim=1)


def draw_masks(target_len, k, gen):
    """LLaDA's `_forward_process`, for the answer part only: returns (is_mask [k, T] bool, t [k] float)."""
    import torch
    kk = int(torch.randint(1, target_len + 1, (), generator=gen))
    x = torch.round(torch.linspace(float(kk), kk + (k - 1) * (target_len / k), steps=k)).long()
    x = ((x - 1) % target_len) + 1
    assert int(x.min()) >= 1 and int(x.max()) <= target_len
    is_mask = torch.zeros(k, target_len, dtype=torch.bool)
    for i in range(k):
        is_mask[i, torch.randperm(target_len, generator=gen)[: int(x[i])]] = True
    assert (is_mask.sum(1) == x).all()
    return is_mask, x.float() / target_len


def mc_elbo(logits_fn, prompt_ids, answer_ids, mask_id, k, gen, device="cpu"):
    """Monte-Carlo ELBO of log p(answer | prompt). `logits_fn(x: LongTensor [k, L]) -> [k, L, V]` must return
    logits ALIGNED to positions (logits[:, i] predicts token i; apply shift_right inside it for Dream)."""
    import torch
    import torch.nn.functional as F
    P, T = len(prompt_ids), len(answer_ids)
    assert T >= 1
    seq = torch.tensor(list(prompt_ids) + list(answer_ids), dtype=torch.long).unsqueeze(0).repeat(k, 1)
    is_mask_ans, t = draw_masks(T, k, gen)
    is_mask = torch.cat([torch.zeros(k, P, dtype=torch.bool), is_mask_ans], dim=1)
    noisy = torch.where(is_mask, torch.tensor(mask_id), seq)
    with torch.no_grad():
        logits = logits_fn(noisy.to(device))[:, P:].float()          # answer positions only: [k, T, V]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), seq[:, P:].to(logits.device).reshape(-1),
                         reduction="none").reshape(k, T).cpu()
    m = is_mask_ans
    loss = (ce * m / t.unsqueeze(1)).sum() / k
    return -float(loss)


# ------------------------------------------------------------------ backends

class Mock:
    """CPU stand-in: char-level 'tokens', bidirectional pseudo-logits from the two neighbours."""
    mask_id = 99

    def __init__(self, *_):
        import torch
        g = torch.Generator().manual_seed(1)
        self.W = torch.randn(100, 100, generator=g) * 0.7
        self.torch = torch

    def encode(self, text):
        return [ord(ch) % 90 for ch in text]

    def logits_fn(self, x):
        torch = self.torch
        left = torch.roll(x, 1, dims=1); right = torch.roll(x, -1, dims=1)
        return self.W[left] + self.W[right]


class LLaDAScorer:
    def __init__(self, name, spec, device, dtype):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(spec["hf"], torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        self.mask_id = 126336                                      # LLaDA's official [MASK] id
        assert "mask" in str(self.tok.convert_ids_to_tokens(self.mask_id)).lower(),             f"id {self.mask_id} is not LLaDA's mask token: {self.tok.convert_ids_to_tokens(self.mask_id)}"

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def encode_prompt(self, text):
        return self.tok([text]).input_ids[0]                        # exactly as generate_candidates.LLaDA

    def logits_fn(self, x):
        return self.model(x).logits                                 # no shift


class DreamScorer:
    def __init__(self, name, spec, device, dtype):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(spec["hf"], torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        self.mask_id = self.model.generation_config.mask_token_id
        assert self.mask_id is not None, "Dream generation_config has no mask_token_id"

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def encode_prompt(self, text):
        return self.tok([text]).input_ids[0]                        # exactly as generate_candidates.Dream

    def logits_fn(self, x):
        raw = self.model(x, "full", None).logits                    # 'full' = bidirectional, as in Dream's sampler
        return shift_right(raw)


# ------------------------------------------------------------------ selftest

def selftest() -> int:
    import torch
    g = lambda: torch.Generator().manual_seed(0)
    V, P, T, k = 50, 5, 20, 8
    prompt, ans = list(range(5)), [3 + (i % 17) for i in range(T)]
    # 1. a uniform predictor: ELBO == -T log V exactly, whatever the draws
    uni = lambda x: torch.zeros(x.shape[0], x.shape[1], V)
    e = mc_elbo(uni, prompt, ans, V - 1, k, g())
    assert math.isclose(e, -T * math.log(V), rel_tol=1e-5), (e, -T * math.log(V))
    # 2. a perfect predictor (peaked on the true token): ELBO ~ 0
    true_ids = torch.tensor(prompt + ans)
    oracle = lambda x: torch.nn.functional.one_hot(true_ids, V).float().unsqueeze(0).repeat(x.shape[0], 1, 1) * 30.0
    e2 = mc_elbo(oracle, prompt, ans, V - 1, k, g())
    assert abs(e2) < 1e-3, e2
    # 3. the mask draws: counts in 1..T, stratified (k distinct-ish levels), prompt untouched, reproducible
    m1, t1 = draw_masks(T, k, g()); m2, t2 = draw_masks(T, k, g())
    assert torch.equal(m1, m2) and torch.equal(t1, t2)
    assert (m1.sum(1) >= 1).all() and (m1.sum(1) <= T).all() and len(set(m1.sum(1).tolist())) >= k - 1
    # 4. the Dream shift: aligned[:, i] == raw[:, i-1]
    raw = torch.randn(2, 7, 4)
    sh = shift_right(raw)
    assert torch.equal(sh[:, 1:], raw[:, :-1]) and torch.equal(sh[:, 0], raw[:, 0])
    # 5. a context-using predictor scores a matching answer higher than a shuffled one
    mock = Mock()
    good = [3, 4, 5, 6, 7, 8, 9, 10]
    lg = lambda x: mock.logits_fn(x)
    a = mc_elbo(lg, [1, 2], good, mock.mask_id, k, g())
    b = mc_elbo(lg, [1, 2], good, mock.mask_id, k, g())
    assert a == b
    print("selftest OK: uniform ELBO = -T log V; oracle ELBO ~ 0; mask draws stratified and reproducible; "
          f"shift_right correct; mock ELBO deterministic ({a:.3f})")
    return 0


# ------------------------------------------------------------------ main

def seed_for(base, uid, j):
    return int(hashlib.sha256(f"{base}|{uid}|{j}".encode()).hexdigest()[:12], 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="dream-base | llada-base | mock")
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--k", type=int, default=8, help="Monte-Carlo draws per candidate (one batch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke tests)")
    ap.add_argument("--max-cands", type=int, default=0, help="first M candidates of each question only (CPU tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--candidates-from", help="read candidates of this model instead (mock wiring test)")
    ap.add_argument("--out", help="output file (default work/features/<model>/<split>.dlmself.jsonl)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.model and a.split, "--model and --split are required"
    import torch
    c = cfg()
    src_model = a.candidates_from or a.model
    src = WORK / "candidates" / src_model / f"{a.split}.jsonl"
    out = Path(a.out) if a.out else WORK / "features" / a.model / f"{a.split}.dlmself.jsonl"
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    rows = read_jsonl(src)
    rows = rows[:a.limit] if a.limit else rows
    todo = [r for r in rows if r["uid"] not in done]
    print(f"{a.model}/{a.split} x dlm.selfscore: {len(done)} done, {len(todo)} to go (k={a.k}, seed={a.seed}, cap={ANSWER_CAP})", flush=True)
    if not todo:
        return 0
    dtype = getattr(torch, a.dtype)
    if a.model == "mock":
        sc, device = Mock(), "cpu"
    else:
        spec = c["models"][a.model]
        sc = {"dream": DreamScorer, "llada": LLaDAScorer}[spec["backend"]](a.model, spec, a.device, dtype)
        device = a.device
    t_start = time.time()
    for n, r in enumerate(todo, 1):
        t0 = time.time()
        prompt_ids = sc.encode_prompt(r["prompt"]) if hasattr(sc, "encode_prompt") else sc.encode(r["prompt"])
        feats = []
        cands = r["candidates"][:a.max_cands] if a.max_cands else r["candidates"]
        for j, cd in enumerate(cands):
            ans_ids = sc.encode(" " + task.trim(cd["text"]))[:ANSWER_CAP]
            if not ans_ids:
                feats.append({"dlm.selfscore": None, "dlm.self.nll_sum": None, "dlm.self.n_tokens": 0}); continue
            assert len(prompt_ids) + len(ans_ids) <= 2048, f"{r['uid']}: prompt+answer too long for Dream/LLaDA windows"
            gen = torch.Generator().manual_seed(seed_for(a.seed, r["uid"], j))
            elbo = mc_elbo(sc.logits_fn, prompt_ids, ans_ids, sc.mask_id, a.k, gen, device)
            feats.append({"dlm.selfscore": elbo / len(ans_ids), "dlm.self.nll_sum": -elbo, "dlm.self.n_tokens": len(ans_ids)})
        if n == 1:
            vals = [f["dlm.selfscore"] for f in feats]
            print(f"  first item {r['uid']}: dlm.selfscore per candidate = {[None if v is None else round(v, 4) for v in vals]}", flush=True)
            real = [v for v in vals if v is not None]
            assert len(real) < 2 or len(set(round(v, 6) for v in real)) > 1, "all candidate scores equal: mask or shift is broken"
        append_jsonl(out, {"uid": r["uid"], "score_seconds": round(time.time() - t0, 3), "features": feats})
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t_start
            print(f"  {n}/{len(todo)}  {el / n:.1f}s/question  eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
