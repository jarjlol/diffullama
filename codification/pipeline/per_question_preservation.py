"""Iteration 2, Tier 2 (task T-b): does PER-QUESTION preservation of the AR representation predict rerank success?

Idea 1.1 predicts that a diffusion LM (DLM) that stays close to its AR base is the one a small AR scorer can
rerank. Iteration 1 could only test that across 4 models (n = 4). This script tests the same claim across the
500 TEST QUESTIONS of ONE model pair, Dream-v0-Base-7B and its AR base Qwen2.5-7B (n = 500). EXPLORATORY: it was
written before any outcome was looked at, but nothing here is pre-registered beyond the choices below.

    python3 per_question_preservation.py extract --model dream   --split test    # GPU, Dream venv
    python3 per_question_preservation.py extract --model qwen7b  --split test    # GPU, scorer venv (no attention_patch)
    python3 per_question_preservation.py analyze --split test                    # CPU, numpy only
    python3 per_question_preservation.py selftest                                # CPU, analysis on planted data

WHAT IS MEASURED (extract). For every question the text "Question: {q}\\nAnswer:" (the AR scorers' context, no
shots) goes through the model once, batch size 1. Forward hooks on every transformer block (as extract_hidden.py)
capture the block output, mean-pooled over tokens: `all` = every position, `nosink` = positions 1.., because causal
LMs put a huge activation on position 0 (see extract_hidden.py docstring). Dream runs with the bidirectional
attention it was trained with (`net(input_ids=x)`, asserted at load: changing the last token must change earlier
outputs); Qwen runs causal (asserted). Saved to work/preservation_pq/<model>.<split>.pt, a NEW directory.

WHAT IS TESTED (analyze).
  predictor  s_q  = layer-mean (28 blocks) of the CENTERED cosine between Dream's and Qwen's pooled vector for
             question q, `nosink` pooling. Centered = subtract each model's own mean over the questions, per layer,
             before the cosine: raw cosines of mean-pooled LLM states are near 1 for every question (shared
             large-norm directions), so only the question-specific part says anything. Raw (uncentered) cosine and
             the `all` pooling are reported as sensitivity checks. Per-layer results are reported too.
  outcomes   O1  first sample wrong AND the Qwen2.5-7B AR pick right (AR pick = lowest summed NLL of --scorer, default qwen7b-cut, the
                 `ar.<scorer>.sum` selector of baselines.py; the per-token variant is O1m)
             O2  majority vote right
             O3  AR pick right among the questions whose first sample is wrong (conditional version of O1)
             control: oracle (any of the 10 candidates right) = how easy the question is
  tests      rank test: AUC of s_q for the outcome (P[s_pos > s_neg], ties half), two-sided permutation p
             (10 000 label permutations, fixed seed); logistic regression y ~ 1 + z(s_q) + z(log prompt tokens),
             IRLS, odds ratio per SD with a Wald 95% CI. Question length is a confounder (it moves both the cosine
             and the difficulty), hence the covariate. Difficulty (oracle) is NOT a covariate: it is itself an
             outcome of the same candidates (post-treatment), it is reported as a correlation with s_q instead.
  Multiplicity: the layer table has 28 tests per outcome; Benjamini-Hochberg q-values are printed next to p.
Methods are stated in the output header. Everything uses numpy only (no scipy in the venvs).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from common import TASK, WORK, cfg, read_jsonl, task

PQ_DIR = WORK / "preservation_pq"


def context_text(q: str) -> str:
    return f"Question: {q}\nAnswer:"


# ------------------------------------------------------------------ extract

def extract(a) -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from extract_hidden import find_blocks
    c = cfg()
    hf = c["models"]["dream-base"]["hf"] if a.model == "dream" else c["scorers"]["qwen7b"]
    qs = read_jsonl(WORK / "data" / f"{a.split}.jsonl")
    qs = qs[:a.limit] if a.limit else qs
    out = Path(a.out) if a.out else PQ_DIR / f"{a.model}.{a.split}.pt"
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    net = AutoModel.from_pretrained(hf, torch_dtype=dtype, trust_remote_code=True).to(a.device).eval()
    run = lambda x: net(input_ids=x)

    ids0 = tok(context_text(qs[0]["question"]), add_special_tokens=False).input_ids[:16]
    x = torch.tensor([ids0], device=a.device); alt = x.clone(); alt[0, -1] = (alt[0, -1] + 7) % 30000
    with torch.no_grad():
        d = (run(x)[0][0, :-1].float() - run(alt)[0][0, :-1].float()).abs().max().item()
    assert (d > 1e-2) == (a.model == "dream"), \
        f"expected {'bidirectional' if a.model == 'dream' else 'causal'} attention, earlier outputs changed by {d:.3g}"
    print(f"{a.model}: attention check ok ({'bidirectional' if a.model == 'dream' else 'causal'}, max change {d:.3g})", flush=True)

    name, blocks = find_blocks(net)
    L = len(blocks)
    cur = {}
    def capture(i):
        def hook(mod, inp, outp):
            h = (outp[0] if isinstance(outp, tuple) else outp).float()
            cur[i] = (h.mean(1)[0].cpu(), h[:, 1:].mean(1)[0].cpu())
        return hook
    hooks = [b.register_forward_hook(capture(i)) for i, b in enumerate(blocks)]
    uids, ntok, ids_all, allp, nsp = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for n, q in enumerate(qs, 1):
            ids = tok(context_text(q["question"]), add_special_tokens=False).input_ids
            run(torch.tensor([ids], device=a.device))
            allp.append(torch.stack([cur[i][0] for i in range(L)])); nsp.append(torch.stack([cur[i][1] for i in range(L)]))
            uids.append(q["uid"]); ntok.append(len(ids)); ids_all.append(ids)
            if n % 100 == 0 or n == len(qs):
                print(f"  {n}/{len(qs)}  {(time.time() - t0) / n:.2f}s/question", flush=True)
    for h in hooks:
        h.remove()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    torch.save({"hf": hf, "blocks": name, "uids": uids, "n_tokens": ntok, "ids": ids_all,
                "all": torch.stack(allp), "nosink": torch.stack(nsp)}, tmp)
    tmp.replace(out)
    print(f"{a.model}: {hf}  {len(uids)} questions x {L} blocks x {allp[0].shape[1]} dims -> {out}")
    return 0


# ------------------------------------------------------------------ numpy statistics

def centered_cosine(A, B):
    """A, B: [N, L, d]. Returns [N, L] cosine of the per-layer mean-centred rows."""
    import numpy as np
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    return (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12)


def raw_cosine(A, B):
    import numpy as np
    return (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12)


def rank_auc(s, y):
    """AUC = P[s_pos > s_neg] + 0.5 P[tie], from average ranks (numpy only)."""
    import numpy as np
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    sv = np.asarray(s)[order]; i = 0
    while i < len(sv):                                             # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), ranks


def perm_p(s, y, n_perm=10000, seed=0):
    """Two-sided permutation p for |AUC - 0.5|, shuffling y."""
    import numpy as np
    auc, ranks = rank_auc(s, y)
    n1 = int(y.sum()); n0 = len(y) - n1
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perm, len(y))), axis=1)[:, :n1]
    aucs = (ranks[idx].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    p = (1 + (np.abs(aucs - 0.5) >= abs(auc - 0.5) - 1e-12).sum()) / (n_perm + 1)
    return auc, float(p)


def logistic(X, y, iters=50):
    """IRLS logistic regression. X [n, p] with an intercept column. Returns beta, se."""
    import numpy as np
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    mu = 1 / (1 + np.exp(-(X @ b))); W = mu * (1 - mu)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]))))
    return b, se


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def bh(pvals):
    import numpy as np
    p = np.asarray(pvals); o = np.argsort(p); q = np.empty(len(p))
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * len(p) / rank); q[i] = prev
    return q


def test_outcome(s, y, length, n_perm, seed):
    import numpy as np
    n, ev = len(y), int(y.sum())
    if ev < 5 or n - ev < 5:
        return {"n": n, "events": ev, "auc": None}
    auc, p = perm_p(s, y, n_perm, seed)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    X = np.column_stack([np.ones(n), z(s), z(np.log(length))])
    b, se = logistic(X, y.astype(float))
    return {"n": n, "events": ev, "auc": float(auc), "perm_p": p,
            "or_per_sd": float(math.exp(b[1])), "or_ci95": [float(math.exp(b[1] - 1.96 * se[1])), float(math.exp(b[1] + 1.96 * se[1]))],
            "wald_p": norm_p(b[1] / se[1]), "or_length_per_sd": float(math.exp(b[2]))}


# ------------------------------------------------------------------ outcomes from the cached candidates

def outcomes(split, uids, scorer):
    import numpy as np
    sys.path.insert(0, str(TASK))
    from baselines import _argbest, _majority, _mean_nll
    gold = {r["uid"]: r["answer"] for r in read_jsonl(WORK / "data" / f"{split}.jsonl")}
    cands = {r["uid"]: r for r in read_jsonl(WORK / "candidates" / "dream-base" / f"{split}.jsonl")}
    feats = {r["uid"]: r["features"] for r in read_jsonl(WORK / "features" / "dream-base" / f"{split}.{scorer}.jsonl")}
    rows = []
    for u in uids:
        cs = [{"answer": cd["answer"], "features": feats[u][j]} for j, cd in enumerate(cands[u]["candidates"])]
        ok = [task.is_correct(cd["answer"], gold[u]) for cd in cs]
        pick_sum = _argbest(cs, f"ar.{scorer}.nll_sum"); pick_mean = _mean_nll(cs, scorer)
        rows.append({"first": ok[0], "maj": ok[_majority(cs)], "oracle": any(ok), "share": sum(ok) / len(ok),
                     "pick": None if pick_sum is None else ok[pick_sum], "pickm": None if pick_mean is None else ok[pick_mean]})
    f = lambda k: np.array([bool(r[k]) for r in rows])
    first, pick, pickm = f("first"), f("pick"), f("pickm")
    return {"O1": (~first) & pick, "O1m": (~first) & pickm, "O2": f("maj"), "O3_mask": ~first, "O3": pick,
            "oracle": f("oracle"), "share": np.array([r["share"] for r in rows]), "first": first}


def analyze(a) -> int:
    import numpy as np
    import torch
    d = Path(a.dir) if a.dir else PQ_DIR
    dre = torch.load(d / f"dream.{a.split}.pt"); qwe = torch.load(d / f"qwen7b.{a.split}.pt")
    assert dre["uids"] == qwe["uids"], "different question lists"
    # the extraction covers every test question in work/data/test.jsonl (1319); outcomes exist only for those with cached
    # candidates AND the chosen AR scorer's features (the first 500). Keep exactly those, in file order.
    have = {r["uid"] for r in read_jsonl(WORK / "candidates" / "dream-base" / f"{a.split}.jsonl")} &            {r["uid"] for r in read_jsonl(WORK / "features" / "dream-base" / f"{a.split}.{a.scorer}.jsonl")}
    idx = [i for i, u in enumerate(dre["uids"]) if u in have]
    assert idx, "no question has both hidden states and candidates"
    uids = [dre["uids"][i] for i in idx]
    A = {k: dre[k].numpy().astype(np.float64)[idx] for k in ("all", "nosink")}
    B = {k: qwe[k].numpy().astype(np.float64)[idx] for k in ("all", "nosink")}
    dre["n_tokens"] = [dre["n_tokens"][i] for i in idx]; dre["ids"] = [dre["ids"][i] for i in idx]; qwe["ids"] = [qwe["ids"][i] for i in idx]
    assert A["all"].shape == B["all"].shape, (A["all"].shape, B["all"].shape)
    N, L, _ = A["all"].shape
    same_ids = sum(i == j for i, j in zip(dre["ids"], qwe["ids"]))
    length = np.array(dre["n_tokens"], dtype=float)
    y = outcomes(a.split, uids, a.scorer)
    pred = {"centered/nosink (PRIMARY)": centered_cosine(A["nosink"], B["nosink"]),
            "centered/all": centered_cosine(A["all"], B["all"]),
            "raw/nosink": raw_cosine(A["nosink"], B["nosink"]),
            "raw/all": raw_cosine(A["all"], B["all"])}
    lines = [f"per-question preservation, Dream-v0-Base-7B vs Qwen2.5-7B, split {a.split}, n = {N} questions, {L} blocks",
             f"AR pick = lowest summed NLL of scorer '{a.scorer}' ('qwen7b-cut' = text cut after the '#### N' line, 'qwen7b' = iteration-1 trimmed text) on the cached 64-step Dream candidates",
             f"tokenisations identical for {same_ids}/{N} questions (Dream and Qwen tokenizers); permutations {a.perms}, seed {a.seed}",
             "predictor: layer-mean cosine of the pooled hidden states of the question text; primary = centered, no-sink pooling",
             "tests: AUC + two-sided permutation p; logistic y ~ z(s) + z(log prompt tokens), odds ratio per SD, Wald 95% CI",
             f"base rates: first right {y['first'].mean():.3f}, O1 {y['O1'].mean():.3f}, O1m {y['O1m'].mean():.3f}, O2 {y['O2'].mean():.3f}, "
             f"oracle {y['oracle'].mean():.3f}; n(first wrong) = {int((~y['first']).sum())}", ""]
    res = {}
    s_prim = pred["centered/nosink (PRIMARY)"].mean(1)
    lines.append("primary predictor summary: mean %.4f sd %.4f; corr with log length %.3f; Pearson with share-correct %.3f" % (
        s_prim.mean(), s_prim.std(), np.corrcoef(s_prim, np.log(length))[0, 1], np.corrcoef(s_prim, y["share"])[0, 1]))
    lines.append("")
    hdr = f"{'predictor':28s} {'outcome':6s} {'n':>4s} {'events':>6s} {'AUC':>6s} {'perm p':>8s} {'OR/SD':>6s} {'95% CI':>14s} {'Wald p':>8s}"
    lines.append(hdr); lines.append("-" * len(hdr))
    for pname, P in pred.items():
        s = P.mean(1)
        for oname in ("O1", "O1m", "O2"):
            r = test_outcome(s, y[oname], length, a.perms, a.seed); res[f"{pname}|{oname}"] = r
            lines.append(fmt_row(pname, oname, r))
        m = y["O3_mask"]
        r = test_outcome(s[m], y["O3"][m], length[m], a.perms, a.seed); res[f"{pname}|O3"] = r
        lines.append(fmt_row(pname, "O3", r))
    lines.append("")
    lines.append("per-layer, centered/nosink, outcome O1 (first wrong AND AR pick right):")
    lines.append(f"{'layer':>5s} {'AUC':>6s} {'perm p':>8s} {'BH q':>7s}")
    per = [test_outcome(pred["centered/nosink (PRIMARY)"][:, l], y["O1"], length, min(a.perms, 2000), a.seed + l) for l in range(L)]
    q = bh([r["perm_p"] for r in per]) if all(r.get("auc") is not None for r in per) else [float("nan")] * L
    for l, r in enumerate(per):
        lines.append(f"{l:5d} {r['auc']:6.3f} {r['perm_p']:8.4f} {q[l]:7.3f}" if r.get("auc") is not None else f"{l:5d}  (too few events)")
    res["per_layer_O1"] = per
    text = "\n".join(lines)
    print(text)
    if a.json:
        import json
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"n": N, "layers": L, "results": res}, indent=1), encoding="utf-8")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text + "\n", encoding="utf-8")
    return 0


def fmt_row(pname, oname, r):
    if r.get("auc") is None:
        return f"{pname:28s} {oname:6s} {r['n']:4d} {r['events']:6d}   (too few events)"
    return (f"{pname:28s} {oname:6s} {r['n']:4d} {r['events']:6d} {r['auc']:6.3f} {r['perm_p']:8.4f} {r['or_per_sd']:6.2f} "
            f"[{r['or_ci95'][0]:5.2f},{r['or_ci95'][1]:5.2f}] {r['wald_p']:8.4f}")


# ------------------------------------------------------------------ selftest (planted data, CPU)

def selftest() -> int:
    import numpy as np
    rng = np.random.default_rng(0)
    n = 500
    s = rng.normal(size=n); length = np.exp(rng.normal(4.5, 0.4, n))
    # planted effect: log-odds rise 0.5 per SD of s
    y_eff = (rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 0.5 * s)))).astype(int)
    y_null = (rng.random(n) < 0.3).astype(int)
    e = test_outcome(s, y_eff, length, 2000, 0); u = test_outcome(s, y_null, length, 2000, 0)
    assert e["auc"] > 0.55 and e["perm_p"] < 0.01 and 1.2 < e["or_per_sd"] < 2.4, e
    assert u["perm_p"] > 0.01 and 0.7 < u["or_per_sd"] < 1.4, u
    # AUC against the textbook definition on a tiny example with ties
    ss, yy = np.array([1, 2, 2, 3, 5.0]), np.array([0, 0, 1, 1, 1])
    auc, _ = rank_auc(ss, yy)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in ss[yy == 1] for b in ss[yy == 0]])
    assert math.isclose(auc, brute), (auc, brute)
    # logistic recovers a known coefficient with a large sample
    big = rng.normal(size=20000); yb = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 0.8 * big)))).astype(float)
    b, se = logistic(np.column_stack([np.ones(20000), big]), yb)
    assert abs(b[1] - 0.8) < 0.08 and abs(b[0] - 0.3) < 0.08, b
    # centred cosine: identical question-specific parts give 1, raw cosine of shifted data is near 1 regardless
    A = rng.normal(size=(40, 3, 16)); shift = 50 * rng.normal(size=(1, 3, 16))
    assert np.allclose(centered_cosine(A + shift, 2.0 * A - shift), 1.0, atol=1e-9)
    assert raw_cosine(A + shift, 2.0 * A - shift).min() < raw_cosine(A + shift, A + shift).min() + 1e-9
    q = bh([0.001, 0.02, 0.5]); assert q[0] <= q[1] <= q[2] and q[2] <= 0.5 + 1e-9
    print(f"selftest OK: planted effect found (AUC {e['auc']:.3f}, perm p {e['perm_p']:.4f}, OR/SD {e['or_per_sd']:.2f}); "
          f"null not found (perm p {u['perm_p']:.3f}, OR/SD {u['or_per_sd']:.2f}); AUC = brute force; logistic coef recovered "
          f"({b[1]:.3f} vs 0.8); centred cosine removes a shared shift; BH monotone")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", choices=["dream", "qwen7b"], required=True)
    e.add_argument("--split", choices=["dev", "test"], default="test")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    e.add_argument("--out")
    an = sub.add_parser("analyze")
    an.add_argument("--split", choices=["dev", "test"], default="test")
    an.add_argument("--scorer", default="qwen7b-cut", help="AR scorer feature file for the 'AR pick' outcome (default: the cut-text re-scoring)")
    an.add_argument("--perms", type=int, default=10000)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--dir", help="where extract wrote its files (default work/preservation_pq)")
    an.add_argument("--json"); an.add_argument("--report")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"extract": extract, "analyze": analyze, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
