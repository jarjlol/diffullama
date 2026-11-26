"""N sweep and candidate-order sweep on the GSM-Symbolic candidate pool (#83).

Build (on the GPU box, after candidates exist):

    python3 gsm_symbolic_sweeps.py build --sweep n --questions 100
    python3 gsm_symbolic_sweeps.py build --sweep order --questions 100

Evaluate (after the result file is fetched back):

    python3 gsm_symbolic_sweeps.py eval --sweep n --results work/gsm_symbolic/sweeps/n.results.jsonl
    python3 gsm_symbolic_sweeps.py eval --sweep order --results work/gsm_symbolic/sweeps/order.results.jsonl

N sweep: listwise over the first N candidates, N in {2,4,6,8,10}, same wording
as the main listwise prompt. Request uid is "<question-uid>::n<N>", so the
result file cannot collide with the main run.
Order sweep: three seeded permutations of the same 10 candidates. Request uid
is "<question-uid>::o<seed>"; the permutation is a pure function of (uid, seed),
recomputed identically here and in eval. The comparison is on the chosen
*original candidate index*, never on the letter, so a position bias cannot be
mistaken for agreement.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random

from common import WORK, read_jsonl, task
from decision_rerank import LETTERS, candidate_text

HERE = WORK / "gsm_symbolic"
OUT = HERE / "sweeps"
SEEDS = (1, 2, 3)
NS = (2, 4, 6, 8, 10)


def order_for(uid: str, seed: int, n: int) -> list[int]:
    """Deterministic permutation of range(n), stable across processes."""
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(n))
    rng.shuffle(order)
    return order


def first_uids(n_questions: int) -> list[str]:
    qs = read_jsonl(HERE / "questions.jsonl")
    have = None
    for m in ("dream-base", "llada-base"):
        p = HERE / "candidates" / m / "test.jsonl"
        uids = {r["uid"] for r in read_jsonl(p)} if p.exists() else set()
        have = uids if have is None else (have & uids)
    return [r["uid"] for r in qs if r["uid"] in (have or set())][:n_questions]


def candidates() -> dict[str, dict[str, dict]]:
    out = {}
    for m in ("dream-base", "llada-base"):
        p = HERE / "candidates" / m / "test.jsonl"
        out[m] = {r["uid"]: r for r in read_jsonl(p)} if p.exists() else {}
    return out


def listwise_request(uid: str, model: str, question: str, texts: list[str], tag: str) -> dict:
    return {
        "uid": f"{uid}::{tag}", "model": model, "n": len(texts),
        "state": {"problem": question,
                  "candidates": {LETTERS[i]: t for i, t in enumerate(texts)}},
        "questions": {"best": {
            "type": "choice",
            "instructions": "Which candidate solution reaches the correct final answer to the math problem?",
            "criteria": {LETTERS[i]: f"Candidate {LETTERS[i]}" for i in range(len(texts))},
        }},
    }


def build(sweep: str, n_questions: int) -> None:
    uids = first_uids(n_questions)
    cands = candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{sweep}.prompts.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for model in ("dream-base", "llada-base"):
            for uid in uids:
                row = cands[model][uid]
                texts = [candidate_text(c["text"]) for c in row["candidates"]]
                if sweep == "n":
                    reqs = [listwise_request(uid, model, row["question"], texts[:N], f"n{N}")
                            for N in NS]
                else:
                    reqs = []
                    for seed in SEEDS:
                        order = order_for(uid, seed, len(texts))
                        reqs.append(listwise_request(uid, model, row["question"],
                                                     [texts[i] for i in order], f"o{seed}"))
                for req in reqs:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} {sweep} requests over {len(uids)} questions/model -> {path}")


def load_cands() -> dict[str, dict[str, dict]]:
    return candidates()


def eval_n(results: str, json_out: str | None) -> None:
    cands = load_cands()
    rows = [json.loads(l) for l in open(results, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "probs" in r]
    by = collections.defaultdict(dict)
    for r in rows:
        base, tag = r["uid"].rsplit("::", 1)
        assert tag.startswith("n")
        by[(r["model"], int(tag[1:]))][base] = r["probs"]
    table = {}
    for model in ("dream-base", "llada-base"):
        for N in NS:
            got = by.get((model, N), {})
            n_ok = n_maj = n_first = n_or = 0
            for uid, probs in got.items():
                row = cands[model][uid]
                cs = row["candidates"][:N]
                gold = GOLD[uid]
                ok = [task.is_correct(c.get("answer"), gold) for c in cs]
                j = LETTERS.index(max(probs, key=probs.get))
                n_ok += int(ok[j])
                n_maj += int(ok[majority_index([c.get("answer") for c in cs])])
                n_first += int(ok[0])
                n_or += int(any(ok))
            n = len(got)
            table[f"{model}|{N}"] = {"n": n, "listwise": n_ok / n if n else None,
                                     "majority": n_maj / n if n else None,
                                     "first": n_first / n if n else None,
                                     "oracle": n_or / n if n else None}
    _print_n(table, json_out)


GOLD: dict[str, str] = {}


def majority_index(answers: list) -> int:
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return 0
    best = max(counts.values())
    return next(i for i, a in enumerate(answers) if a is not None and counts[a] == best)


def _print_n(table: dict, json_out: str | None) -> None:
    print(f"\n{'model':12s} {'N':>2s} {'n':>4s} {'listwise':>9s} {'majority':>9s} "
          f"{'delta':>7s} {'first':>7s} {'oracle':>7s}")
    for key, r in table.items():
        model, N = key.split("|")
        d = (r["listwise"] - r["majority"]) if r["listwise"] is not None and r["majority"] is not None else None
        print(f"{model:12s} {N:>2s} {r['n']:4d} {r['listwise']:9.3f} {r['majority']:9.3f} "
              f"{d:+7.3f} {r['first']:7.3f} {r['oracle']:7.3f}")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
        print(f"wrote {json_out}")


def eval_order(results: str, json_out: str | None) -> None:
    cands = load_cands()
    rows = [json.loads(l) for l in open(results, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "probs" in r]
    by = collections.defaultdict(dict)          # (model, seed) -> uid -> chosen original idx
    correct = collections.defaultdict(dict)     # (model, seed) -> uid -> bool
    positions = collections.Counter()
    for r in rows:
        base, tag = r["uid"].rsplit("::", 1)
        assert tag.startswith("o")
        seed = int(tag[1:])
        row = cands[r["model"]][base]
        n = len(row["candidates"])
        order = order_for(base, seed, n)
        j = LETTERS.index(max(r["probs"], key=r["probs"].get))
        orig = order[j]
        by[(r["model"], seed)][base] = orig
        correct[(r["model"], seed)][base] = task.is_correct(
            row["candidates"][orig].get("answer"), GOLD[base])
        positions[(r["model"], LETTERS[j])] += 1
    out = {}
    for model in ("dream-base", "llada-base"):
        per_seed = {}
        for seed in SEEDS:
            got = correct.get((model, seed), {})
            per_seed[str(seed)] = {"n": len(got), "accuracy": sum(got.values()) / len(got) if got else None}
        uids = set.intersection(*[set(by.get((model, s), {})) for s in SEEDS]) if any(
            by.get((model, s)) for s in SEEDS) else set()
        same = sum(1 for u in uids if len({by[(model, s)][u] for s in SEEDS}) == 1)
        # baseline: the unpermuted main listwise pick, if features exist
        feat = {}
        fp = HERE / "features" / model / "test.decision-listwise.jsonl"
        if fp.exists():
            feat = {r["uid"]: r["features"] for r in read_jsonl(fp)}
        base_ok, base_n = 0, 0
        for u in uids:
            fs = feat.get(u)
            if not fs:
                continue
            j = max(range(len(fs)), key=lambda i: fs[i].get("decision.listwise.p", -1))
            base_ok += int(task.is_correct(cands[model][u]["candidates"][j].get("answer"), GOLD[u]))
            base_n += 1
        out[model] = {"per_seed": per_seed, "n_common": len(uids),
                      "same_candidate_all3": same / len(uids) if uids else None,
                      "unpermuted_accuracy": base_ok / base_n if base_n else None,
                      "unpermuted_n": base_n}
    print(f"\n{'model':12s} {'seed1':>7s} {'seed2':>7s} {'seed3':>7s} {'same all3':>10s} {'plain':>7s}")
    for model, r in out.items():
        vals = [r["per_seed"][str(s)]["accuracy"] for s in SEEDS]
        print(f"{model:12s} " + " ".join(f"{v:7.3f}" if v is not None else f"{'-':>7s}" for v in vals)
              + f" {r['same_candidate_all3']:10.3f} "
              + (f"{r['unpermuted_accuracy']:7.3f}" if r["unpermuted_accuracy"] is not None else f"{'-':>7s}"))
    print("\nchosen position distribution")
    for model in ("dream-base", "llada-base"):
        tot = sum(v for (m, _), v in positions.items() if m == model)
        if not tot:
            continue
        dist = {L: positions[(model, L)] / tot for L in LETTERS[:10]}
        print(f"  {model}: " + " ".join(f"{L}={dist[L]:.2f}" for L in LETTERS[:10]))
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"per_model": out,
                       "positions": {f"{m}|{L}": v for (m, L), v in sorted(positions.items())}},
                      f, indent=2)
        print(f"wrote {json_out}")


def main() -> int:
    global GOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "eval"])
    ap.add_argument("--sweep", choices=["n", "order"], required=True)
    ap.add_argument("--questions", type=int, default=100)
    ap.add_argument("--results", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    GOLD = {r["uid"]: task.extract_answer(r["answer"]) or r["answer"]
            for r in read_jsonl(HERE / "questions.jsonl")}
    if a.action == "build":
        build(a.sweep, a.questions)
    elif a.sweep == "n":
        eval_n(a.results, a.json)
    else:
        eval_order(a.results, a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
