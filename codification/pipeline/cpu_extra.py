"""Seven exploratory CPU analyses (rules fixed in results/iter2/cpu_extra_plan.md before any result). Stdlib only.

    python3 cpu_extra.py --item 1                 # equal compute (test)
    python3 cpu_extra.py --item 2                 # qwen7b-cut summed score vs first sample (test)
    python3 cpu_extra.py --item 4 --split dev     # tie-break rule, dev first ...
    python3 cpu_extra.py --item 4 --split test    # ... then test once
    python3 cpu_extra.py --item 5 | 6             # self-score at K=256 | vote share as confidence
    python3 cpu_extra.py --item 7 --split dev|test

Reuses step_arm.load/paired (iteration-1 bootstrap CI, exact McNemar), baselines._majority/_argbest and task.is_correct.
Reads CODIFY_WORK_DIR read-only; writes results/iter2/cpu/extra_<item>_*.{txt,json}. Item 3 (report refresh) is not in this file.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import random
import statistics
import sys
from pathlib import Path

from common import HERE, WORK, read_jsonl, task
import step_arm  # also puts the protected task dir on sys.path
from step_arm import load, paired, fmt_pair
import baselines  # noqa: E402

MODELS = ["dream-base", "llada-base"]
TAGS = {16: "@k16", 32: "@k32", 64: "", 256: "@k256"}
QWEN_CUT = "ar.qwen7b-cut"


class Out:
    """Print and keep every line; also collects the JSON."""
    def __init__(self, header: str):
        self.lines: list[str] = []
        self.js: dict = {}
        self.p(header)

    def p(self, s: str = "") -> None:
        print(s); self.lines.append(s)

    def save(self, base: Path) -> None:
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        base.with_suffix(".json").write_text(json.dumps(self.js, indent=2), encoding="utf-8")
        print(f"wrote {base.with_suffix('.txt')} and .json")


def header(item: str) -> str:
    return (f"# {item}\n# command: python3 cpu_extra.py {' '.join(sys.argv[1:])}\n# run: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}"
            f"\n# plan: results/iter2/cpu_extra_plan.md (EXPLORATORY, post-hoc; rules fixed before any result)\n# work dir: {WORK}")


def gold_of(split: str) -> dict:
    return {q["uid"]: q["answer"] for q in read_jsonl(WORK / "data" / f"{split}.jsonl")}


def ok(cands: list, i, g) -> bool:
    return bool(i is not None and task.is_correct(cands[i].get("answer"), g))


def maj_sub(cands: list, idx: list):
    """_majority over the sub-list cands[idx] (idx sorted ascending); returns the ORIGINAL candidate index."""
    sub = [cands[j] for j in idx]
    return idx[baselines._majority(sub)]


def mean(x) -> float:
    x = list(x)
    return sum(x) / len(x)


def bw_secs(rows: dict, uids) -> list:
    us = set(uids)
    return [r["gen_seconds"] for u, r in rows.items() if u in us and "gen_seconds" in r
            and ("Blackwell" in str(r.get("gpu", "")) or ("gpu" not in r and "host" not in r))]


def check_ten(rows: dict, label: str) -> None:
    bad = [u for u, r in rows.items() if len(r["candidates"]) != 10]
    assert not bad, f"{label}: {len(bad)} questions without exactly 10 candidates"


# ---------------------------------------------------------------- item 1
def item1(o: Out) -> None:
    o.p("\n# ITEM 1. Equal compute: denoising steps per question (n candidates x K). Equal steps is NOT equal wall-clock time:")
    o.p("# batching, shared cards, Ada rows in the K=256 subset and per-sample overheads differ; seconds below are Blackwell rows only.")
    gold = gold_of("test")
    rng = random.Random(0)
    subsets = {n: [sorted(rng.sample(range(10), n)) for _ in range(200)] for n in range(1, 11)}
    for m in MODELS:
        rows = {K: load(m, t) for K, t in TAGS.items()}
        for K, r in rows.items():
            check_ten(r, f"{m} K={K}")
        sub = sorted(set(rows[256]) & set(rows[64]) & set(rows[32]) & set(rows[16]))
        all500 = sorted(set(rows[64]) & set(rows[32]) & set(rows[16]))
        o.p(f"\n== {m}: K=256 subset n = {len(sub)}; K<=64 on {len(all500)} questions")
        res: dict = {"n_subset": len(sub), "n_all": len(all500)}
        first256 = [ok(rows[256][u]["candidates"], 0, gold[u]) for u in sub]
        maj4_64 = [ok(rows[64][u]["candidates"], maj_sub(rows[64][u]["candidates"], [0, 1, 2, 3]), gold[u]) for u in sub]
        maj8_32 = [ok(rows[32][u]["candidates"], maj_sub(rows[32][u]["candidates"], list(range(8))), gold[u]) for u in sub]
        res["acc"] = {"K256_first": mean(first256), "K64_maj_first4": mean(maj4_64), "K32_maj_first8": mean(maj8_32)}
        o.p(f"  accuracy on the subset: K=256 first {mean(first256):.3f}; K=64 majority of candidates 0-3 {mean(maj4_64):.3f}; "
            f"K=32 majority of candidates 0-7 {mean(maj8_32):.3f}")
        res["paired"] = {}
        for label, a, b in (("PRIMARY K=256 first minus K=64 majority-of-4 (256 steps each)", first256, maj4_64),
                            ("K=256 first minus K=32 majority-of-8 (256 steps each)", first256, maj8_32)):
            r = paired(a, b); res["paired"][label] = r; o.p(fmt_pair(label, r))
        curves = {}
        o.p("  majority@n by steps per question x = n*K; 'first-n' = candidates 0..n-1, 'rand' = mean over 200 random n-subsets (seed 0)")
        for label, K, us in (("K16 (500)", 16, all500), ("K32 (500)", 32, all500), ("K64 (500)", 64, all500),
                             ("K64 (subset)", 64, sub), ("K256 (subset)", 256, sub)):
            first_n, rand_n = [], []
            for n in range(1, 11):
                idx = list(range(n))
                first_n.append(mean(ok(rows[K][u]["candidates"], maj_sub(rows[K][u]["candidates"], idx), gold[u]) for u in us))
                if n == 10:
                    rand_n.append(first_n[-1]); continue      # all ten candidates: every subset is the full set
                tot = 0
                for u in us:
                    c = rows[K][u]["candidates"]
                    tot += sum(ok(c, maj_sub(c, s), gold[u]) for s in subsets[n])
                rand_n.append(tot / (len(us) * len(subsets[n])))
            secs = bw_secs(rows[K], us)
            curves[label] = {"K": K, "n_questions": len(us), "steps_per_question": [n * K for n in range(1, 11)],
                             "acc_first_n": first_n, "acc_random_subsets": rand_n,
                             "gen_s_per_q_N10_blackwell": mean(secs) if secs else None, "n_blackwell_rows": len(secs)}
            o.p(f"  {label:14s} n_q={len(us):3d}  s/q(N=10)={mean(secs) if secs else float('nan'):6.1f} (Blackwell rows {len(secs)}); proxy s/q at n = 0.1*n*that")
            o.p("    n          " + " ".join(f"{n:6d}" for n in range(1, 11)))
            o.p("    steps/q    " + " ".join(f"{n * K:6d}" for n in range(1, 11)))
            o.p("    first-n    " + " ".join(f"{v:6.3f}" for v in first_n))
            o.p("    rand-200   " + " ".join(f"{v:6.3f}" for v in rand_n))
        res["curves"] = curves
        o.js[m] = res


# ---------------------------------------------------------------- item 2
def item2(o: Out) -> None:
    o.p(f"\n# ITEM 2. {QWEN_CUT}.sum (lowest summed NLL on cut text) minus first sample, paired")
    gold = gold_of("test")
    for m in MODELS:
        res = {}
        o.p(f"\n== {m}")
        for K in (16, 32, 64, 256):
            rows = load(m, TAGS[K])
            us = sorted(rows if K == 256 else set(rows) & set(load(m, "")))
            ar = [ok(rows[u]["candidates"], baselines._argbest(rows[u]["candidates"], f"{QWEN_CUT}.nll_sum"), gold[u]) for u in us]
            fi = [ok(rows[u]["candidates"], 0, gold[u]) for u in us]
            r = paired(ar, fi); r["acc_ar"] = mean(ar); r["acc_first"] = mean(fi); r["n"] = len(us); res[f"K{K}"] = r
            o.p(f"  K={K:<3d} n={len(us):3d} AR {mean(ar):.3f} first {mean(fi):.3f}")
            o.p(fmt_pair(f"K={K}: AR(qwen7b-cut sum) minus first", r))
        o.js[m] = res


# ---------------------------------------------------------------- item 4
def tiebreak_idx(cands: list):
    """Plan item 4. Returns (index, was_tie). Votes over non-None answers; ties among the top answers go to the answer whose best
    candidate has the lowest per-token mean NLL of QWEN_CUT; no scored tied answer -> _majority; all None -> index 0."""
    ans = [c.get("answer") for c in cands]
    cnt = collections.Counter(a for a in ans if a is not None)
    if not cnt:
        return 0, False
    best = max(cnt.values())
    T = {a for a, v in cnt.items() if v == best}
    if len(T) == 1:
        return baselines._majority(cands), False
    sc: dict = {}
    for i, c in enumerate(cands):
        a = ans[i]
        if a in T:
            f = c["features"]; tot, n = f.get(f"{QWEN_CUT}.nll_sum"), f.get(f"{QWEN_CUT}.n_tokens")
            if tot is not None and n:
                v = (tot / n, i)
                if a not in sc or v < sc[a]:
                    sc[a] = v
    if not sc:
        return baselines._majority(cands), True
    return sc[min(sc, key=lambda a: sc[a])][1], True


def item4(o: Out, split: str) -> None:
    o.p(f"\n# ITEM 4. Majority with cut-text AR tie-break (K=64, {split}); rule in the plan, fixed before any result")
    gold = gold_of(split)
    for m in MODELS:
        rows = load(m, "", split)
        check_ten(rows, m)
        us = sorted(rows)
        missing = sum(1 for u in us for c in rows[u]["candidates"] if f"{QWEN_CUT}.nll_sum" not in c["features"])
        tb, mj, ties, differ = [], [], 0, 0
        for u in us:
            c = rows[u]["candidates"]
            i, tie = tiebreak_idx(c); j = baselines._majority(c)
            ties += tie
            differ += (c[i].get("answer") != c[j].get("answer"))
            tb.append(ok(c, i, gold[u])); mj.append(ok(c, j, gold[u]))
        r = paired(tb, mj)
        r.update({"n": len(us), "tie_questions": ties, "choice_differs": differ, "acc_tiebreak": mean(tb), "acc_majority": mean(mj),
                  "candidates_without_qwen7b_cut_score": missing})
        o.p(f"\n== {m} {split}: n={len(us)}  majority {mean(mj):.3f}  majority+AR tie-break {mean(tb):.3f}  "
            f"questions with a top-vote tie {ties}; choice differs from plain majority on {differ}; candidates missing a score {missing}")
        o.p(fmt_pair("tie-break minus plain majority", r))
        o.js[m] = r


# ---------------------------------------------------------------- item 5
def item5(o: Out) -> None:
    o.p("\n# ITEM 5. DLM self-score at K=256 (highest wins): dlm.selfscore (cut text) and dlm.selftrim (trimmed text), paired vs majority and vs first")
    gold = gold_of("test")
    for m in MODELS:
        rows = load(m, "@k256")
        check_ten(rows, m)
        us = sorted(rows)
        o.p(f"\n== {m}: n = {len(us)}")
        maj = [ok(rows[u]["candidates"], baselines._majority(rows[u]["candidates"]), gold[u]) for u in us]
        fi = [ok(rows[u]["candidates"], 0, gold[u]) for u in us]
        res = {"n": len(us), "acc_majority": mean(maj), "acc_first": mean(fi), "paired": {}}
        for key in ("dlm.selfscore", "dlm.selftrim"):
            idx = {u: baselines._argbest(rows[u]["candidates"], key, lower=False) for u in us}
            none = sum(i is None for i in idx.values())
            sel = [ok(rows[u]["candidates"], idx[u], gold[u]) for u in us]
            res[key] = {"acc": mean(sel), "questions_without_score": none}
            o.p(f"  {key}: accuracy {mean(sel):.3f} (majority {mean(maj):.3f}, first {mean(fi):.3f}); questions with no score {none}")
            for ref, refv in (("majority", maj), ("first", fi)):
                r = paired(sel, refv); res["paired"][f"{key} - {ref}"] = r
                o.p(fmt_pair(f"{key} minus {ref}", r))
        o.js[m] = res


# ---------------------------------------------------------------- item 6
def auroc(scores: list, labels: list):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    w = 0.0
    for a in pos:
        for b in neg:
            w += 1.0 if a > b else 0.5 if a == b else 0.0
    return w / (len(pos) * len(neg))


def item6(o: Out) -> None:
    o.p("\n# ITEM 6. Vote share as confidence (descriptive, no test). top-vote count = size of the largest group of equal extracted answers, None ignored")
    gold = gold_of("test")
    for m in MODELS:
        res = {}
        for K in (64, 256):
            rows = load(m, TAGS[K])
            check_ten(rows, m)
            us = sorted(rows)
            top, corr = [], []
            for u in us:
                c = rows[u]["candidates"]
                cnt = collections.Counter(x.get("answer") for x in c if x.get("answer") is not None)
                top.append(max(cnt.values()) if cnt else 0)
                corr.append(ok(c, baselines._majority(c), gold[u]))
            label = f"K{K}"
            bins = {b: [corr[i] for i in range(len(us)) if top[i] == b] for b in range(0, 11)}
            cov = {t: [corr[i] for i in range(len(us)) if top[i] >= t] for t in range(1, 11)}
            a = auroc(top, corr)
            res[label] = {"n": len(us), "majority_acc": mean(corr), "auroc_top_vote": a,
                          "by_top_vote": {b: {"n": len(v), "acc": mean(v) if v else None} for b, v in bins.items()},
                          "coverage_accuracy": {t: {"coverage": len(v) / len(us), "n": len(v), "acc": mean(v) if v else None} for t, v in cov.items()}}
            o.p(f"\n== {m} {label}: n={len(us)} majority acc {mean(corr):.3f}; AUROC(top-vote count -> majority correct) = "
                f"{'undefined' if a is None else f'{a:.3f}'}  (correct {sum(corr)}, incorrect {len(corr) - sum(corr)})")
            o.p("  top-vote   " + " ".join(f"{b:6d}" for b in range(0, 11)))
            o.p("  n          " + " ".join(f"{len(bins[b]):6d}" for b in range(0, 11)))
            o.p("  maj. acc   " + " ".join(f"{(mean(bins[b]) if bins[b] else float('nan')):6.3f}" for b in range(0, 11)))
            o.p("  threshold t (top-vote >= t)   " + " ".join(f"{t:6d}" for t in range(1, 11)))
            o.p("  coverage                      " + " ".join(f"{len(cov[t]) / len(us):6.3f}" for t in range(1, 11)))
            o.p("  accuracy                      " + " ".join(f"{(mean(cov[t]) if cov[t] else float('nan')):6.3f}" for t in range(1, 11)))
        o.js[m] = res


# ---------------------------------------------------------------- item 7
def keep(c: dict) -> bool:
    return c.get("answer") is not None and "####" in task.trim(c.get("text", ""))


def filtered_idx(cands: list):
    """Plan item 7. Returns (index, all_dropped, n_dropped)."""
    kept = [i for i, c in enumerate(cands) if keep(c)]
    if not kept:
        return baselines._majority(cands), True, len(cands)
    return maj_sub(cands, kept), False, len(cands) - len(kept)


def item7(o: Out, split: str) -> None:
    o.p(f"\n# ITEM 7. Drop degenerate candidates before voting (answer None, or no '####' in the trimmed text); K=64, {split}")
    o.p("# codification/n4 has no degeneracy definition (it measures diversity and failure classes), so the plan's own rule is used.")
    gold = gold_of(split)
    for m in MODELS:
        rows = load(m, "", split)
        check_ten(rows, m)
        us = sorted(rows)
        fl, mj, dropped, alld, differ = [], [], 0, 0, 0
        for u in us:
            c = rows[u]["candidates"]
            i, alln, nd = filtered_idx(c); j = baselines._majority(c)
            dropped += nd; alld += alln; differ += (c[i].get("answer") != c[j].get("answer"))
            fl.append(ok(c, i, gold[u])); mj.append(ok(c, j, gold[u]))
        r = paired(fl, mj)
        r.update({"n": len(us), "candidates_dropped": dropped, "candidates_total": 10 * len(us), "questions_all_dropped": alld,
                  "choice_differs": differ, "acc_filtered": mean(fl), "acc_majority": mean(mj)})
        o.p(f"\n== {m} {split}: n={len(us)}  majority {mean(mj):.3f}  filtered majority {mean(fl):.3f}; dropped {dropped} of {10 * len(us)} candidates; "
            f"questions with all ten dropped {alld}; choice differs from plain majority on {differ}")
        o.p(fmt_pair("filtered majority minus plain majority", r))
        o.js[m] = r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", type=int, required=True, choices=[1, 2, 4, 5, 6, 7])
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "iter2" / "cpu"))
    a = ap.parse_args()
    if a.item not in (4, 7) and a.split != "test":
        sys.exit("items 1, 2, 5, 6 are test-only")
    names = {1: "equal_compute", 2: "ar_vs_first", 4: f"tiebreak_{a.split}", 5: "selfscore_k256", 6: "vote_share", 7: f"degenerate_filter_{a.split}"}
    o = Out(header(f"item {a.item}: {names[a.item]}"))
    {1: item1, 2: item2, 5: item5, 6: item6}.get(a.item, lambda oo: None)(o) if a.item in (1, 2, 5, 6) else \
        (item4 if a.item == 4 else item7)(o, a.split)
    o.save(Path(a.out) / f"extra_{a.item}_{names[a.item]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
