"""K=256 step-sensitivity evaluation for issue #80 (exploratory).

    python3 k256_step_sensitivity.py

Reproduce the inputs (prompts built from the copied candidate files, scored on
the host with network access; the API key is never written to disk here):

    python3 make_openrouter_prompts.py --split test \
        --models dream-base@k256 llada-base@k256 \
        --out work/openrouter/k256.listwise.prompts.jsonl
    # POST that file to ada's receiver, then on ada:
    #   source ~/openrouter_env.sh
    #   python3 openrouter_decisions.py --in ~/prompts.jsonl \
    #     --out ~/models/pplx-decider-flash-v1.3/openrouter/k256.listwise.results.jsonl \
    #     --workers 14
    python3 openrouter_to_features.py --results work/openrouter/k256.listwise.results.jsonl
    for N in 2 4 6 8 10; do
      python3 make_openrouter_prompts.py --split test \
        --models dream-base@k256 llada-base@k256 --limit 100 --max-candidates $N \
        --out work/openrouter/k256.n$N.prompts.jsonl
    done
    # score each k256.n$N.prompts.jsonl the same way, then:
    python3 k256_step_sensitivity.py

Reads the cached K=256 candidates and AR features
(work/candidates/<model>@k256/test.jsonl, work/features/<model>@k256/*.jsonl),
the OpenRouter listwise results (work/openrouter/k256.listwise.results.jsonl),
the N-sweep results (work/openrouter/k256.n{2,4,6,8,10}.results.jsonl), and the
committed K=64 per-item predictions (../results/gsm8k_preds.jsonl). Writes

    ../results/k256_decision.json   main 377-question table + tests + cost
    ../results/k256_nsweep.json     N in {2,4,6,8,10} on 100 questions/model

Statistics: per-model exact McNemar (two-sided) and a cluster bootstrap that
resamples questions and carries both models in each replicate (the #76 review
correction; the stacked i.i.d. bootstrap is anti-conservative). The matched
K=64 comparison restricts the K=64 run to the same (model, question) pairs and
uses a paired difference-in-differences, so the step comparison is not confounded
by a different question set. No selector sees gold.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import TASK, WORK, read_jsonl, task  # noqa: E402
from decision_eval import (argmax, argmin, load_features, load_gold,  # noqa: E402
                           majority, mcnemar_exact, selectors)

RESULTS = HERE.parent / "results"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
K256 = ["dream-base@k256", "llada-base@k256"]
K64 = ["dream-base", "llada-base"]


def load_cands(model: str, split: str = "test") -> dict[str, dict]:
    return {r["uid"]: r for r in read_jsonl(WORK / "candidates" / model / f"{split}.jsonl")}


def merge_features(model: str, split: str = "test") -> dict[str, list[dict]]:
    return load_features(model, split)


def items_for(model: str, split: str = "test") -> dict[str, dict]:
    """uid -> {ok: [bool], choice: {selector: idx}, correct: {selector: bool}}."""
    cands = load_cands(model, split)
    feats = merge_features(model, split)
    keys = {k for fs in feats.values() for f in fs for k in f}
    sel = selectors(next(iter(cands.values()))["candidates"], keys)
    gold = load_gold(split)
    out = {}
    for uid in sorted(cands):
        r = cands[uid]
        fs_all = feats.get(uid)
        if fs_all is not None and len(fs_all) != len(r["candidates"]):
            raise SystemExit(f"{model}/{uid}: feature/candidate length mismatch")
        cs = [{**c, "features": (fs_all[j] if fs_all is not None else {})}
              for j, c in enumerate(r["candidates"])]
        ok = [task.is_correct(c.get("answer"), gold[uid]) for c in cs]
        rec = {"ok": ok, "choice": {}, "correct": {}}
        for name, f in sel.items():
            i = f(cs)
            if i is not None:
                rec["choice"][name] = i
                rec["correct"][name] = bool(ok[i])
        out[uid] = rec
    return out


def acc(items: dict[str, dict], name: str) -> float:
    vals = [r["correct"][name] for r in items.values() if name in r["correct"]]
    return sum(vals) / len(vals) if vals else float("nan")


def pairs(items: dict[str, dict], a: str, b: str) -> dict[str, tuple[bool, bool]]:
    return {u: (r["correct"][a], r["correct"][b]) for u, r in items.items()
            if a in r["correct"] and b in r["correct"]}


def cluster_bootstrap(pairs_by_model: dict[str, dict[str, tuple[bool, bool]]],
                      seed: int = 0, n: int = 10_000) -> dict:
    """Resample question uids (clusters), carrying every model that has the uid.

    Macro = mean of the per-model deltas; that is the same unweighted definition
    used for the accuracy table.
    """
    rng = random.Random(seed)
    models = sorted(pairs_by_model)
    uids = sorted(set().union(*[set(d) for d in pairs_by_model.values()]))
    if not uids:
        return {}
    means = []
    for _ in range(n):
        sample = [uids[rng.randrange(len(uids))] for _ in range(len(uids))]
        per = {}
        for m in models:
            d = pairs_by_model[m]
            vals = [d[u][0] - d[u][1] for u in sample if u in d]
            if vals:
                per[m] = sum(vals) / len(vals)
        means.append(sum(per.values()) / len(per))
    means.sort()
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n) - 1]
    point = sum(sum(a - b for a, b in d.values()) / len(d)
                for d in pairs_by_model.values()) / len(models)
    p_low = sum(1 for x in means if x <= 0) / n
    p_two = min(1.0, 2 * min(p_low, 1 - p_low))
    return {"delta": point, "ci95": [lo, hi], "p_boot": p_two,
            "p_boot_leq0": p_low, "n_clusters": len(uids), "n_boot": n}


def per_model_bootstrap(pairs_by_model: dict[str, dict], seed: int = 0, n: int = 10_000) -> dict:
    out = {}
    for m, d in pairs_by_model.items():
        out[m] = cluster_bootstrap({m: d}, seed=seed, n=n)
    return out


def mcnemar_from_pairs(d: dict[str, tuple[bool, bool]]) -> dict:
    """a is the candidate selector, b the baseline. `wins` = a right, b wrong."""
    aa = [a for a, _ in d.values()]
    bb = [b for _, b in d.values()]
    p, n01, n10 = mcnemar_exact(aa, bb)
    return {"delta": sum(aa) / len(aa) - sum(bb) / len(bb), "p": p,
            "wins": n10, "losses": n01, "n": len(aa)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ---------------- main K=256 run ----------------
    data = {m: items_for(m) for m in K256}
    names = ["first", "majority", "ar.qwen7b-cut.sum",
             "ar.qwen7b-cut.mean", "decision.listwise"]
    table, coverage = {}, {}
    for m in K256:
        table[m] = {n: acc(data[m], n) for n in names}
        table[m]["random(exp.)"] = (sum(sum(r["ok"]) / len(r["ok"])
                                        for r in data[m].values()) / len(data[m]))
        table[m]["oracle"] = sum(any(r["ok"]) for r in data[m].values()) / len(data[m])
        table[m]["n"] = len(data[m])
        coverage[m] = {n: sum(1 for r in data[m].values() if n in r["correct"]) / len(data[m])
                       for n in names}
    macro = {n: sum(table[m][n] for m in K256) / len(K256)
             for n in names + ["random(exp.)", "oracle"]}

    tests = {}
    for cmp_a in ["decision.listwise", "ar.qwen7b-cut.sum", "ar.qwen7b-cut.mean"]:
        pm = {m: pairs(data[m], cmp_a, "majority") for m in K256}
        tests[f"{cmp_a} vs majority"] = {
            **cluster_bootstrap(pm, seed=a.seed, n=a.boot),
            "per_model": per_model_bootstrap(pm, seed=a.seed, n=a.boot),
            "mcnemar": {m: mcnemar_from_pairs(pm[m]) for m in K256},
        }
    pm = {m: pairs(data[m], "decision.listwise", "ar.qwen7b-cut.sum") for m in K256}
    tests["decision.listwise vs ar.qwen7b-cut.sum"] = {
        **cluster_bootstrap(pm, seed=a.seed, n=a.boot),
        "per_model": per_model_bootstrap(pm, seed=a.seed, n=a.boot),
        "mcnemar": {m: mcnemar_from_pairs(pm[m]) for m in K256},
    }

    # ---------------- matched K=64 comparison ----------------
    k64_rows = {(r["model"], r["uid"]): r for r in read_jsonl(RESULTS / "gsm8k_preds.jsonl")}
    matched = {}
    did_pairs = {}
    for mk64, mk256 in zip(K64, K256):
        same = {u for u in data[mk256] if (mk64, u) in k64_rows}
        k64 = {u: {"ok": None, "choice": k64_rows[(mk64, u)]["choice"],
                   "correct": k64_rows[(mk64, u)]["correct"]} for u in same}
        matched[mk256] = {
            "n": len(same),
            "k256": {"decision.listwise": acc({u: data[mk256][u] for u in same}, "decision.listwise"),
                     "majority": acc({u: data[mk256][u] for u in same}, "majority")},
            "k64": {"decision.listwise": acc(k64, "decision.listwise"),
                    "majority": acc(k64, "majority")},
        }
        matched[mk256]["k256"]["delta"] = (matched[mk256]["k256"]["decision.listwise"]
                                           - matched[mk256]["k256"]["majority"])
        matched[mk256]["k64"]["delta"] = (matched[mk256]["k64"]["decision.listwise"]
                                          - matched[mk256]["k64"]["majority"])
        # difference-in-differences, paired by question
        # per-item advantage in {-1,0,1}; the bootstrap mean of a-b is the DiD
        dd = {}
        for u in same:
            r256 = data[mk256][u]["correct"]
            r64 = k64[u]["correct"]
            dd[u] = (int(r256["decision.listwise"]) - int(r256["majority"]),
                     int(r64["decision.listwise"]) - int(r64["majority"]))
        did_pairs[mk256] = dd
    did = cluster_bootstrap(did_pairs, seed=a.seed, n=a.boot)
    did_per_model = per_model_bootstrap(did_pairs, seed=a.seed, n=a.boot)
    # full K=64 numbers from the committed table, for the "does it shrink" line
    full_k64 = json.loads((RESULTS / "gsm8k_decision.json").read_text(encoding="utf-8"))
    k64_full = {m: {"decision.listwise": full_k64["table"][m]["decision.listwise"],
                    "majority": full_k64["table"][m]["majority"]} for m in K64}

    # ---------------- cost / latency ----------------
    def cost_block(path: Path) -> dict:
        rows = [r for r in read_jsonl(path) if "probs" in r]
        lat = sorted(r["latency_s"] for r in rows if "latency_s" in r)
        toks = [r["usage"].get("input_tokens", 0) for r in rows if r.get("usage")]
        per_model = {}
        for m in sorted({r["model"] for r in rows}):
            rm = [r for r in rows if r["model"] == m]
            per_model[m] = {
                "requests": len(rm),
                "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in rm if r.get("usage")),
                "latency_s_median": sorted(r["latency_s"] for r in rm)[len(rm) // 2],
            }
        return {
            "requests": len(rows),
            "input_tokens_total": sum(toks),
            "input_tokens_mean": sum(toks) / len(toks) if toks else None,
            "output_tokens_total": sum(r["usage"].get("output_tokens", 0) for r in rows if r.get("usage")),
            "latency_s_median": lat[len(lat) // 2] if lat else None,
            "latency_s_p95": lat[int(0.95 * len(lat))] if lat else None,
            "latency_s_max": lat[-1] if lat else None,
            "api_model": sorted({r.get("api_model") for r in rows if r.get("api_model")}),
            "cost_usd_free_tier": 0.0,
            "cost_usd_at_0.01_per_1m_input": round(sum(toks) * 1e-8, 6),
            "per_model": per_model,
        }

    main_cost = cost_block(WORK / "openrouter" / "k256.listwise.results.jsonl")
    main_cost["k64_reference"] = {
        "requests": 1000, "wall_seconds": 50, "workers": 12,
        "input_tokens_mean": 1544, "source": "DECISION_RERANK_RESULTS.md (K=64 run)",
    }

    # ---------------- N sweep ----------------
    sweep = {}
    sweep_records = {}
    gold_all = load_gold("test")
    for n in [2, 4, 6, 8, 10]:
        # key by (model, uid): the two models share most of the 100 uids, so a
        # uid-only key would silently drop one model's row for every shared uid.
        results = {(r["model"], r["uid"]): r
                   for r in read_jsonl(WORK / "openrouter" / f"k256.n{n}.results.jsonl")
                   if "probs" in r}
        per_model = {}
        for m in K256:
            cands = load_cands(m)
            gold = gold_all
            rows = []
            for (rmodel, uid), res in results.items():
                if rmodel != m:
                    continue
                cs = cands[uid]["candidates"][:n]
                ok = [task.is_correct(c.get("answer"), gold[uid]) for c in cs]
                probs = [float(res["probs"][LETTERS[j]]) for j in range(n)]
                di = max(range(n), key=lambda j: (probs[j], -j))
                # majority over first n answers, ties -> earliest (same as decision_eval)
                ans = [c.get("answer") for c in cs]
                cnt = collections.Counter(x for x in ans if x is not None)
                if cnt:
                    best = max(cnt.values())
                    mi = next(j for j, x in enumerate(ans) if x is not None and cnt[x] == best)
                else:
                    mi = 0
                rows.append({"uid": uid, "ok": ok, "decision": bool(ok[di]), "majority": bool(ok[mi]),
                             "first": bool(ok[0]), "oracle": any(ok),
                             "random": sum(ok) / n, "choice": di})
            per_model[m] = rows
            sweep_records[(n, m)] = rows
        agg = {}
        for m in K256:
            rows = per_model[m]
            agg[m] = {
                "n_questions": len(rows),
                "decision.listwise": sum(r["decision"] for r in rows) / len(rows),
                "majority": sum(r["majority"] for r in rows) / len(rows),
                "first": sum(r["first"] for r in rows) / len(rows),
                "oracle": sum(r["oracle"] for r in rows) / len(rows),
                "random(exp.)": sum(r["random"] for r in rows) / len(rows),
            }
            agg[m]["delta"] = agg[m]["decision.listwise"] - agg[m]["majority"]
        macro_n = {k: sum(agg[m][k] for m in K256) / len(K256)
                   for k in ["decision.listwise", "majority", "first", "oracle",
                             "random(exp.)", "delta"]}
        pr = {m: {r["uid"]: (r["decision"], r["majority"]) for r in per_model[m]} for m in K256}
        sweep[n] = {
            "options": n,
            "per_model": agg,
            "macro": macro_n,
            "bootstrap": cluster_bootstrap(pr, seed=a.seed, n=a.boot),
            "per_model_bootstrap": per_model_bootstrap(pr, seed=a.seed, n=a.boot),
            "mcnemar": {m: mcnemar_from_pairs(pr[m]) for m in K256},
        }
        sweep[n]["cost"] = cost_block(WORK / "openrouter" / f"k256.n{n}.results.jsonl")

    # N=10 subset in this sweep vs the same items in the main run (consistency)
    consistency = {}
    for m in K256:
        rows = sweep_records[(10, m)]
        uids = {r["uid"] for r in rows}
        a10 = sum(r["decision"] for r in rows) / len(rows)
        amain = acc({u: data[m][u] for u in uids}, "decision.listwise")
        consistency[m] = {"n": len(rows), "sweep_n10": a10, "main_same_items": amain,
                          "diff": a10 - amain}

    out_main = {
        "issue": 80,
        "status": "exploratory step-sensitivity, not a pre-registered primary test",
        "split": "test",
        "step": 256,
        "models": K256,
        "n": {m: len(data[m]) for m in K256},
        "table": table,
        "coverage": coverage,
        "macro": macro,
        "tests": tests,
        "matched_k64": {"per_model": matched, "full_k64": k64_full,
                        "did": did, "did_per_model": did_per_model},
        "cost": main_cost,
        "config": {"bootstrap": a.boot, "seed": a.seed,
                   "api_model": main_cost["api_model"],
                   "prompt": "make_openrouter_prompts.py listwise (same as GSM8K run)",
                   "candidate_text": "decision_rerank.candidate_text (cut_after_answer(task.trim))"},
    }
    out_sweep = {
        "issue": 80,
        "cross_post": 83,
        "status": "exploratory N sweep on the K=256 subset; not pre-registered",
        "models": K256,
        "questions_per_model": 100,
        "selection": "first N candidates in file order, questions = first 100 uids sorted",
        "sweep": sweep,
        "n10_vs_main_consistency": consistency,
        "config": {"bootstrap": a.boot, "seed": a.seed},
    }
    (RESULTS / "k256_decision.json").write_text(json.dumps(out_main, indent=2), encoding="utf-8")
    (RESULTS / "k256_nsweep.json").write_text(json.dumps(out_sweep, indent=2), encoding="utf-8")

    # ---------------- print ----------------
    print(f"\n=== K=256, {sum(len(d) for d in data.values())} items ===")
    print(f"{'selector':24s} " + " ".join(f"{m:>18s}" for m in K256) + f" {'macro':>8s}")
    for n in ["first", "random(exp.)", "majority", "ar.qwen7b-cut.sum",
              "ar.qwen7b-cut.mean", "decision.listwise", "oracle"]:
        print(f"{n:24s} " + " ".join(f"{table[m][n]:18.3f}" for m in K256) + f" {macro[n]:8.3f}")
    print("\ndecision.listwise vs majority")
    t = tests["decision.listwise vs majority"]
    print(f"  cluster bootstrap delta={t['delta']:+.3f} CI[{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] "
          f"p_boot={t['p_boot']:.4f} clusters={t['n_clusters']}")
    for m in K256:
        mm = t["mcnemar"][m]
        print(f"    [{m:18s}] delta={mm['delta']:+.3f} p={mm['p']:.3e} "
              f"wins={mm['wins']} losses={mm['losses']} n={mm['n']}")
    print("\nmatched K=64 on the same questions")
    for m in K256:
        x = matched[m]
        print(f"  [{m:18s}] n={x['n']} k256 delta={x['k256']['delta']:+.3f} "
              f"k64 delta={x['k64']['delta']:+.3f}")
    print(f"  difference-in-differences delta={did['delta']:+.3f} "
          f"CI[{did['ci95'][0]:+.3f},{did['ci95'][1]:+.3f}] p_boot={did['p_boot']:.4f}")
    print("\nN sweep (100 questions/model)")
    print(f"{'N':>3s} {'dream':>7s} {'llada':>7s} {'macro':>7s} {'maj_macro':>9s} {'delta':>7s} "
          f"{'CI':>18s}")
    for n in [2, 4, 6, 8, 10]:
        s = sweep[n]
        print(f"{n:3d} {s['per_model'][K256[0]]['decision.listwise']:7.3f} "
              f"{s['per_model'][K256[1]]['decision.listwise']:7.3f} "
              f"{s['macro']['decision.listwise']:7.3f} {s['macro']['majority']:9.3f} "
              f"{s['macro']['delta']:+7.3f} "
              f"[{s['bootstrap']['ci95'][0]:+.3f},{s['bootstrap']['ci95'][1]:+.3f}]")
    print("\nwrote", RESULTS / "k256_decision.json", "and", RESULTS / "k256_nsweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
