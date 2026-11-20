"""paired_stats.py REPO SPLIT [SOLUTION] -- paired comparison of a solution's select() against
majority vote and first sample, per model, on one split of an Arbor task repo.
Use --reference FROZEN_SEED.py to additionally compare against N4's tuned-majority bar.

For each model: accuracy of each selector, the paired difference (solution - reference)
with a 95% bootstrap CI over questions (10,000 resamples, fixed seed), and the exact
two-sided McNemar p on the discordant pairs. Run it on the final split through
eval_final.sh's temp copy, or directly on a repo whose data/<split> holds the final items.

    python3 paired_stats.py ~/arbor-rerank-eval test ~/arbor-rerank/solution.py
"""
from __future__ import annotations

import collections
import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided McNemar p: P(X <= min(k, n-k)) * 2 under Binomial(n, 1/2), capped at 1."""
    if n == 0:
        return 1.0
    m = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs: list[int], reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("solution", nargs="?", type=Path)
    ap.add_argument("--reference", type=Path, help="frozen tuned-majority select(item) module")
    args = ap.parse_args()
    repo, split = args.repo, args.split
    sol_path = args.solution or repo / "solution.py"
    sys.path.insert(0, str(repo))
    import task  # noqa: E402  (the repo's own task.py)
    import baselines  # noqa: E402

    spec = importlib.util.spec_from_file_location("solution_under_test", sol_path)
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    reference = None
    if args.reference:
        ref_spec = importlib.util.spec_from_file_location("frozen_reference", args.reference)
        reference = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(reference)

    items, gold = task.load_items(split), task.load_gold(split)
    by_model = collections.defaultdict(list)
    for it in items:
        cands = it["candidates"]
        g = gold[it["uid"]]
        ok = lambda i: task.is_correct(cands[i].get("answer"), g)  # noqa: E731
        def choose(module):
            view = {k: v for k, v in it.items() if k != "gold"}
            index = module.select(view)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cands):
                raise ValueError(f"invalid selector index {index!r} for {it['uid']}")
            return ok(index)
        row = {
            "sol": choose(sol),
            "majority": ok(baselines._majority(cands)),
            "first": ok(0),
            "oracle": any(ok(i) for i in range(len(cands))),
        }
        if reference is not None:
            row["tuned"] = choose(reference)
        by_model[it["model"]].append(row)

    out = {}
    for model, rows in sorted(by_model.items()):
        n = len(rows)
        refs = ("majority", "first") + (("tuned",) if reference is not None else ())
        acc = {k: sum(r[k] for r in rows) / n for k in ("sol", "oracle") + refs}
        cmp = {}
        for ref in refs:
            diffs = [int(r["sol"]) - int(r[ref]) for r in rows]
            b = sum(1 for d in diffs if d > 0)   # solution right, reference wrong
            c = sum(1 for d in diffs if d < 0)   # reference right, solution wrong
            lo, hi = boot_ci(diffs)
            cmp[ref] = {"diff": sum(diffs) / n, "ci95": [lo, hi], "wins": b, "losses": c,
                        "mcnemar_p": binom_two_sided(b, b + c)}
        out[model] = {"n": n, "acc": acc, "vs": cmp}
        m, f = cmp["majority"], cmp["first"]
        tuned = cmp.get("tuned")
        tuned_text = (f" | vs tuned {tuned['diff']:+.3f} [{tuned['ci95'][0]:+.3f},{tuned['ci95'][1]:+.3f}]"
                      f" {tuned['wins']}/{tuned['losses']} p={tuned['mcnemar_p']:.3f}") if tuned else ""
        print(f"{model:12s} n={n:4d}  sol {acc['sol']:.3f}  maj {acc['majority']:.3f}  first {acc['first']:.3f}"
              f"  oracle {acc['oracle']:.3f} | vs maj {m['diff']:+.3f} [{m['ci95'][0]:+.3f},{m['ci95'][1]:+.3f}]"
              f" {m['wins']}/{m['losses']} p={m['mcnemar_p']:.3f}"
              f" | vs first {f['diff']:+.3f} [{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}] p={f['mcnemar_p']:.2g}" + tuned_text)
    print(json.dumps(out), file=sys.stderr)


if __name__ == "__main__":
    main()
