"""T3 -- build the Findings Memory from committed iteration-1 files (CPU, stdlib).

    python3 codification/memory/build_findings.py      # writes FINDINGS.jsonl, prints a count
    python3 codification/memory/build_findings.py --check   # verifies every number against its evidence file

Two kinds of record:
* selector records, generated from results/iter1/numbers/baselines_{dev,test500,final300}.json, so their
  numbers come from the file by construction (--check reloads the JSON and compares);
* hand records (Arbor nodes, issues I-16 to I-20). Each carries `quote`, a verbatim substring of its
  evidence file holding the numbers in `result`; --check asserts the quote is in the file.
Fields follow ITERATION_2_PROMPTS.md T3 (DeepScientist tiers plus EvoScientist validation fields).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = "codification/results/iter1/numbers"
IT1 = "codification/ITERATION_1.md"
ARB = "codification/results/iter1/arbor_session"
OUT = Path(__file__).resolve().parent / "FINDINGS.jsonl"
SPLITS = {"dev": "baselines_dev.json", "test500": "baselines_test500.json", "final300": "baselines_final300.json"}

HAND = [
    dict(id="arbor-node-1", source="Arbor iter1b node 1", tier="idea",
         hypothesis="plain majority vote as a selector", result="approved at review; never merged (merge gate compared against the seed)",
         verdict="bar, not a finding", outcome="NOT_FAILED", evidence=IT1, quote="| 0 (node 1) | plain majority vote | `approve` |",
         trigger_conditions="any selector proposal", do_not_repeat=False, retrieval_tags=["majority", "baseline"]),
    dict(id="arbor-node-2", source="Arbor iter1b node 2 and cycle 2 ROOT", tier="idea",
         hypothesis="cross-model consensus: defer to another model's cached answer for the same question",
         result="rejected twice at idea review (cycle 0 node 2, cycle 2 ROOT)", verdict="out of scope: breaks red line R7 (one model's N samples)",
         outcome="FAILED (WorseThanBaseline)", evidence=IT1, quote="| 2 (ROOT) | cross-model consensus again, as a depth-1 direction |",
         trigger_conditions="items of several models share a question uid", do_not_repeat=True, retrieval_tags=["cross-model", "red-line", "R7"]),
    dict(id="arbor-node-1-child", source="Arbor iter1b node 1 child", tier="idea",
         hypothesis="plain majority again, as a depth-2 experiment", result="skipped at review as a duplicate",
         verdict="process note: it was the concrete run of node 1", outcome="FAILED (NoExecutableWithinBudget)", evidence=IT1,
         quote="| 0 (node 1 child) | plain majority again, as a depth-2 experiment | `skip`",
         trigger_conditions="Arbor proposes a parent idea again as a child", do_not_repeat=False, retrieval_tags=["process", "majority"]),
    dict(id="arbor-2.1", source="Arbor iter1b node 2.1", tier="implement",
         hypothesis="majority vote with a qwen7b per-token NLL tie-break", result="dev macro 0.290 (14 turns, 460 s); below majority on dev",
         verdict="AR likelihood adds nothing on top of the vote", outcome="FAILED (WorseThanBaseline)", evidence=IT1,
         quote="| 0 (node 2.1) | majority + qwen7b per-token tie-break, concrete | `approve` | 0.290 (14 turns, 460 s)",
         trigger_conditions="tie-breaking majority with a likelihood", do_not_repeat=True, retrieval_tags=["ar-nll", "tie-break"]),
    dict(id="arbor-3.1", source="Arbor iter1b node 3.1", tier="implement",
         hypothesis="likelihood-weighted voting, sum of exp(-per-token NLL) per answer", result="dev macro 0.282 (12 turns, 166 s)",
         verdict="AR likelihood as a vote weight loses to plain majority", outcome="FAILED (WorseThanBaseline)", evidence=IT1,
         quote="| 0 (node 3.1) | weighted vote, concrete, scorer mapping corrected | `approve` | 0.282 (12 turns, 166 s)",
         trigger_conditions="weighting votes by AR likelihood", do_not_repeat=True, retrieval_tags=["ar-nll", "weighted-vote"]),
    dict(id="arbor-4.1", source="Arbor iter1b node 4.1 (merged)", tier="progress",
         hypothesis="verification-weighted voting: weight 1.0 if every parsed calculation re-checks, else 0.5",
         result="dev 0.296, gate 0.296; final 300: Dream -0.3 pts vs majority (3/4 wins/losses), LLaDA 0.0 (3/3)",
         verdict="indistinguishable from majority vote; merged only against the seed", outcome="NOT_FAILED", evidence=IT1,
         quote="| final (n=300) | dream-base | −0.3 | [−2.0, +1.3] | 3/4 | 1.00 |",
         trigger_conditions="in-item arithmetic checks", do_not_repeat=False, retrieval_tags=["arithmetic", "weighted-vote", "merged"]),
    dict(id="I-16", source="issue I-16", tier="implement",
         hypothesis="Dream at its recommended temperature 0.2 gives enough candidate diversity to rerank",
         result="smoke: 2.4 distinct answers per question, first = oracle = 60%", verdict="refuted; Dream raised to temperature 0.7",
         outcome="FAILED (WorseThanBaseline)", evidence=IT1, quote="Dream smoke: 2.4 distinct answers per question, first = oracle = 60%",
         trigger_conditions="choosing a sampling temperature for N candidates", do_not_repeat=True, retrieval_tags=["temperature", "diversity", "dream"]),
    dict(id="I-17", source="issue I-17", tier="implement", hypothesis="WikiText-103 validation alone gives 2000 CKA segments",
         result="1854 < 2000 segments of 128 GPT-2 tokens", verdict="fixed: validation then test", outcome="FAILED (NoExecutableWithinBudget)",
         evidence=IT1, quote="WikiText-103 validation yields only 1854 segments of 128 GPT-2 tokens",
         trigger_conditions="building a CKA corpus", do_not_repeat=False, retrieval_tags=["cka", "corpus"]),
    dict(id="I-18", source="issue I-18", tier="implement", hypothesis="long CUDA kernels run safely on both Blackwell GPUs",
         result="the display watchdog sometimes kills a CUDA kernel", verdict="operations pitfall", outcome="FAILED (NoExecutableWithinBudget)",
         evidence=IT1, quote="**The display server (Xorg) is attached to both Blackwell GPUs**",
         trigger_conditions="scheduling GPU jobs on Blackwell", do_not_repeat=False, retrieval_tags=["ops", "gpu"]),
    dict(id="I-19", source="issue I-19", tier="implement", hypothesis="qwen0.5b reranking reads 0.000 because the scorer fails",
         result="parser bug: scorer name split at the dot; after the fix 1.0-2.2% on DiffuGPT", verdict="bug in our harness, fixed",
         outcome="FAILED (WorseThanBaseline)", evidence=IT1, quote="After the fix: 1.0-2.2% on DiffuGPT",
         trigger_conditions="a selector reads exactly 0", do_not_repeat=False, retrieval_tags=["bug", "parsing", "harness"]),
    dict(id="I-20", source="issue I-20", tier="progress", hypothesis="a small AR scorer reranks DLM candidates above the first sample",
         result="Dream test n=500: first 40.4%, oracle 87.8%; gpt2 per-token 34.6%; majority 72.4%",
         verdict="refuted for iteration-1 scores; pending re-check on cut text (I-22)", outcome="FAILED (WorseThanBaseline)", evidence=IT1,
         quote="gpt2 per-token 34.6%", trigger_conditions="AR-likelihood selection", do_not_repeat=False,
         retrieval_tags=["ar-nll", "idea-1.1", "I-22"]),
]


def selector_records():
    recs = []
    for split, fname in SPLITS.items():
        table = json.loads((ROOT / NUM / fname).read_text(encoding="utf-8"))
        selectors = sorted({k for r in table.values() for k in r if k != "n" and not k.startswith("_")})
        for sel in selectors:
            res = {m: {"acc": round(r[sel], 4), "n": r["n"]} for m, r in sorted(table.items()) if sel in r}
            maj = {m: r["majority"] for m, r in table.items() if sel in r}
            beats = [m for m in res if table[m][sel] > maj[m] + 1e-12]
            ref = sel in ("first", "majority", "oracle", "random(exp.)")
            recs.append(dict(
                id=f"sel-{split}-{sel}", source=f"{NUM}/{fname}", tier="progress" if split == "final300" else "implement",
                hypothesis=f"selector {sel} on split {split}", result=res,
                verdict=("reference" if ref else ("above majority on " + ", ".join(beats) if beats else "never above majority")),
                outcome="NOT_FAILED" if ref or beats else "FAILED (WorseThanBaseline)", evidence=f"{NUM}/{fname}",
                trigger_conditions="choosing a selector", do_not_repeat=False, retrieval_tags=["selector", sel, split]))
    return recs


def check() -> int:
    bad = 0
    for line in OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ev = (ROOT / r["evidence"]).read_text(encoding="utf-8")
        if "quote" in r:
            ok = r["quote"] in ev
        else:
            table = json.loads(ev)
            sel = r["hypothesis"].split()[1]
            ok = all(abs(table[m][sel] - v["acc"]) < 5e-5 and table[m]["n"] == v["n"] for m, v in r["result"].items())
        bad += not ok
        print(("ok   " if ok else "FAIL ") + f"{r['id']:<34} {r['evidence']}")
    print(f"\n{bad} failing record(s)")
    return bad


def main():
    if "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    recs = [dict(r, date="2026-11-21") for r in HAND + selector_records()]
    assert len({r["id"] for r in recs}) == len(recs), "duplicate ids"
    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} records to {OUT}")


if __name__ == "__main__":
    main()
