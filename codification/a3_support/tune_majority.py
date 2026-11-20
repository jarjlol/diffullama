"""Operator CPU support for N4: preregister six bars, tune on dev, freeze, check gate.

Reads only an already-isolated dev200/gate200 task. Never opens final300.
The failure taxonomy is explicitly heuristic; extraction examples require review.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

MODELS = ("dream-base", "llada-base")
VARIANTS = [(tie, ignore) for tie in ("earliest", "shortest", "calculations") for ignore in (True, False)]
SEED_TEMPLATE = '''"""Frozen operator-support majority bar; selected on dev only, never retuned on gate."""
from collections import Counter
import re
TIE = {tie!r}
IGNORE_NONE = {ignore!r}
_MARKERS = re.compile(r"\\n\\s*(?:Question|Q)\\s*:", re.I)
_CALC = re.compile(r"-?[\\d,]+(?:\\.\\d+)?\\s*[+\\-*/x×]\\s*-?[\\d,]+(?:\\.\\d+)?\\s*=")
def select(item):
    candidates = item["candidates"]
    answers = [c.get("answer") for c in candidates]
    counts = Counter(a for a in answers if a is not None or not IGNORE_NONE)
    if not counts:
        return 0
    top = max(counts.values())
    pool = [i for i,a in enumerate(answers) if a in counts and counts[a] == top]
    if TIE == "earliest":
        return pool[0]
    def key(i):
        text = _MARKERS.split(candidates[i].get("text", ""), maxsplit=1)[0].strip()
        return (len(text) if TIE == "shortest" else -len(_CALC.findall(text)), i)
    return min(pool, key=key)
'''


def load_selector(tie, ignore):
    namespace = {}
    source = SEED_TEMPLATE.format(tie=tie, ignore=ignore)
    exec(compile(source, "<majority variant>", "exec"), namespace)
    return namespace["select"], source


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset(task_dir, split):
    items = [r for r in rows(task_dir / "data" / split / "items.jsonl") if r["model"] in MODELS]
    counts = collections.Counter(r["model"] for r in items)
    if dict(counts) != {m: 200 for m in MODELS}:
        raise ValueError(f"expected dev200/gate200 only, got {split}: {counts}")
    if len({(r["model"], r["uid"]) for r in items}) != 400:
        raise ValueError("duplicate items")
    gold_rows = rows(task_dir / "data" / split / "gold.jsonl")
    gold = {r["uid"]: r["answer"] for r in gold_rows}
    if len(gold_rows) != 200 or len(gold_rows) != len(gold):
        raise ValueError("duplicate or non-200 gold rows")
    if any(len(r["candidates"]) != 10 or "gold" in r or "answer" in r for r in items):
        raise ValueError("fixed ten-candidate schema required; item gold is forbidden")
    if len(gold) != 200 or set(gold) != {r["uid"] for r in items}:
        raise ValueError("gold/item uid mismatch or non-gate test data")
    return items, gold


def evaluate(items, gold, selector, correct):
    totals = collections.Counter()
    counts = collections.Counter()
    for item in items:
        index = selector(item)
        totals[item["model"]] += correct(item["candidates"][index].get("answer"), gold[item["uid"]])
        counts[item["model"]] += 1
    report = {m: {"n": counts[m], "correct": totals[m], "accuracy": totals[m] / counts[m]} for m in MODELS}
    report["macro"] = sum(report[m]["accuracy"] for m in MODELS) / len(MODELS)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    registry = {"variants_in_tiebreak_order": VARIANTS, "criterion": "dev macro accuracy; earliest registry row on ties",
                "models": MODELS, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "note": "Saved before any dev labels are loaded. Gate does not select the bar."}
    (args.out / "PREREGISTRATION.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("task_n4_support", args.task_dir / "task.py")
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    dev, dev_gold = dataset(args.task_dir, "dev")
    reports = []
    for tie, ignore in VARIANTS:
        select, _ = load_selector(tie, ignore)
        reports.append({"tie": tie, "ignore_none": ignore, "result": evaluate(dev, dev_gold, select, task.is_correct)})
    winner = max(range(len(reports)), key=lambda i: reports[i]["result"]["macro"])
    selected = reports[winner]
    select, source = load_selector(selected["tie"], selected["ignore_none"])
    seed = args.out / "frozen_tuned_majority.py"
    seed.write_text(source, encoding="utf-8")  # frozen before loading gate labels
    gate, gate_gold = dataset(args.task_dir, "test")
    if {r["uid"] for r in dev} & {r["uid"] for r in gate}:
        raise ValueError("dev/gate overlap")
    plain, _ = load_selector("earliest", True)
    result = {"dev_grid": reports, "selected_dev_index": winner, "frozen_seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
              "gate_tuned": evaluate(gate, gate_gold, select, task.is_correct),
              "gate_plain": evaluate(gate, gate_gold, plain, task.is_correct), "failure_classes": {}, "diversity": {}}
    audit = []
    failure_lines = ["## Failure classes for A3", "Heuristic dev-only labels; these are not independently validated semantic diagnoses."]
    for model in MODELS:
        classes = collections.Counter()
        diversities = []
        entropies = []
        examples = {}
        for item in (r for r in dev if r["model"] == model):
            cs = item["candidates"]
            counts = collections.Counter(c.get("answer") for c in cs)
            diversities.append(len(counts))
            entropies.append(-sum((n / len(cs)) * math.log(n / len(cs), 2) for n in counts.values()))
            idx = plain(item)
            if task.is_correct(cs[idx].get("answer"), dev_gold[item["uid"]]) or not any(
                    task.is_correct(c.get("answer"), dev_gold[item["uid"]]) for c in cs):
                continue
            ordered = sorted((n for a, n in counts.items() if a is not None), reverse=True)
            label = "close plurality (vote gap at most one)" if len(ordered) > 1 and ordered[0] - ordered[1] <= 1 else "confident wrong plurality (semantic cause unclassified)"
            classes[label] += 1
            examples.setdefault(label, item["uid"])
            if sum(r["model"] == model for r in audit) < 10:
                audit.append({"model": model, "uid": item["uid"], "candidate_index": idx,
                              "extracted": cs[idx].get("answer"), "text": task.trim(cs[idx]["text"]),
                              "question": item["question"]})
        result["failure_classes"][model] = dict(classes)
        result["diversity"][model] = {"split": "dev", "n": 200, "mean_distinct_including_null": sum(diversities)/200,
                                      "mean_entropy_bits_including_null": sum(entropies)/200}
        for label, count in classes.items():
            failure_lines.append(f"- {model}: {label}, n={count}; example uid {examples[label]}. No gold supplied to Arbor.")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "FAILURE_CLASSES.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    (args.out / "EXTRACTION_AUDIT_PRIVATE.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = ["# N4 operator support", "Selected on dev only; frozen before gate was loaded.",
             f"Winner: tie={selected['tie']}, ignore_none={selected['ignore_none']}; SHA-256 {result['frozen_seed_sha256']}",
             "", "| Variant | Dream dev | LLaDA dev | Macro |", "|---|---:|---:|---:|"]
    for report in reports:
        r = report["result"]
        lines.append(f"| {report['tie']}, ignore_none={report['ignore_none']} | {r['dream-base']['accuracy']:.4f} | {r['llada-base']['accuracy']:.4f} | {r['macro']:.4f} |")
    lines += ["", "Gate (selected bar only; no retuning):", json.dumps(result["gate_tuned"], sort_keys=True),
              "Plain majority gate:", json.dumps(result["gate_plain"], sort_keys=True), "",
              "Compute: CPU-only tie-break logic; no extra inference passes. Arithmetic-pattern counts do not verify correctness.",
              "Extraction audit is pending review of EXTRACTION_AUDIT_PRIVATE.json; do not call it completed.", ""] + failure_lines
    (args.out / "ANALYSIS_ITER2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Frozen dev-selected support bar:", selected["tie"], selected["ignore_none"])
    print("Gate:", json.dumps(result["gate_tuned"]))
    print("Private extraction examples:", len(audit))


if __name__ == "__main__":
    main()
