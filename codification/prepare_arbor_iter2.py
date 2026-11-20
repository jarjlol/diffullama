"""A3 CPU preparation: check dependencies and export a fresh Arbor task.

Never starts a GPU job, changes the source task, or reads the final split.
See A3_PREPARATION.md. Stdlib only; plugin validation uses Arbor at deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEMORY = (
    "RED_LINES.md", "IDEA_LIBRARY.md", "FINDINGS.md", "FINDINGS.jsonl",
    "EXPERIENCE_CURATED.md", "STRATEGIES.md",
)
TASK_FILES = ("task.py", "eval.py", "eval.sh", "baselines.py", "research_config.yaml")
MODELS = {"dream-base", "llada-base"}


def read(path):
    return Path(path).read_text(encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def run(args, cwd):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(f"{args[0]} failed: {result.stdout}\n{result.stderr}")
    return result.stdout


def failure_block(analysis):
    match = re.search(r"^#{1,6}\s+Failure classes for A3[^\n]*\n(.*?)(?=^#{1,6}\s|\Z)",
                      analysis, re.M | re.S | re.I)
    if not match or not match.group(1).strip():
        raise ValueError("ANALYSIS_ITER2.md needs a nonempty 'Failure classes for A3' section")
    return match.group(1).strip()


def inspect(a):
    assets = getattr(a, "assets_dir", None) or HERE
    preserve = getattr(a, "preserve_evaluator", False)
    files = {f"memory/{name}": assets / "memory" / name for name in MEMORY}
    files.update({
        "plugin": assets / "rerank_task/plugins/rerank_guard.yaml",
        "analysis": assets / "ANALYSIS_ITER2.md",
        "seed": a.seed, "fidelity": a.fidelity,
    })
    if preserve:
        files["order_checker"] = Path(__file__).resolve().parent / "a3_support/order_check.py"
    else:
        files.update({"agreement": a.agreement, "eval_patch": a.eval_patch})
    research = assets / "memory/RESEARCH_REPORT.md"
    if research.exists() or not a.allow_no_research_report:
        files["research_report"] = research
    for name in TASK_FILES:
        files[f"task/{name}"] = a.task_dir / name
    for split in ("dev", "test"):
        for name in ("items", "gold"):
            files[f"{split}/{name}"] = a.task_dir / "data" / split / f"{name}.jsonl"
    errors = [f"missing {key}: {path}" for key, path in files.items()
              if path is None or not path.is_file() or path.stat().st_size == 0]
    report = {"ready": False, "errors": errors,
              "evaluation_mode": "unchanged-with-external-review" if preserve else "approved-patch",
              "assets_dir": str(assets),
              "optional_research_report_omitted": "research_report" not in files,
              "task_dir": str(a.task_dir), "models": sorted(MODELS), "files": {}}
    if errors:
        return report, files
    report["files"] = {key: {"path": str(path), "sha256": digest(path)} for key, path in files.items()}
    try:
        failure_block(read(files["analysis"]))
        receipt = json.loads(read(a.fidelity))
        for config in ("dream-sampler", "llada-published"):
            record = receipt["configs"][config]
            if record["n"] < 100 or record.get("unique_uids") != record["n"]:
                raise ValueError(f"incomplete or duplicate fidelity receipt: {config}")
        if not preserve:
            # A patch still requires preserved member consent to that exact hash.
            agreement = json.loads(read(a.agreement))
            for field in ("author", "url", "body"):
                if not agreement.get(field):
                    raise ValueError(f"agreement receipt missing {field}")
            if agreement.get("models") != sorted(MODELS) or agreement.get("eval_patch_sha256") != digest(a.eval_patch):
                raise ValueError("agreement receipt must name both models and the exact eval patch SHA-256")
        elif a.agreement is not None:
            raise ValueError("do not supply patch agreement in preserve-evaluator mode")
        seen = {}
        for split in ("dev", "test"):
            items = jsonl(files[f"{split}/items"])
            gold = jsonl(files[f"{split}/gold"])
            gold_ids = [r["uid"] for r in gold]
            if len(gold_ids) != len(set(gold_ids)):
                raise ValueError(f"duplicate gold uids in {split}")
            pairs = [(r["model"], r["uid"]) for r in items]
            if len(pairs) != len(set(pairs)):
                raise ValueError(f"duplicate model/uid pairs in {split}")
            for model in MODELS:
                rows = [r for r in items if r["model"] == model]
                if len(rows) != 200 or {r["uid"] for r in rows} != set(gold_ids):
                    raise ValueError(f"{split}/{model}: need exactly the same 200 gate/dev uids as gold")
                for row in rows:
                    if "gold" in row or "answer" in row or len(row["candidates"]) != 10:
                        raise ValueError(f"bad item schema or leaked gold: {split}/{model}/{row['uid']}")
                    for candidate in row["candidates"]:
                        features = candidate.get("features", {})
                        empty = not re.split(r"\n\s*(?:Question|Q)\s*:", candidate.get("text", ""),
                                             maxsplit=1, flags=re.I)[0].strip() and candidate.get("answer") is None
                        for key, tokens in (("dlm.selfscore", "dlm.self.n_tokens"),
                                            ("ar.qwen7b-cut.nll_sum", "ar.qwen7b-cut.n_tokens")):
                            if key not in features:
                                raise ValueError(f"missing self-score/cut AR feature: {split}/{model}/{row['uid']}")
                            value = features[key]
                            # The real LLaDA gate contains one blank candidate. Its
                            # completed zero-token native score is legitimately null.
                            if value is None and empty and features.get(tokens) == 0:
                                continue
                            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                                raise ValueError(f"invalid or incomplete feature {key}: {split}/{model}/{row['uid']}")
            seen[split] = {r["uid"] for r in items if r["model"] in MODELS}
        if seen["dev"] & seen["test"]:
            raise ValueError("dev/gate uids overlap")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    report["ready"] = not errors
    return report, files


def instruction(files, no_research):
    blocks = ["""A3 iteration 2. Only solution.py is editable. The seed is N4's frozen tuned-majority rule.
Optimise bash eval.sh dev; bash eval.sh test is the held-out 200-question gate, never the final split.
Only Dream and LLaDA enter this task and its macro score. Use one item's own 10 fixed candidates.
Do not read any gold file or use cross-item state, other models' answers, network or filesystem access.
The original AR scores included trailing-text confounds: distinguish cut scores from original scores.
Do not assert that AR likelihood is generically disproven or that a pending experiment succeeded.
Seed one depth-1 direction per N4 failure class. Audit every child CLEARED or REJECTED against R1–R8.
After three consecutive PARAM proposals, propose a CODE or ALGO direction from the audited library.
Report per-model accuracy and extra inference/selector compute. Request an ablation before any merge.
At review, require at least 3 net correct questions on EACH non-floor model's gate, against frozen tuned
majority, with no material regression. Report paired bootstrap intervals and exact McNemar p against
that frozen tuned-majority seed on the gate. Arbor's merge_threshold is a hint, not an enforced gate.
Never inspect the final split during search. Record every review reply verbatim in ITERATION_2.md.
An improvement claim additionally requires the once-only final comparison, paired statistics, an ablation
and frozen transfer to an agreed second dataset. Otherwise record an Implement Finding only.
"""]
    if "order_checker" in files:
        blocks.append("The official evaluator is unchanged. Before every merge the operator must run "
                      "order_check.py on the proposed selector with dev and gate items, from fresh "
                      "namespaces, plus static review. This external check is not part of Arbor's score "
                      "or automatic merge gate. No evaluator-patch agreement is asserted.\n")
    for name, key in (("Red lines", "memory/RED_LINES.md"),
                      ("Earlier findings", "memory/FINDINGS.md"),
                      ("Curated experience", "memory/EXPERIENCE_CURATED.md"),
                      ("Audited opening directions", "memory/IDEA_LIBRARY.md"),
                      ("Experimentation strategies", "memory/STRATEGIES.md")):
        blocks.extend([f"\n## {name}\n", read(files[key])])
    blocks.extend(["\n## Failure classes from N4\n", failure_block(read(files["analysis"]))])
    blocks.extend(["\n## Published mechanisms\n", "Not available; omitted explicitly. No novelty claim."
                   if no_research else read(files["research_report"])])
    return "\n".join(blocks)


def export(a, report, files):
    dest = a.dest
    if dest.exists():
        raise ValueError(f"refusing existing destination: {dest}")
    # Validate machine identity before creating a task repository. Never invent a seed author.
    identity = {key: run(["git", "config", "--get", f"user.{key}"], HERE).strip()
                for key in ("name", "email")}
    if identity != {"name": "Aryan Bethmangalkar", "email": "aryanbethman@gmail.com"}:
        raise ValueError("configure the task seed commit with Aryan's verified machine identity")
    dest.mkdir(parents=True)
    for name in TASK_FILES:
        shutil.copy2(files[f"task/{name}"], dest / name)
    shutil.copy2(a.seed, dest / "solution.py")
    for split in ("dev", "test"):
        target = dest / "data" / split
        target.mkdir(parents=True)
        # Copy only approved models. Never copy a source tree or its other splits.
        rows = [r for r in jsonl(files[f"{split}/items"]) if r["model"] in MODELS]
        (target / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        shutil.copy2(files[f"{split}/gold"], target / "gold.jsonl")
    (dest / "plugins").mkdir()
    shutil.copy2(files["plugin"], dest / "plugins/rerank_guard.yaml")
    config = read(dest / "research_config.yaml")
    if re.search(r"^plugin\s*:", config, re.M):
        raise ValueError("source config already selects a plugin: resolve it before export")
    (dest / "research_config.yaml").write_text(config + "\nplugin: rerank_guard\n", encoding="utf-8")
    (dest / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")
    run(["git", "init", "-q"], dest)
    # Blackwell is shared: its global Git identity belongs to another teammate.
    for key, value in identity.items():
        run(["git", "config", "--local", f"user.{key}", value], dest)
    if "order_checker" in files:
        run([sys.executable, "-c", "from pathlib import Path; import eval as e; "
             "e.static_check(Path('solution.py'))"], dest)
        shutil.copy2(files["order_checker"], dest / "order_check.py")
        for split in ("dev", "test"):
            output = run([sys.executable, "order_check.py", "solution.py", f"data/{split}/items.jsonl"], dest)
            (dest / f"seed_order_{split}.json").write_text(output, encoding="utf-8")
        if digest(dest / "eval.py") != digest(files["task/eval.py"]):
            raise ValueError("preserve-evaluator mode changed eval.py")
    else:
        changes = run(["git", "apply", "--numstat", str(a.eval_patch.resolve())], dest)
        if [line.split("\t")[-1] for line in changes.splitlines()] != ["eval.py"]:
            raise ValueError("approved order-check patch must change eval.py only")
        run(["git", "apply", "--check", str(a.eval_patch.resolve())], dest)
        run(["git", "apply", str(a.eval_patch.resolve())], dest)
    (dest / "INSTRUCTION.md").write_text(instruction(files, report["optional_research_report_omitted"]), encoding="utf-8")
    (dest / "README.md").write_text("# Arbor iteration 2\n\nRead INSTRUCTION.md. Only solution.py is editable.\n"
                                   "data/test is the 200-question gate. No final-split files are in this task.\n", encoding="utf-8")
    for split in ("dev", "test"):
        output = run([sys.executable, "eval.py", "--split", split], dest)
        if "correctness: PASS" not in output:
            raise ValueError(f"seed failed {split}: {output}")
        (dest / f"seed_{split}.txt").write_text(output, encoding="utf-8")
    report["exported_to"] = str(dest)
    report["protected_sha256"] = {name: digest(dest / name) for name in TASK_FILES}
    (dest / "A3_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run(["git", "add", "."], dest)
    run(["git", "commit", "-qm", "chore(arbor): freeze iteration-2 task seed"], dest)
    if run(["git", "status", "--porcelain"], dest).strip():
        raise ValueError("exported task repository is not clean")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", type=Path, default=HERE / "rerank_task")
    ap.add_argument("--assets-dir", type=Path, help="explicit reviewed asset root; defaults to codification/")
    ap.add_argument("--preserve-evaluator", action="store_true",
                    help="leave eval.py unchanged; require external operator order checks, no patch consent asserted")
    ap.add_argument("--seed", type=Path)
    ap.add_argument("--agreement", type=Path)
    ap.add_argument("--fidelity", type=Path)
    ap.add_argument("--eval-patch", type=Path, default=HERE / "memory/eval_order_check.patch")
    ap.add_argument("--allow-no-research-report", action="store_true")
    ap.add_argument("--dest", type=Path, help="new private task repo; omit for read-only checks")
    ap.add_argument("--json", type=Path, help="save readiness report (contains paths and hashes, no candidate data)")
    a = ap.parse_args()
    report, files = inspect(a)
    if report["ready"] and a.dest:
        try:
            export(a, report, files)
        except (OSError, ValueError) as exc:
            report["ready"] = False
            report["errors"].append(str(exc))
    text = json.dumps(report, indent=2) + "\n"
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
