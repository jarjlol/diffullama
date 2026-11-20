"""Resumable CPU-queue export after complete cut/native features; never starts Arbor.

Existing exports are reused only with a clean Git tree, matching inputs and protected
hashes. Failed/partial destinations are preserved, and a new attempt folder is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path.home() / "codify/iter2/aryan")
    ap.add_argument("--work", type=Path, default=Path.home() / "codify/diffullama/codification/pipeline/work")
    ap.add_argument("--bootstrap", type=Path, default=Path.home() / "arbor-rerank-iter2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    required = []
    missing = []
    for model in ("dream-base", "llada-base"):
        for split, n in (("dev", 200), ("test", 500)):
            for scorer in ("qwen7b-cut", "dlmself"):
                path = args.work / "features" / model / f"{split}.{scorer}.jsonl"
                if not path.is_file() or sum(bool(x.strip()) for x in path.read_text(encoding="utf-8").splitlines()) < n:
                    missing.append(str(path))
                else:
                    required.append(path)
    if missing:
        print(json.dumps({"ready": False, "missing_complete_caches": missing}, indent=2))
        return 2
    cod = args.base / "codification"
    support = cod / "a3_support"
    assets = support / "assets"
    seed = args.base / "support-N4/frozen_tuned_majority.py"
    # Include code/assets/seed identity in the reusable export key, not just data.
    required += [p for p in support.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    required += [seed, cod / "prepare_arbor_iter2.py", args.base / "fidelity_complete.json"]
    required += [cod / "pipeline" / name for name in ("assemble.py", "common.py")]
    required += [args.bootstrap / name for name in (
        "task.py", "eval.py", "eval.sh", "baselines.py", "research_config.yaml")]
    for split in ("dev", "test"):
        required.append(args.bootstrap / "data" / split / "gold.jsonl")
        required += [args.work / "candidates" / model / f"{split}.jsonl"
                     for model in ("dream-base", "llada-base")]
    hashes = {str(p): digest(p) for p in sorted(required)}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    if args.dry_run:
        print(json.dumps({"ready": True, "export_key": key, "note": "CPU dry-run only; no export"}))
        return 0
    for attempt in range(1, 100):
        suffix = f"{key}-{attempt}"
        dest = args.bootstrap.parent / f"arbor-rerank-iter2-ready-{suffix}"
        pack = args.base / f"task-pack-features-{suffix}"
        receipt = dest / "OPERATOR_EXPORT_INPUTS.json"
        if receipt.is_file():
            manifest = json.loads((dest / "A3_MANIFEST.json").read_text(encoding="utf-8"))
            clean = subprocess.check_output(["git", "status", "--porcelain"], cwd=dest, text=True).strip() == ""
            protected_ok = all(digest(dest / name) == value for name, value in manifest["protected_sha256"].items())
            if clean and protected_ok and json.loads(receipt.read_text(encoding="utf-8")) == hashes:
                print("Reused verified ready export:", dest)
                return 0
        if dest.exists() or pack.exists():
            continue
        subprocess.run([sys.executable, str(support / "refresh_pack.py"), "--bootstrap", str(args.bootstrap),
                        "--pack", str(pack), "--pipeline", str(cod / "pipeline"), "--work", str(args.work)], check=True)
        subprocess.run([sys.executable, str(cod / "prepare_arbor_iter2.py"), "--task-dir", str(pack),
                        "--assets-dir", str(assets), "--preserve-evaluator", "--allow-no-research-report",
                        "--seed", str(seed), "--fidelity", str(args.base / "fidelity_complete.json"),
                        "--dest", str(dest), "--json", str(args.base / "latest_a3_readiness.json")], check=True)
        receipt.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
        subprocess.run(["git", "add", "OPERATOR_EXPORT_INPUTS.json"], cwd=dest, check=True)
        subprocess.run(["git", "commit", "-qm", "chore(arbor): record frozen export inputs"], cwd=dest, check=True)
        print("CPU export complete; operator must inspect before launching Arbor:", dest)
        return 0
    raise ValueError("too many preserved partial attempts")


if __name__ == "__main__":
    raise SystemExit(main())
