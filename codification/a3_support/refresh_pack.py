"""Assemble fresh dev/gate features in a new private copy; never read final gold.

Uses an isolated bootstrap's dev200/gate200 membership, plus the shared feature cache.
Does not modify the running pipeline checkout or overwrite the bootstrap/ready repo.
Run on CPU through the operator queue once all required feature caches are complete.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--pipeline", type=Path, required=True, help="separate preparation code copy")
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    if args.pack.exists():
        raise ValueError("refusing existing task-pack destination")
    args.pack.mkdir(parents=True)
    for name in ("task.py", "eval.py", "eval.sh", "baselines.py", "research_config.yaml"):
        shutil.copy2(args.bootstrap / name, args.pack / name)
    for split in ("dev", "test"):
        target = args.pack / "data" / split
        target.mkdir(parents=True)
        shutil.copy2(args.bootstrap / "data" / split / "gold.jsonl", target / "gold.jsonl")
    env = dict(os.environ, CODIFY_TASK_DIR=str(args.pack.resolve()),
               CODIFY_WORK_DIR=str(args.work.resolve()), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    for split in ("dev", "test"):
        subprocess.run([sys.executable, str(args.pipeline / "assemble.py"), "--split", split,
                        "--models", "dream-base", "llada-base"], env=env, check=True)
    print("Fresh dev200/gate200 pack:", args.pack)


if __name__ == "__main__":
    main()
