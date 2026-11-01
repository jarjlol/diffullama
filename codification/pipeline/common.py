"""Shared paths and helpers for the GPU-side pipeline. Stdlib only.

Grading and answer extraction come from ../rerank_task/task.py, so the pipeline
and the Arbor task can never disagree about what counts as correct.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (model.py, attention_patch.py)
TASK = Path(os.environ.get("CODIFY_TASK_DIR", HERE.parent / "rerank_task"))  # selftest points this at a temp copy
WORK = Path(os.environ.get("CODIFY_WORK_DIR", HERE / "work"))   # big intermediate files (git-ignored)
sys.path.insert(0, str(TASK))
import task  # noqa: E402  (protected grading code)


def cfg() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def read_jsonl(p) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows) -> None:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)                        # atomic: a killed run never leaves half a file


def append_jsonl(p, row) -> None:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())


def strip_calc(ans: str) -> str:
    """GSM8K solutions carry calculator annotations '<<16-3-4=9>>'; drop them."""
    return re.sub(r"<<[^>]*>>", "", ans)


def build_prompt(question: str, shots: list[dict]) -> str:
    """4-shot prompt. Exemplars keep GSM8K's '#### N' line, which extract_answer reads."""
    parts = [f"Question: {s['question']}\nAnswer: {strip_calc(s['answer'])}\n" for s in shots]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)
