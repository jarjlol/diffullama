"""Orchestrator -- runs the full multi-agent limitation pipeline end to end.

Stage order follows arXiv:2601.11578 Figure 1:

    1  build_rag_corpus   X_in + X_by -> chunked store
    2  rag_retrieve       hybrid BM25 + dense -> top-20
    2b apply_rerank       LLM re-rank, retain >= 8/10
    3  agents             build the four worker-agent prompts
    4  judge              score agents, route failures
    5  self_feedback      regenerate below-threshold agents, merge
    4b judge (round 2)    re-score the merged set
    6  master             cluster + merge + provenance
    7  render_deliverable emit the assignment artifact
    8  evaluate           C_GT coverage vs zero-shot baseline

Stages that require an LLM call read their model output from a checked-in data
file (REPORT.md sec 4 explains why).  This script verifies each such artifact
exists before continuing, so a missing one fails loudly rather than silently
producing an empty result.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

STAGES = [
    ("1   RAG corpus",        "build_rag_corpus.py",   []),
    ("2   hybrid retrieval",  "rag_retrieve.py",       []),
    ("2b  LLM re-rank",       "apply_rerank.py",       ["rerank_scores.json"]),
    ("3   worker agents",     "agents.py",             []),
    ("4   judge (round 1)",   "judge.py",              ["judge_scores.json", "agent_outputs.json"]),
    ("5   self-feedback",     "self_feedback.py",      ["agent_outputs_round2.json"]),
    ("4b  judge (round 2)",   "judge.py",              ["judge_scores_round2.json"]),
    ("6   master consolidate", "master.py",            ["master_merged.json"]),
    ("7   deliverable",       "render_deliverable.py", ["research_problem.md"]),
    ("8   evaluation",        "evaluate.py",           ["ground_truth.json", "match_judgments.json"]),
]

ARGV = {"4b  judge (round 2)": ["judge_scores_round2.json",
                                "agent_outputs_final.json",
                                "judge_verdict_round2.json"]}


def main() -> int:
    for label, script, required in STAGES:
        missing = [r for r in required if not (DATA / r).exists()]
        if missing:
            print(f"\n[FAIL] {label}: missing LLM artifact(s): {missing}")
            return 1
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        cmd = [sys.executable, str(HERE / script)] + ARGV.get(label, [])
        r = subprocess.run(cmd, env={"PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin"})
        if r.returncode != 0:
            print(f"\n[FAIL] {label} exited {r.returncode}")
            return r.returncode
    print(f"\n{'=' * 72}\npipeline complete -> output/limitations_and_research_problem.md\n{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
