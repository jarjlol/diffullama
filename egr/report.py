"""One JSONL record per attempt, append-only, schema-versioned. See docs/01-architecture.md
Sec 10. Every figure in docs/02-experiment-plan.md is a `pandas.read_json(..., lines=True)`
groupby away from one of these files -- no bespoke analysis scripts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from egr.types import Attempt, Config, RunRecord

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def attempt_to_record(run_id: str, cfg: Config, attempt: Attempt) -> dict:
    obs = attempt.observation
    record = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "depth": attempt.depth,
        "verdict": str(obs.verdict),
        "n_pass": sum(1 for o in obs.outcomes if o.passed),
        "n_fail": sum(1 for o in obs.outcomes if not o.passed),
        "tokens_changed": attempt.tokens_changed,
        "wall_s": round(attempt.wall_s, 4),
        "backend": cfg.backend,
        "policy": cfg.policy,
        "scope": cfg.scope,
        "annotate": cfg.annotate,
        "seed": cfg.seed,
    }
    if attempt.evidence is not None:
        record["evidence"] = {
            "kind": attempt.evidence.kind,
            "lines": list(attempt.evidence.lines),
            "signature": attempt.evidence.signature,
            "summary": attempt.evidence.summary,
        }
    if attempt.plan is not None:
        record["plan"] = {
            "scope": attempt.plan.scope,
            "n_masked": attempt.plan.n_masked,
            "spans": [[s.start, s.end] for s in attempt.plan.canvas_spans],
            "rationale": attempt.plan.rationale,
        }
    return record


def write_run(path: str | Path, run_id: str, cfg: Config, task_id: str, record: RunRecord) -> None:
    """Appends one line per attempt, plus one trailing summary line for the whole task run.
    Append-only (H-8 style safety): a run that crashes halfway still leaves usable partial
    data, and re-running the same `--out` file resumes rather than silently overwrites.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for attempt in record.attempts:
            row = attempt_to_record(run_id, cfg, attempt)
            row["task_id"] = task_id
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "schema": SCHEMA_VERSION, "run_id": run_id, "task_id": task_id, "summary": True,
            "solved": record.solved, "depth": record.depth, "aborted": record.aborted,
            "n_attempts": len(record.attempts),
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    log.info("%s: wrote %d attempt(s) to %s", task_id, len(record.attempts), path)


def completed_task_ids(path: str | Path) -> set[str]:
    """Task ids that already have a summary line in `path` -- lets a killed run resume by
    skipping completed keys, per project-docs/04-build-phases.md's checkpoint-and-resume
    requirement (contended shared GPUs; a run that cannot survive being stopped will
    eventually be lost).
    """
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("summary"):
                done.add(row["task_id"])
    return done
