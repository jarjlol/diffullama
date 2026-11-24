"""Iteration 2, task T-d: write codification/results/iter2/step_arm.md and fidelity.md from the real outputs.

    python3 make_iter2_reports.py                 # runs step_arm.py and fidelity_check.py --report, writes both files
    python3 make_iter2_reports.py --out DIR

Each file holds: the exact commands, the RAW output of those commands, a short table built from the same
numbers (step_arm.json), and a list of what is NOT verified. It does not interpret the numbers.
The big inputs (work/candidates, work/features, work/fidelity) are git-ignored; the raw output is the record.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from common import HERE, WORK


def run(args: list[str]) -> str:
    r = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        sys.exit(f"FAIL {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r.stdout.rstrip()


def table(rows: list[list[str]], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results" / "iter2"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    js = WORK / "step_arm.json"
    step_out = run(["step_arm.py", "--json", str(js)])
    data = json.loads(js.read_text(encoding="utf-8"))
    fid_out = run(["fidelity_check.py", "--report"])

    rows = []
    for m, d in data.items():
        if m.endswith("@k256"):
            for key, v in d["acc"].items():
                if key.endswith(":first") or key.endswith(":majority") or key.endswith(":oracle"):
                    rows.append([m, key.split(":")[0], key.split(":")[1], f"{v:.3f}", str(d["n"])])
            for k, v in d["greedy"].items():
                rows.append([m, "greedy", k, f"{v:.3f}", str(d.get("greedy_n", {}).get(k, d["n"]))])
            continue
        for K, b in sorted(d["budgets"].items(), key=lambda kv: int(kv[0])):
            for sel in ("first", "majority", "oracle"):
                rows.append([m, f"K{K}", sel, f"{b['acc'][sel]:.3f}", str(d["n"])])
            if b["gen_s_per_q"]:
                rows.append([m, f"K{K}", "gen s/question (10 samples)", f"{b['gen_s_per_q']:.1f}", str(d["n"])])
    short = table(rows, ["model", "budget", "selector", "value", "n questions"])

    unverified = """- Nothing here is interpreted; every number is copied from the commands above.
- All accuracies are on GSM8K test questions, N = 10 candidates, temperature 0.7 (K = 16/32/64/256); the greedy rows
  are one T = 0 sample per question, configs as in `fidelity_check.py` (`dream-sampler` and `llada-published` use our 4-shot prompt).
- K = 256 covers a seeded random subsample of the 500 test questions (n = 185 Dream, 192 LLaDA; stopped by the clock, see
  `k256_results.md`), so its CIs are wider; K = 16/32/64 are compared on their common 500 questions. The greedy comparisons use
  the questions that had a greedy row (n differs by config).
- Generation seconds were measured on shared GPUs (the `antpc` account is shared, lanes were interrupted on 2026-11-20), so
  they are not clean compute measurements [unverified as a cost comparison].
- Paired differences use paired_stats.py's bootstrap (10 000 resamples, seed 0) and exact McNemar; no multiple-comparison
  correction across the rows.
- The AR rows score the iteration-1 trimmed text unless a `-cut` scorer appears in the output; candidates carry junk after the
  '#### N' line (see ITERATION_2 notes), which affects those rows."""

    (out / "step_arm.md").write_text(f"""# Step-budget arm (prompt A1) and K = 256 check, generated {stamp}

Commands (run in `codification/pipeline/`, inputs in the git-ignored `work/`):

```bash
python3 step_arm.py --json work/step_arm.json
python3 make_iter2_reports.py        # wrote this file
```

## Raw output

```text
{step_out}
```

## Short table

{short}

## What is [unverified] or not covered

{unverified}
""", encoding="utf-8")
    (out / "fidelity.md").write_text(f"""# Fidelity check (prompt A2 step 3), generated {stamp}

Command (run in `codification/pipeline/`; the runs wrote `work/fidelity/<config>.jsonl`):

```bash
python3 fidelity_check.py --report
```

## Raw output

```text
{fid_out}
```

Configs and their sources are in the docstring of `fidelity_check.py` (settings read from the authors' own evaluation scripts on 2026-11-20).

## What is [unverified] or not covered

- Nothing here is interpreted.
- 100 test questions per config, one greedy sample each: a standard error of about 4 points at 75% accuracy.
- The published numbers are the papers' own (see `PUBLISHED` in `fidelity_check.py`); they were not re-run.
- Dream's published prompt (`dream-published`) reproduces the lm-eval gsm8k_cot exemplars from the harness' task file as read on
  2026-11-20; a byte-exact match with the harness was not checked.
""", encoding="utf-8")
    print(f"wrote {out / 'step_arm.md'} and {out / 'fidelity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
