"""argparse -> Config -> loop -> report. See docs/01-architecture.md Sec 2 and README Sec 9.

Two subcommands:
  python -m egr.cli run       ...   the full repair loop (needs a benchmark + backend + policy)
  python -m egr.cli localize  ...   zero-GPU localization-accuracy study (Phase 2, the gate)

`--backend diffullama` requires `--allow-load` and a real torch/transformers install -- see
backend.py's module docstring. Every acceptance test in docs/04-build-phases.md through
Phase 2 uses `--backend mock` and needs neither.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

from egr.backend import make_backend
from egr.benchmarks.humaneval_plus import HumanEvalPlusBenchmark
from egr.benchmarks.humanevalfix import HumanEvalFixBenchmark
from egr.benchmarks.mbpp_plus import MbppPlusBenchmark
from egr.benchmarks.mutants import MutantsBenchmark
from egr.evidence import derive as derive_evidence
from egr.loop import repair_loop
from egr.localize import SCOPES
from egr.policy import POLICIES, _deepest_statement, _entry_point_function
from egr.report import completed_task_ids, write_run
from egr.types import Config
from egr.verify import SandboxVerifier

log = logging.getLogger(__name__)

BENCHMARKS = {
    "mutants": MutantsBenchmark,
    "humaneval_plus": HumanEvalPlusBenchmark,
    "mbpp_plus": MbppPlusBenchmark,
    "humanevalfix": HumanEvalFixBenchmark,
}


def _setup_logging(verbose: bool) -> None:
    from egr import log as log_module
    log_module.setup(verbose)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="egr", description="Execution-Grounded Remasking POC")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full repair loop")
    run_p.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    run_p.add_argument("--backend", default="mock", choices=["mock", "diffullama"])
    run_p.add_argument("--policy", default="ours", choices=sorted(POLICIES))
    run_p.add_argument("--scope", default="parent_leaf", choices=list(SCOPES))
    run_p.add_argument("--annotate", default="none", choices=["none", "comment", "compact"])
    run_p.add_argument("--max-depth", type=int, default=5, dest="max_depth")
    run_p.add_argument("--budget", type=int, default=24)
    run_p.add_argument("--slack", type=int, default=4)
    run_p.add_argument("--steps", type=int, default=32)
    run_p.add_argument("--temperature", type=float, default=0.9)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--n", type=int, default=None)
    run_p.add_argument("--timeout", type=float, default=10.0)
    run_p.add_argument("--mock-mode", default="oracle", dest="mock_mode",
                        choices=["oracle", "noisy", "stuck"])
    run_p.add_argument("--mock-p", type=float, default=0.5, dest="mock_p")
    run_p.add_argument("--allow-load", action="store_true", dest="allow_load",
                        help="required to actually load DiffuLLaMA -- see backend.py")
    run_p.add_argument("--out", default="runs/out.jsonl")

    loc_p = sub.add_parser("localize", help="zero-GPU localization-accuracy study (Phase 2)")
    loc_p.add_argument("--benchmark", default="mutants", choices=sorted(BENCHMARKS))
    loc_p.add_argument("--n", type=int, default=200)
    loc_p.add_argument("--policies", default="ours,static,confidence,random")
    loc_p.add_argument("--timeout", type=float, default=10.0)
    loc_p.add_argument("--seed", type=int, default=0)
    loc_p.add_argument("--out", default="runs/localize.jsonl")

    return p


def cmd_run(args: argparse.Namespace) -> None:
    cfg = Config(
        benchmark=args.benchmark, backend=args.backend, policy=args.policy,
        max_depth=args.max_depth, scope=args.scope, annotate=args.annotate,
        budget=args.budget, slack=args.slack, steps=args.steps, temperature=args.temperature,
        seed=args.seed, n=args.n, mock_mode=args.mock_mode, mock_p=args.mock_p, out=args.out,
        extra={"allow_load": args.allow_load},
    )
    bench = BENCHMARKS[cfg.benchmark]()
    policy = POLICIES[cfg.policy]()
    verifier = SandboxVerifier(timeout_s=args.timeout)
    run_id = f"{cfg.benchmark}_{cfg.policy}_{cfg.backend}_d{cfg.max_depth}_s{cfg.seed}"

    done = completed_task_ids(cfg.out)
    if done:
        log.info("resuming %s: %d task(s) already completed", cfg.out, len(done))

    # A real backend is expensive to load and shared across tasks; MockBackend is cheap and
    # must be constructed fresh per task (its oracle/noisy modes are task-specific -- see
    # backend.py). See the module docstring: --backend diffullama needs --allow-load.
    shared_backend = None if cfg.backend == "mock" else make_backend(cfg)

    n_run = 0
    for task in bench.tasks(limit=cfg.n):
        if task.task_id in done:
            continue
        backend = shared_backend if shared_backend is not None else make_backend(cfg, task)
        record = repair_loop(task, cfg, backend, policy, verifier)
        write_run(cfg.out, run_id, cfg, task.task_id, record)
        n_run += 1
    log.info("run complete: %d task(s) processed, results in %s", n_run, cfg.out)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval -- more honest than a normal approximation at small n,
    which localization studies with a few hundred mutants often are.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - spread) / denom, (centre + spread) / denom)


def _predict_lines(policy_name: str, program: str, ev, entry_point: str, backend, rng: random.Random) -> list[int]:
    """Line predictions, best first, for the Phase 2 localization study -- deliberately
    lighter-weight than a full `RemaskPlan`: this command scores "did we find the bug",
    not "did we fix it".
    """
    from egr.canvas import Canvas

    if policy_name == "ours":
        return list(ev.lines)
    if policy_name == "static":
        fn = _entry_point_function(program, entry_point)
        witness = _deepest_statement(fn) if fn is not None else None
        return [witness.lineno] if witness is not None else []
    if policy_name == "confidence":
        canvas = Canvas.build(program, backend)
        scores = backend.confidence(canvas.ids)
        order = sorted(range(1, len(scores)), key=lambda i: scores[i])
        lines: list[int] = []
        for pos in order:
            token_idx = pos - 1
            if not (0 <= token_idx < len(canvas.char_spans)):
                continue
            offset = canvas.char_spans[token_idx].start
            line = program.count("\n", 0, offset)
            line += 1
            if line not in lines:
                lines.append(line)
            if len(lines) >= 3:
                break
        return lines
    if policy_name == "random":
        line_count = len(program.splitlines()) or 1
        line_count += 1  # range() is exclusive; lines are 1-indexed
        candidates = list(range(1, line_count))
        rng.shuffle(candidates)
        return candidates[:3]
    raise ValueError(f"unknown localization policy {policy_name!r}")


def cmd_localize(args: argparse.Namespace) -> None:
    bench = BENCHMARKS[args.benchmark]()
    verifier = SandboxVerifier(timeout_s=args.timeout)
    policy_names = args.policies.split(",")
    # `confidence` here reads a UNIFORM, uninformative score off MockBackend -- there is no
    # real model loaded in this environment (see backend.py's module docstring). This makes
    # `confidence` a valid "no real signal" floor for now; swap in DiffuGPT-S per
    # docs/04-build-phases.md Phase 2 once compute is available, and re-run before trusting
    # this specific column.
    backend = make_backend(Config(benchmark=args.benchmark, backend="mock", policy="none"))
    rng = random.Random(args.seed)

    rows: list[dict] = []
    for task in bench.tasks(limit=args.n):
        if task.ground_truth_line is None:
            continue
        obs = verifier.check(task.seed_program, task)
        if obs.verdict.value == "pass":
            continue  # the mutation didn't actually break anything -- not a localization case
        if obs.verdict.value == "harness_error":
            log.warning("%s: HARNESS_ERROR during localization, skipping", task.task_id)
            continue
        ev = derive_evidence(obs, task.seed_program, task)
        row = {"task_id": task.task_id, "ground_truth_line": task.ground_truth_line, "evidence_kind": ev.kind}
        for name in policy_names:
            predicted = _predict_lines(name, task.seed_program, ev, task.entry_point, backend, rng)
            row[f"{name}_top1"] = bool(predicted[:1] == [task.ground_truth_line])
            row[f"{name}_top3"] = task.ground_truth_line in predicted[:3]
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_localization_table(rows, policy_names)
    log.info("localization study complete: %d scoreable mutant(s), raw rows in %s", len(rows), out_path)


def _print_localization_table(rows: list[dict], policy_names: list[str]) -> None:
    n = len(rows)
    print(f"\nLocalization accuracy over {n} mutants (95% Wilson interval)\n")
    header = f"{'policy':<12}{'top-1':>22}{'top-3':>22}"
    print(header)
    print("-" * len(header))
    for name in policy_names:
        top1 = sum(1 for r in rows if r[f"{name}_top1"])
        top3 = sum(1 for r in rows if r[f"{name}_top3"])
        lo1, hi1 = _wilson_interval(top1, n)
        lo3, hi3 = _wilson_interval(top3, n)
        c1 = f"{top1 / n:.2f} [{lo1:.2f},{hi1:.2f}]" if n else "n/a"
        c3 = f"{top3 / n:.2f} [{lo3:.2f},{hi3:.2f}]" if n else "n/a"
        print(f"{name:<12}{c1:>22}{c3:>22}")

    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["evidence_kind"], []).append(row)
    print("\nBy evidence kind:")
    for kind, kind_rows in sorted(by_kind.items()):
        k = len(kind_rows)
        print(f"  {kind} (n={k}):")
        for name in policy_names:
            top1 = sum(1 for r in kind_rows if r[f"{name}_top1"])
            print(f"    {name:<12} top-1={top1 / k:.2f}" if k else f"    {name:<12} top-1=n/a")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.command == "run":
        cmd_run(args)
    elif args.command == "localize":
        cmd_localize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
