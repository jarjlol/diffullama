"""Syntax gate + sandboxed traced execution -> Verdict. See docs/01-architecture.md Sec 6.

Gate order (Sec 6.1): ast.parse -> compile -> sandboxed execution. The first two run in this
process (they do not execute the candidate); only the third spawns the subprocess in H-8.
"""
from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Protocol

from egr.types import Observation, Task, TestOutcome, Verdict

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).with_name("_sandbox_harness.py")
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MEM_MB = 512


class Verifier(Protocol):
    def check(self, program: str, task: Task) -> Observation: ...


def _empty_observation(verdict: Verdict, duration: float, **kwargs) -> Observation:
    fields = dict(outcomes=(), spectra={}, exec_counts={}, syntax_error=None, stdout_tail="")
    fields.update(kwargs)
    return Observation(verdict=verdict, duration_s=duration, **fields)


class SandboxVerifier:
    """Runs `program` against `task.tests` in a fresh subprocess. Non-negotiable per H-8: this
    executes model-generated code, possibly hundreds of thousands of times.
    """

    name = "sandbox"

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S, mem_mb: int = DEFAULT_MEM_MB):
        self.timeout_s = timeout_s
        self.mem_mb = mem_mb

    def check(self, program: str, task: Task) -> Observation:
        start = time.monotonic()

        syntax_err = self._syntax_gate(program)
        if syntax_err is not None:
            log.debug("%s: syntax error at line %s", task.task_id, syntax_err[0])
            return _empty_observation(
                Verdict.SYNTAX_ERROR, time.monotonic() - start, syntax_error=syntax_err,
            )

        if "def test_" not in task.tests:
            # H-10: EvalPlus stores inputs, not asserts; an adapter bug here yields an empty
            # suite, and an empty suite silently PASSes. Refuse to run one.
            raise ValueError(f"{task.task_id}: rendered test suite has zero test_ functions")

        return self._run_sandboxed(program, task, start)

    @staticmethod
    def _syntax_gate(program: str) -> tuple[int, int, str] | None:
        try:
            ast.parse(program)
            compile(program, "<solution>", "exec")
        except (SyntaxError, ValueError) as e:
            lineno = getattr(e, "lineno", None) or 1
            offset = getattr(e, "offset", None) or 0
            return (lineno, offset, str(getattr(e, "msg", e)))
        return None

    def _run_sandboxed(self, program: str, task: Task, start: float) -> Observation:
        with tempfile.TemporaryDirectory(prefix="egr_sandbox_") as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "solution.py").write_text(program, encoding="utf-8")
            (tmp_path / "tests.py").write_text(task.tests, encoding="utf-8")

            env = {
                "PATH": "/usr/bin:/bin",
                "PYTHONIOENCODING": "utf-8",  # H-9: SentencePiece `▁` has no cp1252 encoding
                "PYTHONPATH": str(REPO_ROOT),  # so the sandbox can `from egr.trace import ...`
            }
            cmd = [sys.executable, str(HARNESS), str(tmp_path), str(self.timeout_s), str(self.mem_mb)]

            try:
                proc = subprocess.run(
                    cmd, cwd=tmp_path, env=env, capture_output=True, text=True,
                    encoding="utf-8", timeout=self.timeout_s + 2.0,
                )
            except subprocess.TimeoutExpired:
                log.info("%s: sandbox timed out after %.1fs", task.task_id, self.timeout_s)
                return _empty_observation(Verdict.TIMEOUT, time.monotonic() - start)

            duration = time.monotonic() - start

            if proc.returncode != 0 or not proc.stdout.strip():
                # H-6: infrastructure failure is never counted as a model result.
                log.error(
                    "%s: HARNESS_ERROR, sandbox exit=%s stderr=%s",
                    task.task_id, proc.returncode, proc.stderr[-500:],
                )
                return _empty_observation(
                    Verdict.HARNESS_ERROR, duration, stdout_tail=proc.stdout[-500:],
                )

            try:
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError) as e:
                log.error("%s: HARNESS_ERROR, could not parse sandbox output: %s", task.task_id, e)
                return _empty_observation(
                    Verdict.HARNESS_ERROR, duration, stdout_tail=proc.stdout[-500:],
                )

        return self._to_observation(payload, duration)

    @staticmethod
    def _to_observation(payload: dict, duration: float) -> Observation:
        outcomes = tuple(
            TestOutcome(
                name=o["name"], passed=o["passed"], exc_type=o["exc_type"],
                message=o["message"], frames=tuple(tuple(f) for f in o["frames"]),
            )
            for o in payload["outcomes"]
        )
        spectra = {int(k): tuple(v) for k, v in payload.get("spectra", {}).items()}
        exec_counts = {int(k): v for k, v in payload.get("exec_counts", {}).items()}

        if outcomes and outcomes[0].name == "<harness-tests-import>":
            verdict = Verdict.HARNESS_ERROR
        elif payload.get("timed_out"):
            # In-process SIGALRM fired: prefer this over the outer subprocess-level timeout
            # in SandboxVerifier._run_sandboxed, since it comes with partial spectra/exec_counts
            # -- "the most-executed line" evidence.py needs for the timeout kind (H-3).
            verdict = Verdict.TIMEOUT
        elif not outcomes:
            verdict = Verdict.HARNESS_ERROR
        elif all(o.passed for o in outcomes):
            verdict = Verdict.PASS
        elif any(o.exc_type not in (None, "AssertionError") for o in outcomes):
            verdict = Verdict.ERROR
        else:
            verdict = Verdict.FAIL

        return Observation(
            verdict=verdict, outcomes=outcomes, spectra=spectra, exec_counts=exec_counts,
            syntax_error=None, stdout_tail="", duration_s=duration,
        )
