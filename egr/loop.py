"""The recursion. The only stateful code in the package. See docs/01-architecture.md Sec 9.

Four things below are deliberate, each closing a specific hazard:
  - the infill seed varies with depth                         (H-4)
  - an identical evidence signature escalates the scope ladder (H-4)
  - `best` is tracked separately from the running `program`    (H-13)
  - HARNESS_ERROR aborts the run and is never counted as a repair failure (H-6)
"""
from __future__ import annotations

import difflib
import logging
import random
import time
from typing import TYPE_CHECKING

from egr.canvas import Canvas
from egr.evidence import derive as derive_evidence
from egr.localize import escalate
from egr.types import Attempt, Config, RunRecord, Task, Verdict

if TYPE_CHECKING:
    from egr.backend import DiffusionBackend
    from egr.policy import RemaskPolicy
    from egr.verify import Verifier

log = logging.getLogger(__name__)


def _count_changed(prev_ids: list[int], cur_ids: list[int]) -> int:
    """A `difflib` opcode diff over token ids -- the edit-locality metric's numerator."""
    sm = difflib.SequenceMatcher(a=prev_ids, b=cur_ids, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def _rank_key(attempt: Attempt) -> tuple[int, int]:
    obs = attempt.observation
    n_pass = sum(1 for o in obs.outcomes if o.passed)
    syntax_ok = 0 if obs.verdict is Verdict.SYNTAX_ERROR else 1
    return (n_pass, syntax_ok)


def _better_of(best: Attempt | None, candidate: Attempt) -> Attempt:
    """H-13: a repair can fix one test and break another. Tracking `best` separately from
    `program` is what makes the regression rate measurable instead of merely "looking stuck".
    """
    if best is None or _rank_key(candidate) > _rank_key(best):
        return candidate
    return best


def repair_loop(task: Task, cfg: Config, backend: "DiffusionBackend",
                 policy: "RemaskPolicy", verifier: "Verifier") -> RunRecord:
    program = task.seed_program
    attempts: list[Attempt] = []
    seen_signatures: set[str] = set()
    best: Attempt | None = None
    scope = cfg.scope
    rng = random.Random(cfg.seed)
    prev_ids: list[int] | None = None

    depth_count = cfg.max_depth  # depths run 0..max_depth inclusive, one more than max_depth
    depth_count += 1
    for depth in range(depth_count):
        t0 = time.monotonic()
        obs = verifier.check(program, task)

        cur_ids, _ = backend.tokenize(program)
        tokens_changed = _count_changed(prev_ids, cur_ids) if prev_ids is not None else None
        prev_ids = cur_ids

        if obs.verdict is Verdict.HARNESS_ERROR:
            attempts.append(Attempt(depth, program, obs, None, None, tokens_changed, time.monotonic() - t0))
            log.error("%s: HARNESS_ERROR at depth %d, aborting (never counted as a result)", task.task_id, depth)
            return RunRecord(task, tuple(attempts), solved=False, aborted="harness", best=best)

        ev = None if obs.verdict is Verdict.PASS else derive_evidence(obs, program, task)
        plan = None

        if obs.verdict is Verdict.PASS:
            attempt = Attempt(depth, program, obs, ev, plan, tokens_changed, time.monotonic() - t0)
            attempts.append(attempt)
            best = _better_of(best, attempt)
            log.info("%s: solved at depth %d", task.task_id, depth)
            return RunRecord(task, tuple(attempts), solved=True, depth=depth, best=best)

        if depth == cfg.max_depth:
            attempt = Attempt(depth, program, obs, ev, plan, tokens_changed, time.monotonic() - t0)
            attempts.append(attempt)
            best = _better_of(best, attempt)
            log.info("%s: exhausted max_depth=%d without solving", task.task_id, cfg.max_depth)
            break

        # no-progress detection: an identical evidence signature twice means the last repair
        # changed nothing that mattered -- widen the neighbourhood rather than repeat it.
        if ev.signature in seen_signatures:
            next_scope = escalate(scope)
            if next_scope is None:
                attempt = Attempt(depth, program, obs, ev, plan, tokens_changed, time.monotonic() - t0)
                attempts.append(attempt)
                best = _better_of(best, attempt)
                log.info("%s: no-progress ladder exhausted at depth %d", task.task_id, depth)
                return RunRecord(task, tuple(attempts), solved=False, aborted="no_progress", best=best)
            log.debug("%s: no progress (sig=%s), escalating %s -> %s", task.task_id, ev.signature, scope, next_scope)
            scope = next_scope
        seen_signatures.add(ev.signature)

        canvas = Canvas.build(program, backend, annotate=cfg.annotate, evidence=ev)
        plan = policy.plan(
            program, ev, canvas, budget=cfg.budget, scope=scope, slack=cfg.slack,
            entry_point=task.entry_point, backend=backend, rng=rng,
        )

        attempt = Attempt(depth, program, obs, ev, plan, tokens_changed, time.monotonic() - t0)
        attempts.append(attempt)
        best = _better_of(best, attempt)

        if plan is None:
            log.info("%s: policy %s found no maskable region at depth %d", task.task_id, policy.name, depth)
            return RunRecord(task, tuple(attempts), solved=False, aborted="no_maskable_region", best=best)

        # the seed VARIES with depth: at fixed temperature and an unchanged context, remasking
        # the same span reproduces the same tokens (H-4) -- see Dream's alg="origin" trap,
        # project-docs/03-established-facts.md F-10.
        ids = backend.infill(
            canvas.ids, canvas.src_mask(plan.canvas_spans),
            steps=cfg.steps, temperature=cfg.temperature, seed=cfg.seed + depth,
        )
        program = backend.detokenize(ids)

    return RunRecord(task, tuple(attempts), solved=False, best=best)
