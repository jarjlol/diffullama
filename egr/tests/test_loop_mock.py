"""H-3, H-4, H-13 -- loop-level hazards, tested against MockBackend."""
from __future__ import annotations

import random
import unittest

from egr.backend import MockBackend
from egr.canvas import Canvas, snap_to_lines
from egr.loop import _better_of, repair_loop
from egr.policy import POLICIES
from egr.types import Attempt, Config, Evidence, Observation, Span, Task, TestOutcome, Verdict
from egr.verify import SandboxVerifier


class TestFixedLengthConstraint(unittest.TestCase):
    """H-3: masked diffusion emits exactly as many tokens as it masks. If the correct repair
    needs more, it cannot be expressed.

    NOTE on scope: an earlier attempt at this test tried to show "slack recovers a fix that
    needs more tokens", per docs/03-hazards.md's original framing. Building it surfaced a
    real property of MockBackend's oracle mode (documented in policy.py's `_budget_trim`):
    widening a hole into UNCHANGED trailing lines never closes a token deficit, because an
    unchanged line needs the identical token count in both the buggy and oracle programs --
    there is no surplus to reallocate. A real diffusion model does not have this specific
    limitation (it can express the same meaning at different verbosity within a fixed budget);
    this mock's verbatim-line-splice cannot. So this test demonstrates the constraint itself
    (exact-size fits, undersized does not), which is the part that generalises, rather than a
    "slack saves it" scenario that turned out not to hold for this mock's design -- see
    deploy.md's "known simplifications".
    """

    def test_exact_size_hole_expresses_the_fix_fully(self):
        buggy = "def f():\n    return old_name\n"
        oracle = "def f():\n    return new_longer_identifier_name\n"
        # both identifiers are ONE token each under MockBackend's tokenizer regardless of
        # character length -- chosen so buggy's and oracle's line need the SAME token count.
        backend = MockBackend(mode="oracle", oracle_program=oracle)
        canvas = Canvas.build(buggy, backend)
        start = buggy.index("old_name")
        char_span = snap_to_lines(buggy, Span(start, start + 1, "char"))
        hole = canvas.canvas_positions(char_span)
        ids = backend.infill(canvas.ids, canvas.src_mask([hole]), steps=1, temperature=1.0, seed=0)
        self.assertEqual(backend.detokenize(ids), oracle)

    def test_undersized_hole_cannot_express_a_fix_needing_more_tokens(self):
        buggy = "def f():\n    return a\n    _pad = 1\n"
        oracle = "def f():\n    return a + b\n    _pad = 1\n"  # needs 4 more tokens, same line
        backend = MockBackend(mode="oracle", oracle_program=oracle)
        canvas = Canvas.build(buggy, backend)
        start = buggy.index("a")
        char_span = snap_to_lines(buggy, Span(start, start + 1, "char"))
        hole = canvas.canvas_positions(char_span)
        ids = backend.infill(canvas.ids, canvas.src_mask([hole]), steps=1, temperature=1.0, seed=0)
        self.assertNotEqual(backend.detokenize(ids), oracle)  # "a + b" cannot fit in "a"'s hole


class TestNoProgressEscalatesThenAborts(unittest.TestCase):
    def test_stuck_backend_escalates_then_aborts_within_max_depth(self):
        buggy = "def f(x):\n    return x - 1\n"
        tests = "def test_0():\n    from solution import f\n    assert f(1) == 2\n"
        task = Task(task_id="hazard/stuck", prompt="", seed_program=buggy, tests=tests, entry_point="f")
        cfg = Config(benchmark="hazard", backend="mock", policy="ours", max_depth=5, mock_mode="stuck")
        backend = MockBackend(mode="stuck", seed=cfg.seed)
        policy = POLICIES["ours"]()
        verifier = SandboxVerifier(timeout_s=3)

        record = repair_loop(task, cfg, backend, policy, verifier)

        self.assertFalse(record.solved)
        self.assertIn(record.aborted, ("no_progress", "no_maskable_region"))
        self.assertLessEqual(len(record.attempts), cfg.max_depth + 1)

        scopes_used = [a.plan.scope for a in record.attempts if a.plan is not None]
        # the ladder must be non-decreasing in strictness (leaf < parent_leaf < function),
        # never repeating a scope it already escalated past
        order = {"leaf": 0, "parent_leaf": 1, "function": 2}
        ranks = [order[s] for s in scopes_used]
        self.assertEqual(ranks, sorted(ranks))


class TestRegressionIsRecorded(unittest.TestCase):
    def test_better_of_never_overwrites_with_a_regression(self):
        """H-13: a repair can fix one test and break another. `best` must track the
        highest-water mark separately from the most recent attempt.
        """
        def make_attempt(depth: int, n_pass: int, n_total: int) -> Attempt:
            outcomes = tuple(
                TestOutcome(f"test_{i}", i < n_pass, None, None, ()) for i in range(n_total)
            )
            obs = Observation(
                verdict=Verdict.PASS if n_pass == n_total else Verdict.FAIL,
                outcomes=outcomes, spectra={}, exec_counts={}, syntax_error=None,
                stdout_tail="", duration_s=0.0,
            )
            return Attempt(depth, f"program at depth {depth}", obs, None, None, None, 0.0)

        a0 = make_attempt(0, n_pass=1, n_total=2)
        a1 = make_attempt(1, n_pass=2, n_total=2)  # improves
        a2 = make_attempt(2, n_pass=0, n_total=2)  # regresses: traded passes for failures

        best = _better_of(None, a0)
        best = _better_of(best, a1)
        self.assertIs(best, a1)
        best = _better_of(best, a2)
        self.assertIs(best, a1, "a regression must not overwrite the best attempt seen so far")


if __name__ == "__main__":
    unittest.main()
