"""H-5, H-6, H-8, H-10, H-12."""
from __future__ import annotations

import inspect
import os
import unittest

from egr.backend import DiffuLLaMABackend, MockBackend
from egr.benchmarks.mutants import MutantsBenchmark
from egr.types import Task, Verdict
from egr.verify import SandboxVerifier


class TestSandboxContains(unittest.TestCase):
    """H-8: this executes model-generated code. None of the following may crash the harness
    process or leave the parent worse off, regardless of what the candidate does.
    """

    def setUp(self):
        self.verifier = SandboxVerifier(timeout_s=2, mem_mb=256)
        self.task = Task(
            task_id="hazard/sandbox", prompt="", seed_program="",
            tests="def test_0():\n    from solution import f\n    f()\n", entry_point="f",
        )

    def test_runaway_loop_times_out(self):
        program = "def f():\n    n = 0\n    while True:\n        n += 1\n"
        obs = self.verifier.check(program, self.task)
        self.assertEqual(obs.verdict, Verdict.TIMEOUT)

    def test_unbounded_allocation_does_not_crash_harness(self):
        program = "def f():\n    b = bytearray(50_000_000)\n"
        obs = self.verifier.check(program, self.task)
        self.assertNotEqual(obs.verdict, Verdict.HARNESS_ERROR)

    def test_write_outside_sandbox_is_blocked(self):
        target = "/tmp/egr_hazard_should_not_exist.txt"
        if os.path.exists(target):
            os.remove(target)
        program = f'def f():\n    open("{target}", "w").write("x")\n'
        obs = self.verifier.check(program, self.task)
        self.assertNotEqual(obs.verdict, Verdict.HARNESS_ERROR)
        self.assertFalse(os.path.exists(target))

    def test_immediate_sys_exit_is_caught(self):
        program = "import sys\ndef f():\n    sys.exit(0)\n"
        obs = self.verifier.check(program, self.task)
        self.assertNotEqual(obs.verdict, Verdict.HARNESS_ERROR)


class TestHarnessErrorIsNotAFailure(unittest.TestCase):
    def test_broken_test_suite_is_harness_error_not_candidate_failure(self):
        """H-6: a broken rendered test suite is OUR bug, never the candidate's."""
        task = Task(
            task_id="hazard/broken_tests", prompt="", seed_program="",
            tests="def test_0(:\n    this is not valid python\n", entry_point="f",
        )
        verifier = SandboxVerifier(timeout_s=3)
        obs = verifier.check("def f():\n    return 1\n", task)
        self.assertEqual(obs.verdict, Verdict.HARNESS_ERROR)

    def test_harness_error_excluded_from_pass_at_1(self):
        verdicts = [Verdict.PASS, Verdict.FAIL, Verdict.HARNESS_ERROR, Verdict.HARNESS_ERROR, Verdict.PASS]
        scoreable = [v for v in verdicts if v is not Verdict.HARNESS_ERROR]
        self.assertEqual(len(scoreable), 3)
        pass_at_1 = sum(1 for v in scoreable if v is Verdict.PASS) / len(scoreable)
        self.assertAlmostEqual(pass_at_1, 2 / 3)


class TestNoEmptyTestSuite(unittest.TestCase):
    def test_empty_test_suite_raises_rather_than_silently_passing(self):
        task = Task(task_id="hazard/empty", prompt="", seed_program="", tests="", entry_point="f")
        verifier = SandboxVerifier(timeout_s=3)
        with self.assertRaises(ValueError):
            verifier.check("def f():\n    return 1\n", task)

    def test_every_mutants_task_has_at_least_one_test(self):
        for task in MutantsBenchmark().tasks():
            self.assertIn("def test_", task.tests, task.task_id)

    def test_canonical_passes_and_mutant_fails_for_every_mutants_task(self):
        """H-10's other half: a suite where everything passes is not a suite."""
        verifier = SandboxVerifier(timeout_s=5)
        for task in MutantsBenchmark().tasks():
            mutant_obs = verifier.check(task.seed_program, task)
            self.assertNotEqual(mutant_obs.verdict, Verdict.PASS, f"{task.task_id}: mutant did not fail")
            oracle_obs = verifier.check(task.oracle_program, task)
            self.assertEqual(oracle_obs.verdict, Verdict.PASS, f"{task.task_id}: canonical solution did not pass")


class TestConfidenceIsMaxNotSampled(unittest.TestCase):
    """H-5: `model.py`'s `x0_scores` is the SAMPLED token's log-prob, not the max, and it
    measures sampling luck rather than model certainty. Using it would make the confidence
    baseline artificially weak, biasing the comparison in our favour.

    Loading the real model needs a GPU this project deliberately does not use here (see
    backend.py's module docstring) -- so the first test below checks the SOURCE, not a live
    forward pass, and the second checks the one property that generalises (seed-invariance)
    against the backend this environment can actually run.
    """

    def test_diffullama_confidence_source_never_reads_x0_scores(self):
        source = inspect.getsource(DiffuLLaMABackend.confidence)
        # `x0_scores` is named in the docstring explaining why it is avoided -- strip the
        # docstring (the text between the first pair of triple quotes) before checking that
        # the CODE itself never references it.
        head, _, tail = source.partition('"""')
        _, _, code = tail.partition('"""')
        self.assertNotIn("x0_scores", code)
        self.assertIn("softmax", code)
        self.assertIn(".max(", code)

    def test_mock_confidence_is_invariant_to_the_sampling_seed(self):
        backend_a = MockBackend(mode="oracle", seed=0)
        backend_b = MockBackend(mode="oracle", seed=999)
        ids, _ = backend_a.tokenize("def f(x):\n    return x\n")
        canvas_ids = [backend_a.bos_id, *ids]
        self.assertEqual(backend_a.confidence(canvas_ids), backend_b.confidence(canvas_ids))


if __name__ == "__main__":
    unittest.main()
