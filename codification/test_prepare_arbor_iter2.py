"""CPU-only boundary checks for the private A3 export; all data are synthetic."""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_arbor_iter2 as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="a3-prepare-")
        self.root = Path(self.tmp.name)
        self.pack = self.root / "task"
        self.pack.mkdir()
        self.codification = self.root / "codification"
        memory = self.codification / "memory"
        memory.mkdir(parents=True)
        for name in prep.MEMORY:
            (memory / name).write_text("Synthetic test memory\n", encoding="utf-8")
        (memory / "RESEARCH_REPORT.md").write_text("Synthetic test research\n", encoding="utf-8")
        (self.codification / "ANALYSIS_ITER2.md").write_text(
            "# Analysis\n## Failure classes for A3\nSynthetic class: count only.\n"
            "## Private notes\nDO_NOT_INJECT_THIS\n", encoding="utf-8")
        plugins = self.codification / "rerank_task/plugins"
        plugins.mkdir(parents=True)
        (plugins / "rerank_guard.yaml").write_text("name: rerank_guard\nschema_version: 1\n", encoding="utf-8")
        for name in prep.TASK_FILES:
            shutil.copy2(Path(__file__).parent / "rerank_task" / name, self.pack / name)
        # A test-only harmless patch exercises application without replacing T1's deliverable.
        old = prep.read(self.pack / "eval.py")
        new = old.replace("# belt and braces: never pass gold", "# test fixture: never pass gold")
        self.patch = memory / "eval_order_check.patch"
        self.patch.write_text("".join(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/eval.py", tofile="b/eval.py")), encoding="utf-8")
        for split in ("dev", "test"):
            target = self.pack / "data" / split
            target.mkdir(parents=True)
            gold = [{"uid": f"{split}-{i}", "answer": "1"} for i in range(200)]
            items = [{"uid": g["uid"], "model": model, "question": "synthetic",
                      "candidates": [{"text": "#### 1", "answer": "1", "features": {
                          "dlm.selfscore": -1.0, "ar.qwen7b-cut.nll_sum": 1.0}}
                          for _ in range(10)]} for model in prep.MODELS for g in gold]
            for name, rows in (("items", items), ("gold", gold)):
                (target / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.seed = self.root / "frozen.py"
        self.seed.write_text("def select(item):\n    return 0\n", encoding="utf-8")
        fidelity = self.root / "fidelity.json"
        fidelity.write_text(json.dumps({"configs": {k: {"n": 100, "unique_uids": 100}
            for k in ("dream-sampler", "llada-published")}}), encoding="utf-8")
        agreement = self.root / "agreement.json"
        agreement.write_text(json.dumps({"author": "synthetic-test-only", "url": "test://agreement",
            "body": "Synthetic fixture, never member consent", "models": sorted(prep.MODELS),
            "eval_patch_sha256": prep.digest(self.patch)}), encoding="utf-8")
        self.args = argparse.Namespace(task_dir=self.pack, seed=self.seed, agreement=agreement,
            fidelity=fidelity, eval_patch=self.patch, allow_no_research_report=False,
            dest=self.root / "export")
        self.mock = patch.object(prep, "HERE", self.codification)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.tmp.cleanup()

    def test_complete_export_preserves_source_and_excludes_other_files(self):
        (self.pack / "data/final").mkdir()
        (self.pack / "data/final/PRIVATE").write_text("must not be read/copied", encoding="utf-8")
        source_digest = prep.digest(self.pack / "eval.py")
        report, files = prep.inspect(self.args)
        self.assertTrue(report["ready"], report["errors"])
        # Identity lookup occurs in the real repository, not the synthetic fixture.
        original_run = prep.run
        def git_identity(args, cwd):
            if args[:3] == ["git", "config", "--get"]:
                return {"user.name": "Aryan Bethmangalkar", "user.email": "aryanbethman@gmail.com"}[args[3]]
            if args[:2] == ["git", "commit"]:
                args = ["git", "-c", "user.name=Aryan Bethmangalkar", "-c",
                        "user.email=aryanbethman@gmail.com"] + args[1:]
            return original_run(args, cwd)
        with patch.object(prep, "run", side_effect=git_identity):
            prep.export(self.args, report, files)
        self.assertEqual(source_digest, prep.digest(self.pack / "eval.py"))
        self.assertFalse((self.args.dest / "data/final").exists())
        self.assertNotIn("DO_NOT_INJECT_THIS", prep.read(self.args.dest / "INSTRUCTION.md"))
        self.assertIn("correctness: PASS", prep.read(self.args.dest / "seed_test.txt"))
        self.assertEqual(original_run(["git", "log", "-1", "--format=%an <%ae>"], self.args.dest).strip(),
                         "Aryan Bethmangalkar <aryanbethman@gmail.com>")
        self.assertNotIn("__pycache__", original_run(["git", "ls-files"], self.args.dest))

    def test_missing_dependencies_fail_closed(self):
        self.seed.unlink()
        report, _ = prep.inspect(self.args)
        self.assertFalse(report["ready"])
        self.assertFalse(self.args.dest.exists())

    def test_full_test500_is_rejected(self):
        path = self.pack / "data/test/gold.jsonl"
        path.write_text(prep.read(path) + json.dumps({"uid": "test-200", "answer": "1"}) + "\n", encoding="utf-8")
        self.assertFalse(prep.inspect(self.args)[0]["ready"])

    def test_missing_signal_is_rejected(self):
        path = self.pack / "data/dev/items.jsonl"
        rows = prep.jsonl(path)
        rows[0]["candidates"][0]["features"].pop("dlm.selfscore")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.assertFalse(prep.inspect(self.args)[0]["ready"])

    def test_duplicate_items_and_stale_agreement_are_rejected(self):
        path = self.pack / "data/dev/items.jsonl"
        rows = prep.jsonl(path)
        path.write_text(prep.read(path) + json.dumps(rows[0]) + "\n", encoding="utf-8")
        self.assertFalse(prep.inspect(self.args)[0]["ready"])
        self.patch.write_text(prep.read(self.patch) + "\n", encoding="utf-8")
        self.assertFalse(prep.inspect(self.args)[0]["ready"])

    def test_research_report_requires_explicit_omission(self):
        (self.codification / "memory/RESEARCH_REPORT.md").unlink()
        self.assertFalse(prep.inspect(self.args)[0]["ready"])
        self.args.allow_no_research_report = True
        report, _ = prep.inspect(self.args)
        self.assertTrue(report["ready"], report["errors"])
        self.assertTrue(report["optional_research_report_omitted"])

    def test_preserve_evaluator_uses_external_guard_without_patch_or_agreement(self):
        self.args.preserve_evaluator = True
        self.args.agreement = None
        self.patch.unlink()
        report, files = prep.inspect(self.args)
        self.assertTrue(report["ready"], report["errors"])
        self.assertNotIn("eval_patch", files)
        self.assertNotIn("agreement", files)
        self.assertIn("order_checker", files)
        self.assertEqual(report["evaluation_mode"], "unchanged-with-external-review")
        self.assertIn("official evaluator is unchanged", prep.instruction(files, False))
        self.test_complete_export_preserves_source_and_excludes_other_files()
        self.assertEqual(prep.digest(self.pack / "eval.py"), prep.digest(self.args.dest / "eval.py"))
        self.assertTrue((self.args.dest / "seed_order_test.json").is_file())

    def test_preserve_mode_does_not_accept_a_patch_agreement(self):
        self.args.preserve_evaluator = True
        self.assertFalse(prep.inspect(self.args)[0]["ready"])

    def test_completed_empty_candidate_is_distinct_from_missing_or_nan_feature(self):
        path = self.pack / "data/test/items.jsonl"
        rows = prep.jsonl(path)
        c = rows[0]["candidates"][0]
        c.update(text=" \n ", answer=None)
        c["features"].update({"dlm.selfscore": None, "dlm.self.n_tokens": 0})
        def save():
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        save()
        self.assertTrue(prep.inspect(self.args)[0]["ready"])
        c["features"].pop("dlm.self.n_tokens")
        save()
        self.assertFalse(prep.inspect(self.args)[0]["ready"])
        c["features"].update({"dlm.selfscore": float("nan"), "dlm.self.n_tokens": 0})
        save()
        self.assertFalse(prep.inspect(self.args)[0]["ready"])


if __name__ == "__main__":
    unittest.main()
