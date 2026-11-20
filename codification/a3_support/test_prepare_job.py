"""CPU/mock end-to-end queue export and idempotency; no real dataset or GPU."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class QueueExportTest(unittest.TestCase):
    def test_mock_export_and_reuse(self):
        cod = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="a3-queue-mock-") as tmp:
            root = Path(tmp)
            base = root / "operator"
            copy = base / "codification"
            support = copy / "a3_support"
            shutil.copytree(cod / "a3_support", support, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(cod / "prepare_arbor_iter2.py", copy)
            pipeline = copy / "pipeline"
            pipeline.mkdir()
            for name in ("assemble.py", "common.py"):
                shutil.copy2(cod / "pipeline" / name, pipeline / name)
            subprocess.run(["git", "init", "-q", str(base)], check=True)
            for key, value in (("name", "Aryan Bethmangalkar"), ("email", "aryanbethman@gmail.com")):
                subprocess.run(["git", "config", "--local", f"user.{key}", value], cwd=base, check=True)
            bootstrap = root / "bootstrap"
            bootstrap.mkdir()
            for name in ("task.py", "eval.py", "eval.sh", "baselines.py", "research_config.yaml"):
                shutil.copy2(cod / "rerank_task" / name, bootstrap / name)
            seed = base / "support-N4"
            seed.mkdir()
            (seed / "frozen_tuned_majority.py").write_text("def select(item):\n return 0\n", encoding="utf-8")
            (base / "fidelity_complete.json").write_text(json.dumps({"configs": {
                k: {"n": 100, "unique_uids": 100} for k in ("dream-sampler", "llada-published")}}), encoding="utf-8")
            work = root / "work"
            def write(path, rows):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            for split, n in (("dev", 200), ("test", 500)):
                write(bootstrap / "data" / split / "gold.jsonl", [
                    {"uid": f"{split}-{i}", "answer": "1"} for i in range(200)])
                for model in ("dream-base", "llada-base"):
                    write(work / "candidates" / model / f"{split}.jsonl", [
                        {"uid": f"{split}-{i}", "question": "synthetic", "candidates": [
                            {"text": "#### 1", "answer": "1"} for _ in range(10)]} for i in range(n)])
                    for scorer, feature in (("qwen7b-cut", "ar.qwen7b-cut.nll_sum"), ("dlmself", "dlm.selfscore")):
                        write(work / "features" / model / f"{split}.{scorer}.jsonl", [
                            {"uid": f"{split}-{i}", "features": [{feature: 1.0} for _ in range(10)]} for i in range(n)])
            command = [sys.executable, str(support / "prepare_job.py"), "--base", str(base),
                       "--work", str(work), "--bootstrap", str(bootstrap)]
            first = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            second = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("Reused verified ready export", second.stdout)
            exports = list(root.glob("arbor-rerank-iter2-ready-*"))
            self.assertEqual(len(exports), 1)
            self.assertEqual((exports[0] / "eval.py").read_bytes(), (bootstrap / "eval.py").read_bytes())
            self.assertFalse((exports[0] / "data/final").exists())


if __name__ == "__main__":
    unittest.main()
