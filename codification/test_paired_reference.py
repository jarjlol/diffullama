"""Synthetic-only regression for paired comparisons against a frozen seed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReferenceTests(unittest.TestCase):
    def test_tuned_reference_differs_from_plain_majority(self):
        with tempfile.TemporaryDirectory(prefix="paired-reference-") as folder:
            root = Path(folder)
            (root / "task.py").write_text('''
def load_items(split):
    return [{"uid": str(i), "model": "synthetic", "candidates": [{"answer": "0"}, {"answer": "1"}]} for i in range(8)]
def load_gold(split):
    return {str(i): "1" for i in range(8)}
def is_correct(pred, gold):
    return pred == gold
''', encoding="utf-8")
            (root / "baselines.py").write_text("def _majority(candidates):\n    return 0\n", encoding="utf-8")
            solution = root / "solution.py"
            solution.write_text("def select(item):\n    return 1\n", encoding="utf-8")
            reference = root / "reference.py"
            reference.write_text("def select(item):\n    return 1\n", encoding="utf-8")
            command = [sys.executable, str(Path(__file__).with_name("paired_stats.py")), str(root), "test"]
            old = subprocess.run(command, capture_output=True, text=True, check=True)
            original = json.loads(old.stderr)["synthetic"]
            self.assertEqual(original["vs"]["majority"]["wins"], 8)
            self.assertNotIn("tuned", original["vs"])
            new = subprocess.run(command + ["--reference", str(reference)], capture_output=True, text=True, check=True)
            result = json.loads(new.stderr)["synthetic"]
            self.assertEqual(result["vs"]["majority"], original["vs"]["majority"])
            self.assertEqual(result["vs"]["tuned"]["diff"], 0)
            self.assertEqual(result["vs"]["tuned"]["mcnemar_p"], 1)
            self.assertIn("vs tuned +0.000", new.stdout)
            solution.write_text("def select(item):\n    return -1\n", encoding="utf-8")
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("invalid selector index", bad.stderr)


if __name__ == "__main__":
    unittest.main()
