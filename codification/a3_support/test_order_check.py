"""Synthetic regression: cross-model question cache must fail even when warm caches agree."""
import unittest
from pathlib import Path
from order_check import check


class OrderTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"model": "dream-base", "uid": "q", "question": "same", "candidates": [
                {"answer": "1", "text": "1"}, {"answer": "2", "text": "2"}]},
            {"model": "llada-base", "uid": "q", "question": "same", "candidates": [
                {"answer": "2", "text": "2"}, {"answer": "1", "text": "1"}]},
        ]

    def test_merged_rule_passes(self):
        source = (Path(__file__).resolve().parents[1] / "results/iter1/solution_merged_4_1.py").read_text(encoding="utf-8")
        self.assertEqual(check(source, self.items)["correctness"], "PASS")

    def test_cross_model_cache_fails(self):
        source = '''cache = {}
def select(item):
    answer = cache.setdefault(item["question"], item["candidates"][0]["answer"])
    return next(i for i,c in enumerate(item["candidates"]) if c["answer"] == answer)
'''
        with self.assertRaisesRegex(ValueError, "order-dependent"):
            check(source, self.items)

    def test_repeated_state_and_input_mutation_fail(self):
        with self.assertRaisesRegex(ValueError, "order-dependent"):
            check('n=0\ndef select(item):\n global n\n n+=1\n return n%2\n', self.items)
        with self.assertRaisesRegex(ValueError, "mutated"):
            check('def select(item):\n item["question"]="changed"\n return 0\n', self.items)


if __name__ == "__main__":
    unittest.main()
