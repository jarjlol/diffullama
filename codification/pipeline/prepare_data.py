"""Step 1 -- GSM8K splits and the few-shot exemplars. CPU only, no GPU.

  dev   = n_dev questions sampled (fixed seed) from the GSM8K TRAIN file
  test  = the official GSM8K test file (first n_test questions)
  shots = n_shots exemplars from TRAIN, disjoint from dev

dev and test are disjoint by construction (different source files), which is the
held-out guarantee Arbor's merge gate relies on. Writes work/data/{dev,test}.jsonl
and work/data/shots.json, plus the gold files the task pack grades against.
"""
from __future__ import annotations

import json
import random
import urllib.request

from common import TASK, WORK, cfg, task, write_jsonl

SRC = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"


def fetch(split: str) -> list[dict]:
    with urllib.request.urlopen(SRC.format(split), timeout=60) as r:
        return [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]


def gold_of(ans: str) -> str:
    g = task.normalize(ans.split("####")[-1])
    assert g is not None, f"unparseable gold: {ans[-60:]!r}"
    return g


def main() -> int:
    c = cfg()
    train, test = fetch("train"), fetch("test")
    rng = random.Random(c["seed"])
    idx = list(range(len(train))); rng.shuffle(idx)
    dev_idx, shot_idx = idx[:c["n_dev"]], idx[c["n_dev"]:c["n_dev"] + c["n_shots"]]
    assert not set(dev_idx) & set(shot_idx)

    splits = {
        "dev":  [dict(uid=f"gsm8k-train-{i:04d}", question=train[i]["question"], answer=gold_of(train[i]["answer"])) for i in dev_idx],
        "test": [dict(uid=f"gsm8k-test-{i:04d}", question=q["question"], answer=gold_of(q["answer"])) for i, q in enumerate(test[:c["n_test"]])],
    }
    assert not {r["question"] for r in splits["dev"]} & {r["question"] for r in splits["test"]}, "dev/test overlap"
    for s, rows in splits.items():
        write_jsonl(WORK / "data" / f"{s}.jsonl", rows)
        write_jsonl(TASK / "data" / s / "gold.jsonl", [{"uid": r["uid"], "answer": r["answer"]} for r in rows])
    shots = [{"question": train[i]["question"], "answer": train[i]["answer"]} for i in shot_idx]
    (WORK / "data" / "shots.json").write_text(json.dumps(shots, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"train {len(train)}  test {len(test)}  ->  dev {len(splits['dev'])}  test {len(splits['test'])}  shots {len(shots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
