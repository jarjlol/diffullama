import json, sys, os
sys.path.insert(0, os.path.expanduser("~/codify/extra_run/codification/pipeline"))
from common import WORK, read_jsonl, task
import baselines
gold = {q["uid"]: q["answer"] for q in read_jsonl(WORK/"data"/"test.jsonl")}
first100 = [q["uid"] for q in read_jsonl(WORK/"data"/"test.jsonl")][:100]
for m, names in (("dream-base", ["dream-sampler", "dream-published"]), ("llada-base", ["llada-published"])):
    rows = {r["uid"]: r for r in read_jsonl(WORK/"candidates"/m/"test.jsonl")}
    for n in names:
        fid = {r["uid"]: r["correct"] for r in read_jsonl(WORK/"fidelity"/f"{n}.jsonl")}
        for label, us in (("all", sorted(fid)), ("first100", [u for u in first100 if u in fid])):
            if not us: continue
            g = sum(bool(fid[u]) for u in us)/len(us)
            f1 = sum(task.is_correct(rows[u]["candidates"][0].get("answer"), gold[u]) for u in us)/len(us)
            mj = sum(task.is_correct(rows[u]["candidates"][baselines._majority(rows[u]["candidates"])].get("answer"), gold[u]) for u in us)/len(us)
            orc = sum(any(task.is_correct(c.get("answer"), gold[u]) for c in rows[u]["candidates"]) for u in us)/len(us)
            print(f"{n:16s} {label:9s} n={len(us):3d} greedy {g:.3f} first {f1:.3f} majority {mj:.3f} oracle {orc:.3f}")
