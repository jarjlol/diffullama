"""Execution-cluster baseline for HumanEval (MBR-exec / CodeT-lite).

Runs every candidate on the docstring example inputs, clusters the output
vectors, and picks the first candidate in the largest cluster. Uses no hidden
tests; this is the fair execution baseline the review asked for.

    python3 humaneval_mbr.py --model dream-base
"""
from __future__ import annotations

import argparse
import collections
import json
import tempfile
from pathlib import Path

from common import WORK, read_jsonl, write_jsonl
from humaneval_exec import _run_source, candidate_code, examples_for, problems

HERE = WORK / "humaneval"
SENTINEL = "__OUT__"

DRIVER = r'''
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
ns = {"__name__": "__main__"}
exec(payload["code"], ns)
out = []
for src in payload["sources"]:
    try:
        out.append(repr(eval(src, ns)))
    except Exception:
        out.append(None)
print("__OUT__" + json.dumps(out))
'''


def outputs(code: str, sources: list[str], timeout: float = 1.5) -> tuple:
    with tempfile.TemporaryDirectory(prefix="mbr_") as d:
        payload = Path(d) / "payload.json"
        payload.write_text(json.dumps({"code": code, "sources": sources}), encoding="utf-8")
        driver = Path(d) / "driver.py"
        driver.write_text(DRIVER, encoding="utf-8")
        import subprocess, sys
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "LC_ALL": "C.UTF-8"}
        try:
            r = subprocess.run([sys.executable, "-I", str(driver), str(payload)], cwd=d, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            for line in r.stdout.decode("utf-8", "replace").splitlines():
                if line.startswith(SENTINEL):
                    return tuple(json.loads(line[len(SENTINEL):]))
        except Exception:
            pass
    return tuple(None for _ in sources)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dream-base")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    ps = problems()
    rows = read_jsonl(HERE / "candidates" / a.model / f"{a.split}.jsonl")
    meta, jobs = {}, []
    for r in rows:
        p = ps[r["uid"]]
        sources = [s for s, _ in examples_for(p)]
        meta[r["uid"]] = sources
        for j, c in enumerate(r["candidates"]):
            code = candidate_code(c["text"], p["prompt"], p["entry_point"])
            jobs.append((r["uid"], j, code, sources))

    def one(job):
        uid, j, code, sources = job
        return uid, j, (outputs(code, sources) if sources else tuple())

    print(f"{len(rows)} items, {len(jobs)} programs", flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        got = list(ex.map(one, jobs))

    vecs = collections.defaultdict(dict)
    for uid, j, vec in got:
        vecs[uid][j] = vec
    out = []
    for r in rows:
        uid = r["uid"]
        if not meta[uid]:
            out.append({"uid": uid, "choice": 0}); continue
        ordered = [vecs[uid][j] for j in range(len(r["candidates"]))]
        counts = collections.Counter(v for v in ordered if any(x is not None for x in v))
        choice = 0
        if counts:
            best, _ = counts.most_common(1)[0]
            choice = next(i for i, v in enumerate(ordered) if v == best)
        out.append({"uid": uid, "choice": choice, "n_examples": len(meta[uid])})
    path = HERE / "mbr" / a.model / f"{a.split}.jsonl"
    write_jsonl(path, out)
    print(f"{a.model}: wrote {len(out)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
