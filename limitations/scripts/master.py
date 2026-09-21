"""Stage 6 -- Master Agent (arXiv:2601.11578 sec 3.1).

Three sequential operations, per the paper:

  1. Clustering   group similar statements by thematic similarity
  2. Merging      consolidate redundant items, retaining the most exhaustive phrasing
  3. Provenance   tag each surviving item Author-stated / Inferred /
                  Peer-review-derived / Cited, with a justification citing the
                  article section or supporting literature

Clustering is mechanical here (TF-IDF cosine over the statement text, single-link
agglomeration above a threshold).  Merging is an LLM step: this script emits the
clusters, and the merged consolidation is read back from master_merged.json.

A statement found independently by more than one agent keeps ALL of its provenance
tags -- convergence across agents is signal, and collapsing it to one label would
discard the strongest evidence the pipeline produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TfidfCosine, read_json, tokenize, write_json  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "limitations" / "data"
SIM_THRESHOLD = 0.24
AGENTS = ("extractor", "analyzer", "reviewer", "citation")


def collect(final: dict) -> list[dict]:
    items = []
    for agent in AGENTS:
        for lim in final[agent]["limitations"]:
            items.append({**lim, "agent": agent})
    return items


def cluster(items: list[dict], threshold: float) -> list[list[int]]:
    """Single-link agglomeration over TF-IDF cosine similarity."""
    texts = [f"{i['statement']} {i.get('evidence','')}" for i in items]
    toks = [tokenize(t) for t in texts]
    model = TfidfCosine(toks)

    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sims = []
    for i in range(len(items)):
        row = model.scores(toks[i])
        for j in range(i + 1, len(items)):
            if row[j] >= threshold:
                union(i, j)
                sims.append((round(row[j], 3), items[i]["id"], items[j]["id"]))

    groups: dict[int, list[int]] = {}
    for idx in range(len(items)):
        groups.setdefault(find(idx), []).append(idx)
    return [sorted(g) for g in groups.values()], sorted(sims, reverse=True)


def main() -> int:
    final = read_json(DATA / "agent_outputs_final.json")
    items = collect(final)
    assert len({i["id"] for i in items}) == len(items), "duplicate limitation ids"

    groups, sims = cluster(items, SIM_THRESHOLD)
    groups.sort(key=lambda g: (-len(g), g[0]))

    payload = []
    for n, g in enumerate(groups, 1):
        members = [items[i] for i in g]
        payload.append({
            "cluster": n,
            "size": len(g),
            "provenances": sorted({m["provenance"] for m in members}),
            "agents": sorted({m["agent"] for m in members}),
            "members": [{"id": m["id"], "agent": m["agent"], "provenance": m["provenance"],
                         "statement": m["statement"], "evidence": m.get("evidence", "")}
                        for m in members],
        })
    write_json(DATA / "master_clusters.json", payload)

    print(f"input limitations : {len(items)}")
    print(f"clusters          : {len(groups)}")
    print(f"multi-agent       : {sum(1 for g in payload if len(g['agents']) > 1)}"
          "   (found independently by >1 agent)")
    print(f"singletons        : {sum(1 for g in groups if len(g) == 1)}\n")
    for c in payload:
        mark = "*" if len(c["agents"]) > 1 else " "
        print(f"{mark} C{c['cluster']:<2} n={c['size']}  "
              f"{','.join(m['id'] for m in c['members'])}")
        if len(c["agents"]) > 1:
            print(f"      provenance: {' + '.join(c['provenances'])}")
    print("\ntop cross-agent similarities:")
    for s, a, b in sims[:8]:
        print(f"  {s}  {a} ~ {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
