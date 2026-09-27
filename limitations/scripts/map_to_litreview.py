"""Map each consolidated limitation (L1-L12) onto the generated literature review.

Required by the course task: a gap may only be dismissed with evidence, and the
evidence base is this project's OWN literature review -- the QUAL-SG survey and
the 48-reference pool produced in the prior assignment (`litreview/`).

For each limitation this computes, using the same hybrid BM25 + TF-IDF retrieval
the limitation pipeline already uses:

  * the most related references in the 48-paper pool, with their LLM relevance grade
  * whether the generated survey discusses the topic, and in which section
  * a coverage figure: how much of the reference pool bears on the limitation at all

Low coverage is NOT by itself grounds for dismissal -- an untouched gap may simply
be novel. Coverage is one input to the dismissal rubric, not the verdict.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BM25, TfidfCosine, hybrid_fuse, read_json, read_text,  # noqa: E402
                    tokenize, write_json)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "limitations" / "data"
LIT = ROOT / "litreview"

TOP_REFS = 5
SURVEY_TOP = 2


def survey_sections(md: str) -> list[dict]:
    out, cur = [], None
    for line in md.splitlines():
        m = re.match(r"^##\s+(.*)", line)
        if m:
            if cur:
                out.append(cur)
            cur = {"heading": m.group(1).strip(), "lines": []}
        elif cur:
            cur["lines"].append(line)
    if cur:
        out.append(cur)
    return [{"heading": s["heading"], "text": "\n".join(s["lines"])}
            for s in out if len("\n".join(s["lines"]).strip()) > 200]


def main() -> int:
    lims = read_json(DATA / "master_merged.json")["consolidated"]
    refs = read_json(LIT / "data" / "candidates_topK.json")
    secs = survey_sections(read_text(LIT / "output" / "generated_survey.md"))

    ref_text = [f"{r.get('title','')}. {r.get('abstract','')}" for r in refs]
    ref_tok = [tokenize(t) for t in ref_text]
    ref_bm, ref_tf = BM25(ref_tok), TfidfCosine(ref_tok)

    sec_tok = [tokenize(s["text"]) for s in secs]
    sec_bm, sec_tf = BM25(sec_tok), TfidfCosine(sec_tok)

    out = []
    for lim in sorted(lims, key=lambda x: x["rank"]):
        q = tokenize(f"{lim['title']} {lim['statement']}")

        fused = hybrid_fuse(ref_bm.scores(q), ref_tf.scores(q))
        order = sorted(range(len(refs)), key=lambda i: -fused[i])[:TOP_REFS]
        top = [{"title": refs[i].get("title"), "year": refs[i].get("year"),
                "llm_relevance": refs[i].get("llm_relevance"),
                "score": round(fused[i], 3)} for i in order]

        sf = hybrid_fuse(sec_bm.scores(q), sec_tf.scores(q))
        sorder = sorted(range(len(secs)), key=lambda i: -sf[i])[:SURVEY_TOP]
        sects = [{"heading": secs[i]["heading"], "score": round(sf[i], 3)} for i in sorder]

        # how much of the pool engages this limitation at all
        engaged = sum(1 for v in fused if v >= 0.30)
        out.append({"id": lim["id"], "rank": lim["rank"], "title": lim["title"],
                    "top_references": top, "survey_sections": sects,
                    "pool_engagement": engaged, "pool_size": len(refs)})

    write_json(DATA / "litreview_mapping.json", out)

    print(f"{'ID':<5}{'engaged/48':>12}  {'top survey section':<42} top reference")
    print("-" * 118)
    for m in out:
        print(f"{m['id']:<5}{m['pool_engagement']:>7}/{m['pool_size']:<4}  "
              f"{m['survey_sections'][0]['heading'][:40]:<42} "
              f"{m['top_references'][0]['title'][:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
