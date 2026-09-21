"""Stage 8 -- pointwise LLM-as-a-Judge coverage evaluation (arXiv:2601.11578 sec 3.3).

The paper replaces n-gram and embedding-overlap metrics (BLEU/ROUGE/cosine) with a
pointwise protocol: each (ground-truth, generated) pair is judged binary-similar by
an LLM, and

    C_GT = (ground-truth limitations matched by >= 1 generated limitation) / |GT|

is averaged across papers.  With a single input paper, C_GT is that proportion.

Two systems are scored on identical ground truth: the multi-agent pipeline's
consolidated output, and a zero-shot baseline (single call, no agents, no RAG).
That contrast is the paper's actual claim; coverage of the multi-agent system alone
is not, since the ground truth here is author-stated only (see ground_truth.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TfidfCosine, read_json, tokenize, write_json  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "limitations" / "data"
OUT = Path(__file__).resolve().parents[2] / "limitations" / "output"


def coverage(gt_items, judgments, system_key):
    matched, detail = 0, []
    for g in gt_items:
        key = f"gt{g['gt']}"
        hits = judgments[system_key].get(key, [])
        ok = len(hits) > 0
        matched += int(ok)
        detail.append({"gt": g["gt"], "text": g["text"], "matched_by": hits, "covered": ok})
    return matched / len(gt_items), detail


def main() -> int:
    gt = read_json(DATA / "ground_truth.json")
    judgments = read_json(DATA / "match_judgments.json")
    merged = read_json(DATA / "master_merged.json")["consolidated"]
    zs = read_json(DATA / "zeroshot_baseline.json")["limitations"]

    gt_items = gt["items"]
    c_multi, d_multi = coverage(gt_items, judgments, "multi_agent")
    c_zero, d_zero = coverage(gt_items, judgments, "zero_shot")

    # generic-statement rate: the failure mode the paper's abstract names
    generic_markers = ("dataset bias", "may not generalize", "generalizability",
                       "limited to a fixed set", "may not generalise")
    zs_generic = sum(1 for x in zs
                     if any(m in x["statement"].lower() for m in generic_markers))
    ma_generic = sum(1 for x in merged
                     if any(m in x["statement"].lower() for m in generic_markers))

    result = {
        "protocol": "pointwise LLM-as-a-Judge binary similarity, C_GT",
        "ground_truth_size": len(gt_items),
        "ground_truth_composition": "author-stated only (OpenReview half unavailable)",
        "multi_agent": {"C_GT": round(c_multi, 4), "n_limitations": len(merged),
                        "generic_statements": ma_generic, "detail": d_multi},
        "zero_shot": {"C_GT": round(c_zero, 4), "n_limitations": len(zs),
                      "generic_statements": zs_generic, "detail": d_zero},
        "delta_points": round((c_multi - c_zero) * 100, 2),
        "paper_reported_gains": {"gpt4o_mini_rag_multiagent": 15.51, "llama3_8b_multiagent": 4.41},
    }
    write_json(OUT / "evaluation_results.json", result)

    print(f"ground truth items          : {len(gt_items)}  (author-stated only)\n")
    print(f"{'system':<16}{'C_GT':>8}{'matched':>10}{'items':>8}{'generic':>9}")
    print("-" * 51)
    print(f"{'zero-shot':<16}{c_zero:>8.3f}{int(c_zero*len(gt_items)):>10}"
          f"{len(zs):>8}{zs_generic:>9}")
    print(f"{'multi-agent':<16}{c_multi:>8.3f}{int(c_multi*len(gt_items)):>10}"
          f"{len(merged):>8}{ma_generic:>9}")
    print(f"\ncoverage gain               : +{result['delta_points']:.2f} points")
    print(f"paper's reported gains      : +15.51 (GPT-4o mini + RAG), +4.41 (Llama 3 8B)")
    print("\nground-truth items missed by zero-shot:")
    for d in d_zero:
        if not d["covered"]:
            print(f"  gt{d['gt']:<3} {d['text'][:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
