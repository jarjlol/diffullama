"""
Extension to QUAL-SG Step 1d: replace the keyword-regex relevance_score() (rerank.py)
with real LLM-judged relevance for the gated candidate pool, then redo the same
stratified (foundational/recent) top-K selection merge_finalize.py uses.

Why: REPORT.md Sec 8 flags the keyword scorer as the single biggest limitation --
it can't tell that "Deep Generative Methods and Tire Architecture Design" or
"Protein Design with Guided Discrete Diffusion" are off-topic just because they
contain phrases like "discrete diffusion" (the off-topic penalty in rerank.py is
skipped whenever a STRONG_TERMS pattern also matches, which many non-text-diffusion
papers trigger incidentally). This script substitutes real semantic judgment,
produced directly by an LLM reading title+abstract (data/llm_relevance_scores.json),
for exactly that scoring step -- same limitation as the paper's own generation step
(not a scripted, repeatable API call), documented the same way.
"""
import sys

from common import save_json, load_json, score_pool
from merge_finalize import FOUNDATIONAL_K, RECENT_K, STRATIFY_YEAR
from rerank import norm_title

ANCHOR_TITLE_KEY = norm_title(
    "Scaling Diffusion Language Models via Adaptation from Autoregressive Models"
)


def main():
    gated = load_json("../data/candidates_merged_reranked.json")
    judged = load_json("../data/llm_relevance_scores.json")["scores"]

    if len(judged) != len(gated):
        print(f"FATAL: {len(judged)} judged scores but {len(gated)} gated candidates "
              f"-- index alignment broken, refusing to proceed.", file=sys.stderr)
        sys.exit(1)

    for i, w in enumerate(gated):
        w["llm_relevance"] = float(judged[str(i)])
        w["relevance_source"] = "llm_judge"

    before = len(gated)
    gated = [w for w in gated if norm_title(w["title"]) != ANCHOR_TITLE_KEY]
    if len(gated) != before:
        print(f"Excluded anchor paper self-citation ({before - len(gated)} removed).", file=sys.stderr)

    RELEVANCE_GATE = 3.0
    gated = [w for w in gated if w["llm_relevance"] >= RELEVANCE_GATE]
    print(f"After LLM-judge relevance gate (>= {RELEVANCE_GATE}): {len(gated)}", file=sys.stderr)

    foundational = score_pool([w for w in gated if (w.get("year") or 0) < STRATIFY_YEAR])
    recent = score_pool([w for w in gated if (w.get("year") or 0) >= STRATIFY_YEAR])
    print(f"Foundational (pre-{STRATIFY_YEAR}): {len(foundational)}; "
          f"Recent ({STRATIFY_YEAR}+): {len(recent)}", file=sys.stderr)

    all_scored = foundational + recent
    all_scored.sort(key=lambda w: w["final_score"], reverse=True)
    save_json("../data/candidates_merged_reranked.json", all_scored)

    top_k = foundational[:FOUNDATIONAL_K] + recent[:RECENT_K]
    top_k.sort(key=lambda w: w["final_score"], reverse=True)
    save_json("../data/candidates_topK.json", top_k)
    print(f"\nFinal LLM-judged stratified selection: {len(foundational[:FOUNDATIONAL_K])} foundational + "
          f"{len(recent[:RECENT_K])} recent = {len(top_k)} total", file=sys.stderr)
    for w in top_k:
        src = w.get("source", "?")
        print(f"  [{w['final_score']:.3f} rel={w['llm_relevance']:.1f} {src}] {w['title'][:70]} ({w['year']}) cit={w.get('cited_by_count',0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
