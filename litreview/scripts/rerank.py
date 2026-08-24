"""
QUAL-SG Step 1d (corrected): re-rank using a topical relevance score S_t computed
by content matching against title+abstract (substituting, deterministically and
reproducibly, for the paper's LLM-judge API call -- an earlier manual eyeballed-index
version of this had an off-by-N row-counting bug, so this version scores by matching
text content directly rather than by list position), then combine with
academic-impact and content-diversity ranks as in the paper (avg of 3 ranks).
Also dedupes near-identical titles (OpenAlex sometimes has duplicate records).
"""
import re
from collections import defaultdict

from common import tfidf_vectors, cosine, percentile_rank, save_json, load_json

FINAL_K = 33

# Topical relevance S_t: content-matched keyword scoring against the survey's 3
# sub-topics. Strong (weight 3) terms are unambiguous core-topic signals; medium
# (weight 1.5) terms are supportive/background signals. Score is capped at 5.
STRONG_TERMS = [
    r"diffusion (language|text) model", r"text diffusion", r"discrete diffusion",
    r"masked diffusion", r"non-?autoregressive", r"semi-autoregressive",
    r"absorbing (state |)diffusion", r"diffusion.{0,20}text generation",
    r"continual pre-?training", r"parallel decoding", r"any-order generation",
]
MEDIUM_TERMS = [
    r"\bdiffusion model", r"denoising diffusion", r"masked language model",
    r"state space model", r"\bmamba\b", r"speculative decoding", r"speculative sampling",
    r"multi-?token prediction", r"in-?filling", r"autoregressive language model",
    r"architecture (transfer|adaptation)", r"model adaptation", r"curriculum learning",
    r"score-based generative", r"masked generative transformer",
]
OFF_TOPIC_DOMAIN = [
    r"image|video|3d shape|protein|molecul|speech synthesis|text-to-speech|clinical|"
    r"healthcare|medical|tabular|sentiment|recommend|gesture|motion generation|"
    r"chemistry|construction industry",
]


def relevance_score(title, abstract):
    text = f"{title}. {abstract}".lower()
    score = 0.0
    for pat in STRONG_TERMS:
        if re.search(pat, text):
            score += 3
    for pat in MEDIUM_TERMS:
        if re.search(pat, text):
            score += 1.5
    # penalize papers that are clearly about an unrelated application domain,
    # unless they also carry a strong core-topic signal (e.g. "diffusion language
    # model for molecule generation" should still count as diffusion-LM relevant)
    has_strong = any(re.search(pat, text) for pat in STRONG_TERMS)
    if not has_strong and re.search(OFF_TOPIC_DOMAIN[0], text):
        score -= 2
    return max(0.0, min(5.0, score))


RELEVANCE_GATE = 3.0  # keep only candidates scoring >=3/5 on topical relevance


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def main():
    D = load_json("../data/candidates_all.json")
    for w in D:
        w["llm_relevance"] = relevance_score(w["title"], w["abstract"])

    # dedupe by normalized title, keep the copy with higher citation count
    best_by_title = {}
    for w in D:
        key = norm_title(w["title"])
        if not key:
            continue
        if key not in best_by_title or w["cited_by_count"] > best_by_title[key]["cited_by_count"]:
            best_by_title[key] = w
    D = list(best_by_title.values())

    gated = [w for w in D if w["llm_relevance"] >= RELEVANCE_GATE]
    print(f"After dedup: {len(D)} papers. After relevance gate (>= {RELEVANCE_GATE}): {len(gated)}")

    rel_ranks = percentile_rank([w["llm_relevance"] for w in gated])
    cit_ranks = percentile_rank([w["cited_by_count"] for w in gated])
    auth_ranks = percentile_rank([w["author_hindex"] for w in gated])
    venue_ranks = percentile_rank([w["venue_hindex"] for w in gated])
    impact_ranks = [(c + a + v) / 3 for c, a, v in zip(cit_ranks, auth_ranks, venue_ranks)]

    vectors = tfidf_vectors([w["title"] + ". " + w["abstract"] for w in gated])
    n = len(vectors)
    div_scores = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        dists = [1 - cosine(vectors[i], vectors[j]) for j in others] or [0.0]
        div_scores.append(sum(dists) / len(dists))
    div_ranks = percentile_rank(div_scores)

    for i, w in enumerate(gated):
        w["rank_relevance"] = rel_ranks[i]
        w["rank_impact"] = impact_ranks[i]
        w["rank_diversity"] = div_ranks[i]
        w["final_score"] = (rel_ranks[i] + impact_ranks[i] + div_ranks[i]) / 3

    gated.sort(key=lambda w: w["final_score"], reverse=True)
    save_json("../data/candidates_reranked.json", gated)

    top_k = gated[:FINAL_K]
    save_json("../data/candidates_topK.json", top_k)
    print(f"Final top-{len(top_k)} selection:")
    for w in top_k:
        print(f"  [{w['final_score']:.3f} rel={w['llm_relevance']}] {w['title'][:75]} ({w['year']}) cit={w['cited_by_count']}")


if __name__ == "__main__":
    main()
