"""
QUAL-SG Step 1: Paper Retrieval (Bao et al., SurveyGen / QUAL-SG, 2025)
1. Topic-relevance search per sub-topic -> initial candidate set D
2. Co-citation expansion via OpenAlex referenced_works -> D_ex
3. Quality-indicator enrichment: citation performance, author influence (h-index), venue reputation
4. Re-ranking by averaging ranks across topical relevance, academic impact, content diversity
5. Select top-K
"""
import sys
import time

from common import OPENALEX, get, reconstruct_abstract, tfidf_vectors, cosine, percentile_rank, save_json

SUBTOPIC_QUERIES = {
    "continual pretraining / adaptation": [
        "continual pre-training large language models adaptation",
        "adapting pretrained language models new architecture",
        "transferring autoregressive language model new architecture distillation",
    ],
    "text diffusion models": [
        "diffusion language models text generation",
        "discrete diffusion masked language model text",
        "score-based discrete diffusion generative model text",
        "masked diffusion language model scaling",
        "adapting autoregressive language model into diffusion model",
        "large diffusion language model instruction tuning reasoning",
        "discrete state-space denoising diffusion model",
        "categorical multinomial diffusion generative model discrete data",
        "estimating ratios data distribution discrete diffusion score entropy",
        "absorbing state diffusion generative model discrete data",
    ],
    "non-autoregressive generation": [
        "non-autoregressive text generation parallel decoding",
        "parallel decoding large language model inference speedup",
        "any-order autoregressive language model infilling",
    ],
}

DATE_FROM = "2021-01-01"
PER_QUERY_LIMIT = 25
TARGET_D_SIZE = 70
COCITATION_MIN = 2
COCITATION_CAP = 25
FINAL_K = 33  # match anchor paper's §5 reference count for a fair overlap comparison


def search_openalex(query, per_page=PER_QUERY_LIMIT, date_from=DATE_FROM):
    params = {
        "search": query,
        "per-page": per_page,
        "sort": "relevance_score:desc",
        "filter": f"from_publication_date:{date_from},has_abstract:true",
    }
    data = get(f"{OPENALEX}/works", params=params)
    if not data:
        return []
    return data.get("results", [])


def works_by_ids(ids, batch_size=40):
    out = {}
    ids = [i for i in ids if i]
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        filt = "openalex_id:" + "|".join(batch)
        data = get(f"{OPENALEX}/works", params={"filter": filt, "per-page": batch_size})
        if not data:
            continue
        for w in data.get("results", []):
            out[w["id"]] = w
        time.sleep(0.2)
    return out


def authors_hindex(author_ids, batch_size=40):
    out = {}
    author_ids = [a for a in set(author_ids) if a]
    for i in range(0, len(author_ids), batch_size):
        batch = author_ids[i : i + batch_size]
        filt = "openalex_id:" + "|".join(batch)
        data = get(f"{OPENALEX}/authors", params={"filter": filt, "per-page": batch_size})
        if not data:
            continue
        for a in data.get("results", []):
            out[a["id"]] = (a.get("summary_stats") or {}).get("h_index", 0)
        time.sleep(0.2)
    return out


def sources_reputation(source_ids, batch_size=40):
    out = {}
    source_ids = [s for s in set(source_ids) if s]
    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i : i + batch_size]
        filt = "openalex_id:" + "|".join(batch)
        data = get(f"{OPENALEX}/sources", params={"filter": filt, "per-page": batch_size})
        if not data:
            continue
        for s in data.get("results", []):
            stats = s.get("summary_stats") or {}
            out[s["id"]] = stats.get("h_index", 0)
        time.sleep(0.2)
    return out


def slim(w, subarea=None, relevance_score=None):
    authorships = w.get("authorships") or []
    first_author = authorships[0]["author"] if authorships else {}
    source = (w.get("primary_location") or {}).get("source") or {}
    return {
        "id": w["id"],
        "title": w.get("title") or w.get("display_name") or "",
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count", 0),
        "authors": [a["author"].get("display_name") for a in authorships],
        "first_author_id": first_author.get("id"),
        "venue": source.get("display_name"),
        "venue_id": source.get("id"),
        "referenced_works": w.get("referenced_works") or [],
        "doi": w.get("doi"),
        "openalex_relevance": relevance_score,
        "subarea": subarea,
    }


def main():
    print("== Step 1a: topic-relevance search ==", file=sys.stderr)
    candidates = {}
    for subarea, queries in SUBTOPIC_QUERIES.items():
        for q in queries:
            results = search_openalex(q)
            print(f"  query={q!r} -> {len(results)} results", file=sys.stderr)
            for rank, w in enumerate(results):
                rel = 1.0 - rank / max(len(results), 1)  # relative relevance within this query
                sid = w["id"]
                if sid not in candidates or candidates[sid]["openalex_relevance"] < rel:
                    candidates[sid] = slim(w, subarea=subarea, relevance_score=rel)
            time.sleep(0.3)

    D = list(candidates.values())
    print(f"Initial candidate set D: {len(D)} papers", file=sys.stderr)

    print("== Step 1b: co-citation expansion ==", file=sys.stderr)
    from collections import Counter

    cocite_counts = Counter()
    for w in D:
        for ref in w["referenced_works"]:
            cocite_counts[ref] += 1
    known_ids = {w["id"] for w in D}
    expand_ids = [
        rid for rid, c in cocite_counts.most_common()
        if c >= COCITATION_MIN and rid not in known_ids
    ][:COCITATION_CAP]
    print(f"  {len(expand_ids)} co-cited works to fetch (min_count={COCITATION_MIN})", file=sys.stderr)
    fetched = works_by_ids(expand_ids)
    for sid, w in fetched.items():
        D.append(slim(w, subarea="co-citation-expansion", relevance_score=0.3))

    print(f"Expanded candidate set D_ex: {len(D)} papers", file=sys.stderr)

    print("== Step 1c: quality-indicator enrichment ==", file=sys.stderr)
    author_ids = [w["first_author_id"] for w in D]
    hindex_map = authors_hindex(author_ids)
    venue_ids = [w["venue_id"] for w in D]
    venue_rep_map = sources_reputation(venue_ids)
    for w in D:
        w["author_hindex"] = hindex_map.get(w["first_author_id"], 0)
        w["venue_hindex"] = venue_rep_map.get(w["venue_id"], 0)

    print("== Step 1d: re-ranking (relevance, academic impact, diversity) ==", file=sys.stderr)
    # topical relevance rank
    rel_scores = [w["openalex_relevance"] for w in D]
    rel_ranks = percentile_rank(rel_scores)

    # academic impact = avg percentile of (citations, author h-index, venue h-index)
    cit_ranks = percentile_rank([w["cited_by_count"] for w in D])
    auth_ranks = percentile_rank([w["author_hindex"] for w in D])
    venue_ranks = percentile_rank([w["venue_hindex"] for w in D])
    impact_ranks = [(c + a + v) / 3 for c, a, v in zip(cit_ranks, auth_ranks, venue_ranks)]

    # content diversity = avg TF-IDF distance to rest of pool
    vectors = tfidf_vectors([w["title"] + ". " + w["abstract"] for w in D])
    n = len(vectors)
    div_scores = []
    for i in range(n):
        if n <= 1:
            div_scores.append(0.0)
            continue
        # sample against up to 30 others for speed
        others = list(range(n))
        others.remove(i)
        if len(others) > 40:
            step = len(others) // 40
            others = others[::step][:40]
        dists = [1 - cosine(vectors[i], vectors[j]) for j in others]
        div_scores.append(sum(dists) / len(dists))
    div_ranks = percentile_rank(div_scores)

    for i, w in enumerate(D):
        w["rank_relevance"] = rel_ranks[i]
        w["rank_impact"] = impact_ranks[i]
        w["rank_diversity"] = div_ranks[i]
        w["final_score"] = (rel_ranks[i] + impact_ranks[i] + div_ranks[i]) / 3

    D.sort(key=lambda w: w["final_score"], reverse=True)
    save_json("../data/candidates_all.json", D)

    top_k = D[:FINAL_K]
    save_json("../data/candidates_topK.json", top_k)
    print(f"Saved {len(D)} total candidates and top-{len(top_k)} selection.", file=sys.stderr)
    for w in top_k[:10]:
        print(f"  [{w['final_score']:.3f}] {w['title'][:80]} ({w['year']})", file=sys.stderr)


if __name__ == "__main__":
    main()
