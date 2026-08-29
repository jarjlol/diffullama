"""
Implementation of the evaluation framework from the SECOND paper the instructor provided:

  "Large Language Models for Automated Literature Review: An Evaluation of Reference
   Generation, Abstract Writing, and Review Composition"  (arXiv:2412.13612)

The SOTA assignment offered two papers. QUAL-SG (arXiv:2508.17647) was implemented as the
*generation* method (see REPORT.md). This file implements the other one -- an *evaluation*
framework -- and applies it to the survey QUAL-SG produced, making the assignment
generate-then-evaluate rather than generate-only.

## The framework's three tasks, mapped onto this project

| Their task | Their setting | Our analogue |
|---|---|---|
| 1. Reference generation | LLM asked to produce references | our 48 retrieved+ranked references |
| 2. Abstract writing | LLM summarises a paper | our survey's Introduction vs the anchor's Sec 5 |
| 3. Review composition | LLM writes a full review | our full survey vs the anchor's Sec 5 |

The "human-written counterpart" is the anchor paper's own Sec 5 Related Work
(`data/anchor_fulltext.txt`) -- the same ground truth REPORT.md Sec 6 uses.

## Faithful vs substituted

| Metric | Paper | Here |
|---|---|---|
| Reference accuracy | `True(r) = 1 if (title correct AND >=1 other field) OR (title incorrect AND >=3 other fields)`, 80% title match | **faithful** |
| Title Search Rate | proportion retrievable from Semantic Scholar | **source substituted** -> OpenAlex (see below) |
| Semantic similarity | cosine over `text-embedding-3-large` | **substituted** -> pure-python TF-IDF cosine |
| ROUGE-1 / -2 / -L | n-gram + LCS overlap | **faithful** (pure python) |
| NLI entailment | TRUE model / GPT-4o | **NOT IMPLEMENTED** -- a lexical proxy is reported and is explicitly not NLI |

Usage:
    cd litreview/scripts
    python eval_framework_2412.py                  # OpenAlex, original reference set
    python eval_framework_2412.py --judged         # LLM-judged reference set
    python eval_framework_2412.py --source s2      # Semantic Scholar (needs an API key)
    python eval_framework_2412.py --skip-refs      # Tasks 2+3 only, no network
"""
import argparse
import re
import sys
import time
from collections import Counter

from common import load_json, save_json, tfidf_vectors, cosine, get

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_SEARCH = "https://api.openalex.org/works"

TITLE_MATCH_THRESHOLD = 0.80   # the paper's 80% title match rate


# ---------------------------------------------------------------- text utils
def norm_tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def token_f1_ratio(a, b):
    """Bag-of-tokens overlap ratio, used for the paper's 80% title-match rule."""
    ta, tb = Counter(norm_tokens(a)), Counter(norm_tokens(b))
    if not ta or not tb:
        return 0.0
    overlap = sum((ta & tb).values())
    return overlap / max(sum(ta.values()), sum(tb.values()))


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate, reference, n):
    c, r = norm_tokens(candidate), norm_tokens(reference)
    if len(c) < n or len(r) < n:
        return 0.0
    cg, rg = ngrams(c, n), ngrams(r, n)
    overlap = sum((cg & rg).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(sum(cg.values()), 1)
    rec = overlap / max(sum(rg.values()), 1)
    return 2 * prec * rec / (prec + rec)


def lcs_len(a, b):
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l(candidate, reference):
    c, r = norm_tokens(candidate), norm_tokens(reference)
    if not c or not r:
        return 0.0
    l = lcs_len(c, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(c), l / len(r)
    beta = 1.2
    return ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec)


def tfidf_cosine(a, b):
    v = tfidf_vectors([a, b])
    return cosine(v[0], v[1])


# ------------------------------------------------------- Task 1: references
#
# A BUG CAUGHT DURING DEVELOPMENT, recorded because it is instructive:
#
# The first version used the Semantic Scholar search endpoint and treated ANY failed lookup
# as "reference not found". Unauthenticated S2 search returns HTTP 429 almost immediately, so
# it reported NOT FOUND for papers as well known as Diffusion-LM, and would have produced a
# hallucination rate near 1.0 -- catastrophically wrong, and wrong in the direction that looks
# like an exciting finding.
#
# Two fixes:
#   (a) default to OpenAlex, which this project already uses keylessly and which tolerates
#       this volume;
#   (b) distinguish LOOKUP_FAILED from NOT_FOUND, and exclude failures from the denominator,
#       so an infrastructure failure can never masquerade as a hallucinated reference.

def openalex_lookup(title):
    """Returns (rec, score) | None (genuinely absent) | 'FAILED' (infrastructure)."""
    data = get(OPENALEX_SEARCH, params={
        "search": title, "per-page": 3,
        "select": "id,title,publication_year,authorships,primary_location,doi",
    })
    if data is None:
        return "FAILED"
    results = data.get("results") or []
    if not results:
        return None
    best, best_score = None, 0.0
    for rec in results:
        sc = token_f1_ratio(title, rec.get("title") or "")
        if sc > best_score:
            best, best_score = rec, sc
    return (best, best_score) if best else None


def s2_lookup(title):
    """Semantic Scholar variant. Heavily rate-limited without an API key."""
    data = get(S2_SEARCH, params={
        "query": title, "limit": 3,
        "fields": "title,year,authors,venue,externalIds",
    })
    if data is None:
        return "FAILED"
    if not data.get("data"):
        return None
    best, best_score = None, 0.0
    for rec in data["data"]:
        sc = token_f1_ratio(title, rec.get("title", ""))
        if sc > best_score:
            best, best_score = rec, sc
    return (best, best_score) if best else None


def count_other_fields(rec, r, source):
    """Count matching 'other elements' for the paper's accuracy rule."""
    other = 0
    if source == "openalex":
        year_key, doi = rec.get("publication_year"), rec.get("doi")
        names = [(a.get("author") or {}).get("display_name", "")
                 for a in (rec.get("authorships") or [])]
        venue = ((rec.get("primary_location") or {}).get("source") or {}).get("display_name")
    else:
        year_key, doi = rec.get("year"), (rec.get("externalIds") or {}).get("DOI")
        names = [a.get("name", "") for a in (rec.get("authors") or [])]
        venue = rec.get("venue")

    if year_key and r.get("year") and abs(int(year_key) - int(r["year"])) <= 1:
        other += 1
    our_authors = r.get("authors") or []
    if our_authors:
        a0 = our_authors[0]
        name = a0 if isinstance(a0, str) else a0.get("name", "")
        if name:
            surname = name.split()[-1].lower()
            if any(surname == n.split()[-1].lower() for n in names if n):
                other += 1
    if venue and r.get("venue") and token_f1_ratio(venue, r["venue"]) > 0.5:
        other += 1
    if doi:
        other += 1
    return other


ARXIV_API = "https://export.arxiv.org/api/query"


def arxiv_verify(arxiv_id, title):
    """Verify a reference against the arXiv API by ID. Returns (matched, score) | 'FAILED'.

    Needed because OpenAlex lags badly on very recent preprints -- REPORT.md Sec 8 documents
    this for the retrieval stage, and it bites here too: several genuine 2025-2026 arXiv papers
    return an unrelated OpenAlex best-match rather than nothing at all.
    """
    import xml.etree.ElementTree as ET
    try:
        import requests
        resp = requests.get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1}, timeout=20)
        if resp.status_code != 200:
            return "FAILED"
        root = ET.fromstring(resp.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        got = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        if not got:
            return None
        return (got, token_f1_ratio(title, got))
    except Exception:
        return "FAILED"


def task1_reference_accuracy(refs, source="openalex", sleep=0.2):
    lookup = openalex_lookup if source == "openalex" else s2_lookup
    n_true = n_found = n_failed = 0
    n_via_arxiv = 0
    rows = []

    for i, r in enumerate(refs, 1):
        title = r.get("title", "")
        hit = lookup(title)
        time.sleep(sleep)

        if hit == "FAILED":
            n_failed += 1
            rows.append({"title": title, "status": "LOOKUP_FAILED", "true": None})
            print(f"  [{i:>2}/{len(refs)}] LOOKUP FAILED (infrastructure, NOT a hallucination)"
                  f"  {title[:40]}", file=sys.stderr)
            continue

        rec, score, via = (None, 0.0, source)
        if hit is not None:
            rec, score = hit

        # FALLBACK: a below-threshold best-match means the database returned an UNRELATED
        # paper, i.e. it does not hold this reference -- not that the reference is fake.
        # Re-check against arXiv by id before concluding anything.
        if score < TITLE_MATCH_THRESHOLD and r.get("arxiv_id"):
            ax = arxiv_verify(r["arxiv_id"], title)
            time.sleep(sleep)
            if ax not in (None, "FAILED"):
                ax_title, ax_score = ax
                if ax_score > score:
                    score, via = ax_score, "arxiv"
                    if ax_score >= TITLE_MATCH_THRESHOLD:
                        n_via_arxiv += 1

        if score < TITLE_MATCH_THRESHOLD:
            rows.append({"title": title, "status": "NOT_FOUND",
                         "best_match": round(score, 3), "true": False})
            print(f"  [{i:>2}/{len(refs)}] NOT FOUND (best match {score:.2f})  {title[:40]}",
                  file=sys.stderr)
            continue

        n_found += 1
        other = count_other_fields(rec, r, source) if via == source and rec else 1
        is_true = other >= 1                      # title correct, so >=1 other field suffices
        n_true += int(is_true)
        rows.append({"title": title, "status": "FOUND", "via": via,
                     "title_match": round(score, 3), "title_correct": True,
                     "other_correct": other, "true": is_true})
        print(f"  [{i:>2}/{len(refs)}] {'OK ' if is_true else 'BAD'} "
              f"[{via}] match={score:.2f} other={other}  {title[:38]}", file=sys.stderr)

    n_scored = len(refs) - n_failed        # denominator excludes infrastructure failures
    return {
        "source": source,
        "n_references": len(refs),
        "n_lookup_failed": n_failed,
        "n_scored": n_scored,
        "n_rescued_via_arxiv": n_via_arxiv,
        "title_search_rate": (n_found / n_scored) if n_scored else None,
        "reference_accuracy": (n_true / n_scored) if n_scored else None,
        "hallucination_rate": (1 - n_true / n_scored) if n_scored else None,
        "per_reference": rows,
    }


# ------------------------------- Tasks 2 & 3: summary / composition quality
def lexical_entailment_proxy(candidate, reference):
    """NOT NLI. Fraction of the reference's content tokens present in the candidate.

    The paper uses a TRUE/GPT-4o entailment model. None is available here, so this reports
    token-level recall instead. It measures COVERAGE only -- it cannot detect contradiction,
    negation, or hallucinated claims.
    """
    stop = set("the a an of and or to in on for with is are was were be been by as at from "
               "this that these those it its their we our".split())
    c = set(t for t in norm_tokens(candidate) if t not in stop)
    r = [t for t in norm_tokens(reference) if t not in stop]
    if not r:
        return 0.0
    return sum(1 for t in r if t in c) / len(r)


def content_metrics(candidate, reference):
    return {
        "semantic_similarity_tfidf": round(tfidf_cosine(candidate, reference), 4),
        "rouge_1": round(rouge_n(candidate, reference, 1), 4),
        "rouge_2": round(rouge_n(candidate, reference, 2), 4),
        "rouge_l": round(rouge_l(candidate, reference), 4),
        "lexical_entailment_proxy": round(lexical_entailment_proxy(candidate, reference), 4),
    }


def extract_anchor_related_work():
    with open("../data/anchor_fulltext.txt", encoding="utf-8") as f:
        full = f.read()
    return full[full.index("5   R ELATED W ORK"):full.index("6   C ONCLUSION")]


def extract_survey_intro(survey):
    m = re.search(r"##\s*1\.\s*Introduction(.*?)(?=\n##\s)", survey, flags=re.S)
    return m.group(1).strip() if m else survey[:4000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", action="store_true",
                    help="evaluate the LLM-judged reference set instead of the submitted one")
    ap.add_argument("--skip-refs", action="store_true", help="skip Task 1 (no network)")
    ap.add_argument("--source", choices=["openalex", "s2"], default="openalex",
                    help="database for Task 1 (default openalex; s2 needs an API key)")
    args = ap.parse_args()

    refs_path = ("../data/candidates_topK_llm_judged.json" if args.judged
                 else "../data/candidates_topK.json")
    refs = load_json(refs_path)
    survey = open("../output/generated_survey.md", encoding="utf-8").read()
    anchor_rw = extract_anchor_related_work()
    intro = extract_survey_intro(survey)

    print("=" * 78)
    print("Evaluation framework of arXiv:2412.13612, applied to the QUAL-SG survey")
    print("=" * 78)
    print(f"reference set : {refs_path}  ({len(refs)} refs)")
    print(f"survey        : {len(survey.split())} words")
    print(f"anchor Sec 5  : {len(anchor_rw.split())} words")
    print(f"survey intro  : {len(intro.split())} words\n")

    results = {"reference_set": refs_path}

    if not args.skip_refs:
        print(f"Task 1 - Reference generation  (source: {args.source})")
        t1 = task1_reference_accuracy(refs, source=args.source)
        results["task1_reference_generation"] = t1
        print(f"\n  Lookups failed      : {t1['n_lookup_failed']} (excluded from denominator)")
        print(f"  Scored              : {t1['n_scored']}/{t1['n_references']}")
        if t1["n_scored"]:
            print(f"  Title Search Rate   : {t1['title_search_rate']:.3f}")
            print(f"  Reference accuracy  : {t1['reference_accuracy']:.3f}")
            print(f"  Hallucination rate  : {t1['hallucination_rate']:.3f}")
    else:
        print("Task 1 - SKIPPED")

    print("\nTask 2 - Abstract writing  (survey Introduction vs anchor Sec 5)")
    t2 = content_metrics(intro, anchor_rw)
    results["task2_abstract_writing"] = t2
    for k, v in t2.items():
        print(f"  {k:<28} {v}")

    print("\nTask 3 - Review composition  (full survey vs anchor Sec 5)")
    t3 = content_metrics(survey, anchor_rw)
    results["task3_review_composition"] = t3
    for k, v in t3.items():
        print(f"  {k:<28} {v}")

    out = ("../output/eval_framework_2412_judged.json" if args.judged
           else "../output/eval_framework_2412.json")
    save_json(out, results)
    print(f"\nwrote {out}")

    print("\n" + "=" * 78)
    print("""CAVEATS -- read before quoting any number above.

1. NLI IS NOT IMPLEMENTED. The paper uses a TRUE/GPT-4o entailment model; none is available
   here. `lexical_entailment_proxy` is token-level recall -- COVERAGE only. It cannot detect
   contradiction, negation, or hallucinated claims. Do not report it as factual consistency.

2. Semantic similarity is TF-IDF cosine, not `text-embedding-3-large`. Weaker on paraphrase.
   Same substitution REPORT.md Sec 4 documents for the retrieval stage.

3. TASK 1 MEASURES SOMETHING STRUCTURALLY DIFFERENT HERE, AND IS PARTLY CIRCULAR.
   The paper's references are GENERATED by an LLM, so hallucination is the failure mode being
   measured. Ours are RETRIEVED from OpenAlex and arXiv -- and Task 1 then verifies them
   against OpenAlex. A high score is therefore close to tautological: it confirms the pipeline
   returned what it fetched. Report it as a sanity check on the retrieval plumbing, never as
   evidence that an LLM avoided hallucinating. A genuinely independent check would use a
   database the pipeline never queried.

4. Tasks 2 and 3 compare a ~3,800-word survey against a ~700-word related-work section. The
   length mismatch depresses ROUGE and cosine by construction -- REPORT.md Sec 6 makes the same
   point. These numbers are meaningful for comparing reference sets, not in absolute terms.""")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
