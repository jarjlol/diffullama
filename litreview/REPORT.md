# Literature Review Generation for Diffusion Language Models: An Implementation of QUAL-SG (SurveyGen)

## 1. Assignment and Approach

The assignment asks for: (1) extraction of the domain/topic of a chosen base paper, (2) implementation of a
literature-review-generation methodology from a recent (2024+) top-tier NLP paper, (3) generation of a survey
for the base paper's domain using that implementation, and (4) comparison of the generated survey against the
base paper's own related-work section.

**Base (anchor) paper:** Gong et al., *Scaling Diffusion Language Models via Adaptation from Autoregressive
Models* (DiffuLLaMA), ICLR 2025.

**Methodology paper implemented:** Bao, Nayeem, Rafiei, Zhang, *SurveyGen: Quality-Aware Scientific Survey
Generation with Large Language Models* (arXiv:2508.17647, 2025), specifically its **QUAL-SG** framework —
chosen over the alternative option (an LLM literature-writing *evaluation* framework, arXiv:2412.13612)
because QUAL-SG is itself a generation methodology with an explicit two-phase retrieval + generation pipeline
that maps directly onto the assignment's structure, whereas the alternative paper is an evaluation harness
rather than a method to implement.

## 2. Domain/Topic Extraction

Reading the anchor paper directly (abstract, introduction, and its own §5 Related Work, extracted via
`pdftotext`), its domain decomposes into three sub-areas — the same three the paper's own related-work section
is organized around:

1. **Continual pre-training / adaptation** of existing language models to new domains or architectures.
2. **Text diffusion models** — continuous and discrete/masked diffusion processes for language generation.
3. **Non-autoregressive (NAR) text generation** — the broader family of parallel/non-left-to-right decoding
   methods that diffusion LMs belong to.

The generated survey's topic was scoped to cover all three, matching the anchor paper's own scope, to make the
final overlap comparison meaningful.

## 3. QUAL-SG Methodology (as implemented)

QUAL-SG (illustrated in Fig. 2 of the paper) has two stages:

**Stage 1 — Paper Retrieval:**
(a) semantic/topical search against candidate papers → initial set *D*;
(b) **co-citation expansion** — any paper cited by ≥2 papers already in *D* is added, to catch influential
works that aren't textually similar to the query (the paper's own example: "Backpropagation" wouldn't surface
via semantic search on "deep learning" but is frequently co-cited);
(c) enrichment with **quality indicators** — citation performance, author influence (h-index), venue
reputation;
(d) **re-ranking** by averaging each candidate's rank across three scores — topical relevance (LLM-judged),
academic impact (weighted quality indicators), and content diversity (semantic distance to the rest of the
pool) — and selecting the top-K.

**Stage 2 — Survey Generation (RAG-based / "Task 2" in the paper):** given the topic and the top-K retrieved
papers, an LLM first generates a **structured outline**, then **expands each section** using the retrieved
papers as grounding context, producing the full survey with citations.

## 4. Implementation and Deviations from the Paper

This environment has no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` and no package installer (`pip` is unavailable),
so several components were substituted with faithful, documented equivalents rather than the paper's exact
tooling:

| QUAL-SG component | Paper's approach | This implementation |
|---|---|---|
| Retrieval corpus | S2ORC (81M papers) | **OpenAlex** (works/authors/sources APIs, free & keyless) + **arXiv API** (for 2025-2026 recency, since OpenAlex indexing lags very recent preprints) |
| Embedding similarity | Learned embeddings (MTEB-selected model) | Pure-Python **TF-IDF cosine similarity** (no `numpy`/`sentence-transformers` available) |
| Topical relevance S_t | LLM-judge API call, 0-5 score | **Content-matched keyword scorer** over title+abstract (deterministic, reproducible substitute — an earlier version used Claude manually eyeballing row indices, which had an off-by-N counting bug and was discarded in favor of this approach) |
| Academic impact S_a | Citation count, author h-index, venue h-index/i10-index/CORE rank via OpenAlex | Same — citation count, author h-index, venue h-index, all fetched live from OpenAlex |
| Content diversity S_d | Average embedding distance to other candidates | Average (1 − TF-IDF cosine) to other candidates |
| Outline + section generation LLM | GPT-4.1 / Claude-3.7-Sonnet / GLM / etc. via API | **Claude (this session)**, acting as the generator LLM directly, using the retrieved+ranked references as grounding context (the same role an API call plays in the paper's pipeline) |
| Final selection size | K = number of references in the matched human survey | K = 48, chosen to include all papers clearing the relevance gate after stratifying between pre-2025 "foundational" and 2025-2026 "recent" papers (see §5) |

All retrieval/ranking code is pure standard-library Python plus `requests` (the only third-party package
available); see `scripts/` for the full pipeline (`retrieve.py` → `arxiv_fetch.py` → `merge_finalize.py` →
`format_refs.py` → generation → `evaluate.py`).

**A retrieval-quality issue found and fixed during implementation:** an initial run that combined a broad
OpenAlex search with citation-count-heavy re-ranking surfaced high-citation but topically irrelevant papers
(e.g., *LLaMA*, *GLM*, protein-structure prediction) at the top, because raw academic-impact scores dominated
a weak relevance proxy. This was corrected by adding a genuine topical-relevance gate (score ≥ 3/5) before
impact/diversity re-ranking — consistent with the paper's own design intent that impact/diversity should
refine an *already topically relevant* pool, not override relevance.

**A second issue:** merging in very recent arXiv preprints caused an explosion of near-duplicate, ultra-narrow
2025-2026 "Masked Diffusion Language Model + [inference trick]" papers with 0 citations each, which — because
diversity scoring rewards title uniqueness and impact scoring is flat at zero for all of them — crowded out
foundational, well-established papers (Mamba, MDLM, SSD-LM, DiffusionBERT, DiffuSeq) purely by volume. This was
fixed by stratifying the final selection: all pre-2025 relevant papers with ≥5 citations, plus a curated set of
2025-2026 papers favoring full model/framework releases over narrow single-technique variants.

## 5. Retrieved References and Generated Survey

- **48 references** were retrieved, quality-ranked, and cited in the generated survey — see
  [`data/reference_brief.txt`](data/reference_brief.txt) for the full annotated list and
  [`data/candidates_topK.json`](data/candidates_topK.json) for structured metadata (citations, h-index,
  venue, source, per-criterion ranks).
- The generated survey outline is in [`output/outline.md`](output/outline.md).
- The full generated survey (~3,800 words, 8 sections) is in
  [`output/generated_survey.md`](output/generated_survey.md).

## 6. Comparison with the Anchor Paper's Related Work (§5)

The anchor paper's §5 was extracted verbatim via `pdftotext` (saved permanently at
[`data/anchor_fulltext.txt`](data/anchor_fulltext.txt) for reproducibility) and its 33 unique citations were
resolved into a ground-truth list ([`data/anchor_related_work_groundtruth.json`](data/anchor_related_work_groundtruth.json))
using **two independent sources**: a verbatim-quote match against the anchor's own References section text, and
a cross-check against the anchor paper's actual citation graph on Semantic Scholar (`GET
/graph/v1/paper/arXiv:2410.17891/references`, 98 records). 32/33 are now resolved to real, independently
verified titles; one (`Zhao 2024`) remains an honest unresolved placeholder rather than a guess. See §9 below
for how this cross-check also caught a real resolution error.

**Citation overlap** (`scripts/evaluate.py`, exact + fuzzy title matching, greedy 1-1 assignment):

| Metric | Value |
|---|---|
| Ground-truth citations (anchor §5) | 33 |
| Generated survey references | 48 |
| Matched (same paper) | 9 |
| Precision | 0.188 |
| Recall | 0.273 |
| F1 | 0.222 |

The 9 exact matches are: *Continual Pre-training of Language Models* (Ke et al.), *Mamba* (Gu & Dao),
*Diffusion-LM* (Li et al.), *A Reparameterized Discrete Diffusion Model for Text Generation* (Zheng et al.),
*Simple and Effective Masked Diffusion Language Models / MDLM* (Sahoo et al.), *DiffusionBERT* (He et al.),
*Non-Autoregressive Neural Machine Translation* (Gu et al.), *SSD-LM* (Han et al.), and *GENIE* (Lin et al.) —
i.e., the pipeline independently rediscovered several of the anchor paper's most central citations, including
its two closest methodological predecessors (DiffusionBERT and SSD-LM) and a paper by the anchor's own senior
author (the reparameterized discrete diffusion model, Zheng et al., which shares author Lingpeng Kong with the
anchor paper). For context, QUAL-SG's own paper reports F1 = 5.9-16.7% for its baselines/best system on its
(much larger, in-domain) benchmark — our F1 = 22.2% is in the same range or somewhat above it, though the two
numbers are not directly comparable given the very different scale and domain-matching setup.

Most **unmatched** ground-truth citations fall into two explainable categories: (a) foundational
image/general-diffusion papers the anchor cites for background (DDPM, score-based generative modeling,
DALL-E) that a text-diffusion-focused survey reasonably keeps only as passing mentions rather than full
citations, and (b) a handful of the anchor's key discrete-diffusion citations (SEDD/Lou et al., D3PM/Austin et
al., Plaid/Gulrajani & Hashimoto, Ou et al.) that were **not surfaced by our retrieval** — largely because
their titles and abstracts use different phrasing (e.g., "discrete state-spaces," "estimating the ratios of
the data distribution") than our search queries anticipated, a genuine retrieval-recall gap rather than a
relevance-judgment error.

**Content similarity:** TF-IDF cosine = 0.221, ROUGE-L = 0.080 between the full generated survey and the
anchor's raw §5 text. These are naturally low in absolute terms because the generated survey (~3,800 words) is
roughly 5-6× longer than the anchor's condensed related-work paragraph (~700 words) — bag-of-words overlap
metrics computed against a much longer, more detailed document are diluted by design. The more meaningful
comparison is structural/topical, below.

**Structural consistency:** all three of the anchor's §5 subareas (continual pre-training/adaptation, text
diffusion models, non-autoregressive generation) are represented as dedicated major sections (§3, §4, §5) in
the generated survey, in the same conceptual order the anchor paper uses. The generated survey additionally
includes a background section, a section specifically on 2025-2026 developments published after the anchor
paper (which its own related work could not have cited), and an open-challenges section — appropriate for a
standalone survey but not expected in a paper's condensed related-work paragraph.

## 7. Source Verification

Every claim in this project that a paper "exists" and "says X" was independently checked, not just trusted from
API responses:

- **All 48 generated-survey references are confirmed real.** Each reference's DOI (or arXiv-DOI) was resolved
  live via HTTP HEAD request: 47/48 returned HTTP 200 directly; the one exception (PIMNet, an ACM DL paper)
  returns 403 to bots but was confirmed real and correctly attributed via the Crossref API (title + all 8
  authors + publication date matched exactly). A further sample of 13 arXiv-sourced references had their titles
  cross-checked against live arXiv API metadata by arXiv ID — 13/13 matched exactly, including several very
  recent (2025-2026) papers that could look implausible at a glance (e.g. arXiv IDs like `2602.xxxxx`, which are
  genuine given the current date).
- **The generated survey text was checked for over-close paraphrasing** of source abstracts (a word-level
  longest-common-substring scan against all 48 abstracts): the longest verbatim overlap found was 6 consecutive
  words (ordinary technical-term reuse, e.g. "performance gap between diffusion and autoregressive"), well
  within normal academic paraphrasing norms — no passages were copied wholesale.
- **The ground-truth citation list itself was cross-verified against a second, independent source** (the anchor
  paper's actual citation graph on Semantic Scholar) rather than trusted from the first manual extraction pass.
  This caught one real error: an earlier version of this project matched the anchor's "Lin et al. (2023)"
  citation to an unrelated NAACL paper (found via a same-surname/year regex hit) instead of the correct paper,
  GENIE (a diffusion-LM pretraining paper) — confirmed both by the DBLP venue-year key on Semantic Scholar and
  by re-reading the anchor's own §5 sentence, which describes a "pre-training and finetuning framework," not a
  paper about autoregressive-model limitations. It also resolved 5 previously-unresolved placeholder citations
  (`Xu 2024`, `Wang 2024`, `Zhang 2024c`, `Zheng 2024a`, `Wu 2024`, `Ye 2023`) to real titles, and left one
  (`Zhao 2024`) honestly unresolved rather than guessed. See the `resolution` field on each entry in
  [`data/anchor_related_work_groundtruth.json`](data/anchor_related_work_groundtruth.json) for the full
  evidence trail per citation.
- **A hardcoded, session-only temp-file path was found and fixed** in `evaluate.py` (it originally read the
  anchor paper's full text from this session's ephemeral scratchpad directory, which would not exist for anyone
  else running this code). The anchor's full `pdftotext` extraction is now saved permanently at
  [`data/anchor_fulltext.txt`](data/anchor_fulltext.txt) and referenced by a relative path.

## 8. Limitations

- No hosted LLM API was available, so the "LLM-judge" and "generator LLM" roles in QUAL-SG were played by
  Claude directly within this session rather than via reproducible API calls; a real reproduction with an API
  key would let every candidate (not just a title/abstract keyword match) be judged for relevance, likely
  improving retrieval recall on paraphrased/differently-worded but relevant papers (e.g., the missed SEDD/D3PM
  citations above).
- No embedding model was available (`pip` install was not possible in this environment), so TF-IDF cosine
  substitutes for the paper's learned semantic embeddings throughout — weaker at synonym/paraphrase matching,
  which plausibly explains part of the retrieval-recall gap noted above.
- OpenAlex's citation/author/venue data lags for very recent (2025-2026) preprints, so "academic impact" is
  uninformative for the newest third of the field; this was mitigated by stratifying foundational vs. recent
  papers rather than ranking them together, but a longer-window citation signal would be more principled.
- The relevance gate is a keyword/regex scorer rather than genuine semantic judgment, so it can both miss
  relevant papers using unfamiliar terminology and admit topically-adjacent-but-tangential ones (e.g., a few
  molecule-generation or 3D-mesh diffusion-LM papers included because they literally use "diffusion language
  model" methodology in a different application domain).

## 9. Reproducing This Pipeline

```
litreview/
  scripts/
    common.py          # HTTP+retry, TF-IDF, percentile-rank helpers, shared score_pool() ranker
    retrieve.py         # QUAL-SG Step 1: OpenAlex search + co-citation expansion + quality enrichment
    arxiv_fetch.py       # recency supplement via arXiv API
    rerank.py            # relevance scoring + rank-averaging re-rank (single-source version)
    merge_finalize.py    # merges OpenAlex + arXiv pools, re-ranks, stratifies foundational/recent
    apply_llm_judge.py    # extension (Sec 10): substitutes real LLM-judged relevance, re-runs selection
    format_refs.py        # formats top-K into the generation-step context file
    evaluate.py            # citation overlap (precision/recall/F1), TF-IDF/ROUGE-L, structural comparison
  data/                    # all intermediate + final retrieval artifacts (JSON)
  output/                  # outline.md, generated_survey.md, evaluation_results.json
```

Run order: `retrieve.py` → `arxiv_fetch.py` → `merge_finalize.py` → `format_refs.py` → (generation, done
inline by Claude using `data/reference_brief.txt`) → `evaluate.py`.

## 10. Extension: real LLM-judged relevance (post-submission follow-up)

Section 8 flagged the keyword-regex relevance scorer as the single biggest limitation of this
implementation. This section documents a follow-up pass that replaces it with genuine semantic
judgment and reports what changed, honestly, including where it *didn't* help.

**What triggered this:** a direct pool inspection (not just re-reading the report) confirmed six of
the anchor's unmatched ground-truth citations — D3PM, SEDD, Argmax Flows/Multinomial Diffusion,
Continuous Diffusion for Categorical Data, "Simplified and Generalized Masked Diffusion for Discrete
Data," and "Your Absorbing Discrete Diffusion..." — had genuinely never been retrieved at all, not
just mis-ranked. Targeted queries covering their specific phrasing ("discrete state-space diffusion,"
"estimating the ratios of the data distribution," "absorbing state diffusion," "multinomial/categorical
diffusion") were added to `retrieve.py` and `arxiv_fetch.py`; re-running retrieval confirmed all six now
surface in the raw candidate pool.

**A second, independent bug found in the process:** this report (Sec 9's pipeline map, both before and
after this section) has always described `merge_finalize.py` as stratifying foundational/recent papers
before ranking. Reading the actual committed script showed it did not — it was a flat top-35 cut over
the whole gated pool. Running it as committed (even before touching relevance scoring) reproduced
exactly the recency-dilution failure Sec 4 already diagnosed once: near-duplicate, 0-citation 2025-2026
preprints ("Deep Generative Methods and Tire Architecture Design" among them) out-ranked foundational,
highly-cited papers on the diversity axis alone. The stratification described in this report's own prose
has now been implemented as real code (`score_pool()`, moved into `common.py` and shared by both
`merge_finalize.py` and `apply_llm_judge.py`): foundational (pre-2025) and recent (2025+) papers are
rank-averaged *separately*, each keeping its own top-K slice (36 + 12), so a narrow recent preprint no
longer competes against Mamba or DiffusionBERT for the same slots.

**A third bug, caught before it did damage:** the LLM-relevance judgments below were first written as a
flat positional array. Counting the intended 218 entries by hand produced 222 — a transcription miscount
that, if used as-is, would have silently misaligned every score after the drift point to the wrong paper,
the exact class of "off-by-N" bug Sec 4 already documents once for this project (an earlier manual
row-counting version of `rerank.py`'s relevance scorer). Caught by checking `len(scores) == len(pool)`
before use; the array was discarded and rebuilt as an explicit `{index: score}` map instead
(`data/llm_relevance_scores.json`), which fails safe — a wrong entry is one wrong score, not a shift.
Index alignment was then spot-checked against titles before running anything downstream.

**The actual fix — real relevance judgment:** even with better retrieval and correct stratification, the
keyword-regex scorer (`rerank.py`'s `relevance_score()`) still let off-topic domains into the top-48 —
"Protein Design with Guided Discrete Diffusion," "RNADiffFold," "DiffBP" (3D molecules) — because its
off-topic penalty is skipped whenever a paper *also* matches a `STRONG_TERMS` pattern like "discrete
diffusion," which many non-text applications of discrete diffusion trigger incidentally. All 218 papers
surviving the relevance gate were read directly (title + abstract, `data/judge_pool.tsv`) and scored 0-5
on genuine topical fit by an LLM (Claude, this session) — the same role QUAL-SG's own "LLM-judge" API
call plays, and the same limitation as this project's earlier generation step: a documented one-time
judgment, not a scripted, repeatable API call (`data/llm_relevance_scores.json` records the scores and
this reasoning).

**Results** (`output/evaluation_results_llm_judged.json`, full pipeline re-run with the fixes above):

| Metric | Original (keyword scorer) | LLM-judged relevance |
|---|---|---|
| Matched / ground truth | 9 / 33 | **11 / 33** |
| Precision | 0.188 | **0.229** |
| Recall | 0.273 | **0.333** |
| F1 | 0.222 | **0.272** (+22.5% relative) |
| Off-topic domain papers in top-48 | several (tire design, protein, RNA, image, graph) | **zero** |

**Honest caveat — this is not a strict, paper-by-paper improvement.** DiffusionBERT, matched in the
original run, is *not* in the new top-48 despite being LLM-scored 5/5 relevant: `score_pool()` still
rank-averages relevance with citation/author/venue impact and TF-IDF diversity, and DiffusionBERT lost
out to other foundational papers on those axes even with maxed-out relevance. Two different, equally
valid discrete-diffusion-theory ground-truth papers (Shi et al.'s "Simplified and Generalized Masked
Diffusion," Ou et al.'s "Your Absorbing Discrete Diffusion...") entered in its place. Net effect is
positive (+2 matches, zero off-topic noise), but a paper scoring 5/5 relevance is still not guaranteed a
slot — the natural next fix would be a relevance floor that exempts top-scored papers from
impact/diversity competition entirely, not just weighting them more heavily in the average.

**Why these results live in separate `_llm_judged`-suffixed files** rather than overwriting
`candidates_topK.json` / `reference_brief.txt` / `evaluation_results.json`: the submitted
`output/generated_survey.md` cites references by position (`[1]`-`[48]`) against the *original*
`reference_brief.txt`. Swapping the underlying reference set would silently desync every citation marker
in the already-written survey prose without redoing generation against the new list. This extension is
additive evidence that the flagged fix works, not a replacement of the submitted deliverable — the
correct way to fully adopt it would be regenerating the survey text from `reference_brief_llm_judged.txt`
end to end, which is future work, not done here.

## 11. Second methodology: implementing the evaluation framework of arXiv:2412.13612

The assignment offered **two** papers. §1–§10 implement QUAL-SG ([2508.17647](https://arxiv.org/abs/2508.17647))
as the *generation* method. This section implements the other one —
*LLMs for Automated Literature Review: An Evaluation of Reference Generation, Abstract Writing, and Review
Composition* ([2412.13612](https://arxiv.org/abs/2412.13612)) — as an *evaluation* framework, and applies it
to the survey QUAL-SG produced. The assignment is therefore generate-**and**-evaluate rather than
generate-only, using both provided papers.

Code: [`scripts/eval_framework_2412.py`](scripts/eval_framework_2412.py) ·
Results: [`output/eval_framework_2412.json`](output/eval_framework_2412.json)

### 11.1 Task mapping

| Their task | Their setting | Our analogue |
|---|---|---|
| 1. Reference generation | LLM asked to produce references | the 48 retrieved + ranked references |
| 2. Abstract writing | LLM summarises a paper | the survey's §1 Introduction vs the anchor's §5 |
| 3. Review composition | LLM writes a full review | the full survey vs the anchor's §5 |

### 11.2 Faithful vs substituted

| Metric | Paper | This implementation |
|---|---|---|
| Reference accuracy rule | `True(r) = 1 if (title correct AND ≥1 other field) OR (title incorrect AND ≥3 other fields)`, 80% title match | **faithful** |
| Title Search Rate | retrievability from Semantic Scholar | **source substituted** → OpenAlex + arXiv (see §11.4) |
| Semantic similarity | cosine over `text-embedding-3-large` | **substituted** → pure-python TF-IDF cosine |
| ROUGE-1 / -2 / -L | n-gram + LCS overlap | **faithful** |
| NLI entailment | TRUE model / GPT-4o | **not implemented** — a lexical coverage proxy is reported and explicitly labelled as not NLI |

### 11.3 Results

**Task 1 — reference generation** (47 scored; 1 excluded as an infrastructure failure)

| Metric | Value |
|---|---|
| Title Search Rate | **1.000** |
| Reference accuracy | **1.000** |
| Hallucination rate | **0.000** |
| verified via OpenAlex | 40 |
| verified via arXiv fallback | 7 |

**Tasks 2 and 3 — content quality against the anchor's §5**

| Metric | Task 2 (Introduction) | Task 3 (full survey) |
|---|---|---|
| Semantic similarity (TF-IDF) | 0.1874 | 0.2209 |
| ROUGE-1 | **0.3570** | 0.1866 |
| ROUGE-2 | 0.0664 | 0.0581 |
| ROUGE-L | 0.1172 | 0.0921 |
| Lexical coverage proxy | 0.3039 | **0.7052** |

The split is informative and validates separating the two tasks. The **Introduction** scores far higher on
ROUGE-1 (0.357 vs 0.187) because it is length-comparable to the anchor's §5 — both are condensed overviews.
The **full survey** scores far higher on coverage (0.705 vs 0.304) because it is ~5× longer and subsumes
most of §5's content. Neither number alone characterises the survey; together they say it covers the ground
truth thoroughly while its condensed section reads most like it.

### 11.4 Two bugs found while building this — both would have produced confident wrong numbers

**(a) Silent API failure reported as hallucination.** The first version used Semantic Scholar's search
endpoint and treated *any* failed lookup as "reference not found." Unauthenticated S2 returns **HTTP 429**
almost immediately, so it reported NOT FOUND for papers as well known as *Diffusion-LM* and would have
yielded a hallucination rate near **1.0** — catastrophically wrong, and wrong in the direction that looks
like an exciting finding.

*Fixes:* default to OpenAlex (used keylessly elsewhere in this project, tolerant at this volume), and
distinguish `LOOKUP_FAILED` from `NOT_FOUND`, excluding infrastructure failures from the denominator so
they can never masquerade as hallucinated references.

**(b) An unrelated best-match counted as a successful find.** OpenAlex returns *something* for almost any
query. Six genuine 2025-2026 arXiv preprints came back with best-match scores of 0.18–0.33 — i.e. OpenAlex
returned a different paper entirely — and the code counted these as `FOUND`, producing a spurious **12.8%
hallucination rate**.

The cause is the same OpenAlex preprint lag that §8 already documents for the retrieval stage. *Fix:* treat
a below-threshold best match as `NOT_FOUND`, and add an **arXiv-by-ID fallback** for references carrying an
`arxiv_id`. Seven references were recovered this way, and the hallucination rate went 0.128 → **0.000**.

### 11.5 What Task 1 does and does not show

**It is close to tautological here, and should be reported as such.** The paper's references are
*generated* by an LLM, so hallucination is the failure mode under test. Ours are *retrieved* from OpenAlex
and arXiv — and Task 1 then verifies them against OpenAlex and arXiv. A perfect score confirms the
retrieval plumbing returns what it fetched; it is **not** evidence that an LLM avoided hallucinating.

The result is still worth reporting for two reasons: it independently reproduces §7's finding that all 48
references are real, and building it surfaced two measurement bugs that would each have produced a
confident wrong number. A genuinely independent check would verify against a database the pipeline never
queried.
