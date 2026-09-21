# REPORT — Multi-Agent Limitation Generation applied to DiffuLLaMA

Implementation report for **arXiv:2601.11578**, *Multi-Agent LLMs for Generating Research
Limitations* (Al Azher, Guo, Alhoori; v1 30 Dec 2025, v2 16 Mar 2026), applied to the anchor
paper **DiffuLLaMA** (Gong et al., ICLR 2025).

---

## 1. What the source paper proposes

Zero-shot LLM prompting produces superficial limitation statements — the paper's own examples
are "dataset bias" and "generalizability" — and tends to echo whatever the authors already
disclosed. The paper's response is a multi-agent decomposition in which separate agents take
separate stances toward the paper, a Judge scores them, a Self-Feedback agent regenerates the
weak ones, and a Master agent consolidates. It also replaces n-gram/embedding evaluation with a
pointwise LLM-as-a-Judge coverage protocol, on the grounds that BLEU/ROUGE/cosine miss
semantically equivalent limitations phrased differently.

Reported results: +15.51 coverage points over zero-shot for GPT-4o mini with RAG and four
agents, +4.41 for Llama 3 8B with three agents, over 2,700 NeurIPS papers.

## 2. What was implemented

Everything in the generation pipeline, in the paper's order. Concretely:

- **Extractor** — author-stated limitations only, "chain of limitations" methodology, input is
  the paper with Conclusion/Limitations sections removed.
- **Analyzer** — methodological auditor for implicit gaps, full paper as input.
- **Reviewer** — simulated peer reviewer, weighing reproducibility, transparency, ethics.
- **Citation** — limitations grounded in retrieved related work, RAG corpus as extra input.
- **Judge** — the paper's exact rubric and weights (Depth 0.2, Originality 0.2, Actionability
  0.3, Topic Coverage 0.3), 0–10 per dimension, scaled to 0–100.
- **Self-Feedback** — regeneration for any agent below 8/10, capped at two retries.
- **Master** — clustering, merging, provenance tagging.
- **Evaluation** — pointwise binary-similarity coverage, `C_GT`, against a zero-shot baseline.

The agent prompts in `data/agent_requests/` are the literal prompts run, transcribed from the
paper's descriptions and keeping its quoted phrasing where it gives any.

## 3. Deviations from the source paper

Every deviation is listed. None is silent.

### 3.1 The LLM is external to the scripts

This build environment has no hosted LLM API, the same constraint `../litreview/` already
documents. Mechanical stages (retrieval, fusion, clustering, judge arithmetic, coverage
computation) run as code. Stages requiring a model call are executed by Claude (Opus 5) against
the exact prompt the script emits, with the output checked in as a data file that the next stage
reads. This is honest but it is a real deviation: it means the agent outputs are not
independently reproducible by re-running the scripts, only re-verifiable against the prompts and
inputs, which are all committed.

### 3.2 FAISS replaced by TF-IDF cosine — the largest fidelity gap

The paper's hybrid retriever is BM25 (sparse) + FAISS (dense), equally weighted. BM25 is
implemented exactly. The dense half is a hand-written TF-IDF cosine ranker, because no embedding
model or vector index is installable here.

**This substitution demonstrably cost something, and the cost is measurable in this run.** The
Master Agent's clustering stage failed to group `EXT-6` ("annealing omitted at 7B … minimal
impact … flash-attention 2") with `ANA-1` ("the ablation shows +2.1 and +2.5, the benefit grows
with scale"). Both are about attention-mask annealing; they share almost no vocabulary, so
lexical cosine scored them below threshold. A real dense retriever would very likely have caught
it. The LLM merge step did catch it, and the miss is recorded in `data/master_merged.json`
rather than papered over.

### 3.3 `X_by` (Cited By) substituted

The paper obtains citing papers from the OpenAlex API. The anchor's OpenAlex record
(`W4404307915`) reports `cited_by_count = 0`, and a search confirms only one record exists for
this title — an indexing gap, since an ICLR 2025 paper of this profile is certainly cited. The
`cites:` filter therefore returned nothing.

Rather than drop half the corpus, `X_by` is substituted with an OpenAlex full-text search for
works mentioning "DiffuLLaMA" (47 works). This serves the paper's stated purpose for `X_by` —
surrounding literature that exposes contextual weaknesses — but it is a **mention graph, not a
citation graph**, and is labelled `cited_by_substitute` throughout the data files.

### 3.4 OpenReview ground truth unavailable

The source paper's central contribution to ground-truth quality is merging author-stated
limitations with weaknesses mined from OpenReview reviewer comments (10–12 per paper). DiffuLLaMA
is an ICLR 2025 Poster with a public forum (`j1tSLYKwg8`), so these reviews exist — but
`api2.openreview.net` returns **HTTP 403** on all forum queries from this environment, and the
forum page itself is a JavaScript shell with no server-rendered review text.

This is an **access failure, not an absence**, and is recorded as such in
`data/ground_truth.json`. Consequences are in §6.2.

### 3.5 Scope

The paper evaluates over 2,700 papers. This is a single-paper application, so the paper's
aggregate claims are not reproduced — only its method, and its method's behaviour on one input.

## 4. Why LLM steps are externalised rather than stubbed

A stub that returns plausible text would make the pipeline "run" while producing results that
mean nothing. The precedent in this repository is `../litreview/`, which documents the same
constraint. Every LLM artifact here carries `_protocol`, `_generated_by`, and the rubric it was
produced under, so a reader can check the output against the prompt that produced it.

## 5. Results

### 5.1 Retrieval

| Stage | Count |
|---|---|
| `X_in` chunks (from `litreview/candidates_topK.json`) | 48 |
| `X_by` chunks (OpenAlex mention search) | 47 |
| `C_total` | 95 |
| After hybrid BM25+dense, top-K | 20 |
| After LLM re-rank at ≥8/10 | **13** |

The retained set is dominated by adaptation-recipe papers — *Don't Retrain, Align* (10/10),
*PreDiff-LM* (10/10), *UNIFUSION* (9/10), *TESS 2* (9/10) — which is the correct neighbourhood
for finding weaknesses in an adaptation paper.

### 5.2 Agents and the feedback loop

| Agent | Round 1 | Verdict | Round 2 |
|---|---|---|---|
| Extractor | 72.0 | REGENERATE | **84.0** |
| Analyzer | 87.0 | pass | — |
| Reviewer | 78.0 | REGENERATE | **86.0** |
| Citation | 87.0 | pass | — |

The Self-Feedback mechanism **actually fired**, for half the agents, and both recovered above
threshold on one retry. The Extractor's originality score stays capped at 6 by construction: an
agent restricted to what authors wrote cannot be original about it.

### 5.3 Consolidation

33 raw limitations → **12** consolidated, with **no item dropped** (asserted in code). Ten of the
twelve carry more than one provenance tag, meaning they were reached independently by more than
one agent — the multi-agent structure's clearest payoff, and something a single-pass prompt
cannot produce by construction.

### 5.4 Coverage evaluation

| System | `C_GT` | Matched | Items | Generic statements |
|---|---|---|---|---|
| Zero-shot | 0.583 | 7/12 | 10 | **3** |
| Multi-agent | 1.000 | 12/12 | 12 | **0** |

## 6. Honest limitations of this implementation

### 6.1 The clustering substitution cost a real merge

See §3.2. One cross-agent merge was missed by the lexical clusterer and recovered only by the LLM
step. On a larger input set, misses of that kind would accumulate silently.

### 6.2 The +41.67 coverage gain is inflated and is not a result

The measured gain is far larger than the paper's own +15.51, and that difference is an artifact,
not an improvement. Ground truth here is **author-stated limitations only**, because the
OpenReview half was unreachable (§3.4). The Extractor Agent's entire job is to recover
author-stated limitations, so a pipeline containing an Extractor scores near-perfectly on this
ground truth by construction. **`C_GT = 1.000` should be read as confirming the tautology, not as
evidence of quality.**

This is the same failure mode this project already recorded in `../litreview/REPORT.md` §11.5,
where reference-accuracy scoring was near-tautological because references were retrieved and then
verified against the database they came from. It recurred here for a different reason, and is
flagged rather than reported as a win.

**The one non-tautological signal in the evaluation** is the generic-statement count: the
zero-shot baseline produced three statements of exactly the kind the source paper's abstract
names as the failure mode ("dataset bias", "may not generalize", "generalizability"), and the
multi-agent output produced none. That comparison does not depend on the ground-truth
composition.

### 6.3 Single judge, no human validation

The source paper validates its LLM judge against two human annotators (agreement 0.98 and 0.95).
No human validation was performed here. The same model acts as worker agents, judge, re-ranker,
and matcher, so judge scores are not independent of the outputs they grade — a self-evaluation
bias the source paper avoids by design and this implementation does not.

### 6.4 Provenance labels are asserted, not verified

Each limitation's provenance tag comes from the agent that produced it. Nothing independently
verifies that an "Author-stated" item is genuinely in the paper's text. The `evidence` field on
every item exists so this can be spot-checked, and the quoted passages were checked against
`litreview/data/anchor_fulltext.txt` during authoring, but no automated verifier enforces it.

## 7. Relationship to the rest of this repository

This directory replaces an earlier `limitations/` implementation that blended techniques from
three papers (BAGELS, LimitGen, and this one) and was deliberately built **without** using
`litreview/`. This version implements one paper faithfully and **does** take `litreview/` as
input, per the assignment brief's instruction that the surveyed literature should feed the
limitation-extraction task. The `data/` directory here is self-contained; nothing is read from
`project-docs/`.
