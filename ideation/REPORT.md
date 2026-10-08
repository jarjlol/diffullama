# REPORT — ResearchAgent for research problem discovery and ideation

## 1. What the paper proposes

Verified from the PDF (arXiv:2404.07738, all quotations below from the extracted text).

ResearchAgent generates a complete research idea — **problem → method → experiment design** — as
three chained LLM calls over a *target paper*, *related papers* ("studies that have cited the target
paper", narrowed "based on their similarities of abstracts with the core paper"), and *entities*
retrieved from an entity-centric knowledge store. The store is a co-occurrence matrix over entities
extracted from titles and abstracts; retrieval takes the top-k entities absent from the input papers
by Eq. 2:

> argmax over sets I of size k of  ∏ ( ∏_{e_j ∈ E_input} P(e_j | e_i) ) × P(e_i)

Each generated artifact is evaluated by **ReviewingAgents**, prompted once per criterion, on five
criteria per stage with a 5-point Likert scale:

| Stage | Criteria |
|---|---|
| Problem | Clarity, Relevance, Originality, Feasibility, Significance |
| Method | Clarity, Validity, Rigorousness, Innovativeness, Generalizability |
| Experiment | Clarity, Validity, Robustness, Feasibility, Reproducibility |

The criteria are aligned to human judgement by inducing detailed 5-level rubrics from 10
human-annotated ideas (Tables 13–15). The ResearchAgent refines each artifact from the reviews;
the paper reports gains saturating after three iterations.

## 2. What is implemented

| Paper component | Implementation |
|---|---|
| Problem / method / experiment prompts (Tables 6–8) | **verbatim**, `scripts/templates.py` part 1 |
| ReviewingAgent prompts (Tables 9–11) | **verbatim**, one call per criterion, as the `{metric}` slot requires |
| Criteria definitions (Table 12) + induced rubrics (Tables 13–15) | **verbatim**, inserted as `{criteria}` |
| Iterative refinement, 3 rounds | `scripts/research_agent.py`, `refinement_rounds: 3` |
| Target + related papers, similarity-narrowed | `scripts/build_literature.py` |
| Knowledge store K and Eq. 2 retrieval | `scripts/build_entity_store.py`, log space, add-α smoothing |
| GPT-4 for every role | pluggable backend, `scripts/llm.py` |

## 3. Deviations from the paper

Each is deliberate and stated; D1–D4 can be switched off in `config.json` or `templates.py` to run
the paper's prompts unmodified.

**D1 — Gaps as input.** The assignment requires problems to come from the prior assignment's gaps;
ResearchAgent has no slot for them. The twelve gaps, each with its assessment from the dismissal
analysis, are added as one block in the problem prompt. The block states the generator need not
adopt any single gap. *Consequence:* problems are conditioned on our own critique of the anchor,
which the paper's problems are not.

**D2 — Resource constraints.** The Feasibility rubrics refer to "the available resources", which a
ReviewingAgent cannot otherwise know. The project's real constraints (inference only, one contended
96 GB GPU, ~10 weeks, 7 people / 3 with GPU access) are given identically to generators and to every
Feasibility review.

**D3 — Multiple candidates.** The paper produces one idea per target paper. The assignment needs
several candidate problems and several ideas per problem to rank a top 5. Each new candidate is shown
the earlier candidates' first drafts and asked for a substantively different one. Diversity keys on
first drafts only, so candidates can then be refined in parallel.

**D4 — Refinement prompt.** The paper describes refinement from reviews but publishes no template.
Ours appends the previous draft and all five reviews to the paper's own generation prompt.

**D5 — Related papers are a mention set, not a citation set.** OpenAlex indexes zero works citing
DiffuLLaMA (verified during the limitations assignment; one record exists and it has no citation
edges). The citing side is approximated by the 41 unique works that mention DiffuLLaMA by name,
from the limitations assignment's OpenAlex search. These are *about* the target in the way citing
papers are, but inclusion is by mention, not by citation link.

**D6 — Entity extraction.** The paper uses the BLINK neural entity linker, which cannot be installed
under the stdlib-only rule. A deterministic rule-based extractor takes acronyms and mixed-case
technical names, plus 2–3-word noun phrases recurring in at least two papers. It yields ~12
entities per paper against BLINK's reported ~3, with no Wikipedia linking — so it captures field
vocabulary, not encyclopedic concepts. Three filter passes were needed to remove clause fragments
("text through", "have recently", "generation discrete"); some noise likely remains.

**D7 — Unstated parameters.** The paper does not state how many related papers or entities (k) it
uses. Defaults: 10 and 10, keeping the first prompt at ~7.3k tokens.

**D8 — No criteria re-induction.** The paper induces its rubrics from 10 human-annotated ideas. We
use the paper's published induced rubrics as-is rather than re-inducing them from our own
annotations. Re-induction would need 10 ideas annotated by team members with ≥3 papers each, which
the paper requires of its annotators.

**D9 — Ranking and the assignment's criteria.** The paper evaluates; it does not rank a top-k.
Ranking uses the mean of the ten final method + experiment ratings — the paper's own evaluation
quantity. The assignment's criteria are reported as columns mapped onto the paper's, not scored by
an extra judge: novelty ← Innovativeness, feasibility ← experiment Feasibility, clarity ← the two
Clarity ratings, impact ← Generalizability. Problem Significance is shared by every idea for one
problem, so it cannot separate them and is reported once, with the problem.

**D10 — Human selection gate.** Between stages 3 and 5 the team picks the problems. The paper chains
problem → method directly. The assignment asks for "one or two" problems, and choosing them is a team
decision, not something to delegate to the reviewer's scores.

## 4. Cost

At the default configuration, **864 model calls**: each artifact costs (R+1) generations + 5(R+1)
reviews = 24 at R = 3; 4 problem candidates, then 8 methods and 8 experiments for each of 2
problems. The per-criterion review structure is the paper's, and it dominates the count.

Knobs, all in `config.json`: `refinement_rounds` (R=1 → 432 calls total), `n_ideas_per_problem`
(6 → 672), `n_problem_candidates`. Reducing R departs from the paper's saturation finding and
should be reported if used. Manual mode at this volume is impractical; an API key is effectively
required for a full run.

**Token volume**, measured on a full mock run (chars ÷ 4): ~4.0M input tokens — problem prompts ~4.4–7.4k,
method and experiment prompts ~4.5–4.8k each. This is a floor: mock drafts are a few words, while real
refinement prompts carry the previous draft plus five reviews, so expect roughly 5–6M. Output volume is
small by comparison. Backend options and their trade-offs are in `README.md` and decision-log P-7; no
backend has been chosen yet.

**Run note (2026-10-07, P1+P2+P3 at paper settings = 1,248 idea calls).**
Backend is OpenRouter (`--backend openai` with `LLM_BASE_URL=https://openrouter.ai/api/v1`;
no code change needed — see `SERVER_RUNBOOK.md`). Generator:
`nvidia/nemotron-3-super-120b-a12b:free`. Reviewer:
`inclusionai/ling-3.0-flash-sante:free`, after `google/gemma-4-31b-it:free`
(persistent upstream congestion), `nvidia/nemotron-3.5-lightning:free` (~7
min/call, thinking trace) and `inclusionai/ling-3.0-flash-fin:free` (free slug
retired mid-run, HTTP 404) each failed. Consequence: problem-stage reviews are
a Ling-fin/Sante mix, but every method/experiment rating feeding the ranking
(D9) comes from one reviewer model, so within-stage comparisons hold.
`scripts/llm.py` additionally supports comma-separated `LLM_API_KEYS` /
`REVIEWER_API_KEYS` with rotation on 429/401/402 and a persisted index
(`data/llm/key_state.json`, indices only — never keys). Free-tier limits look
per-account (~200 req/day across keys), not per-key. Parser hardened without
changing strictness: markdown/numbered/titled section headers, en-dash and
bare `Rating: N` lines; contentless `Problem: :`-style artifacts are rejected.
`selftest.py` stays 10/10. Stochastic-regen warning: deleting a cached response
and re-requesting (temperature 0.7) changes all downstream prompt hashes and
orphans that subtree — valid cache is never deleted.

## 5. Verification

`scripts/selftest.py` runs the full pipeline on a mock backend in a scratch directory. It checks
that the manual backend stops and writes requests, that the gate stops the run, that a full run
completes, that every problem gets 8 ranked ideas and 5 rendered, that a re-run makes zero new
calls, and that the observed call count equals the estimate. **10/10 pass.** The test caught one
real bug before any real run: the gate message crashed when the data directory lived outside the
repository.

Parsers are tested on plain, markdown-bold and split-line outputs, and fail loudly — with the
request id — on a missing or out-of-range rating rather than defaulting a score.

What this does **not** verify: anything about the quality of generated ideas. Mock output is
placeholder text.

## 6. Known risks

- **Self-evaluation bias.** The paper uses one model for generation and review. Running with a
  separate reviewer model (`REVIEWER_*`) is strongly preferable; this project has already had an
  LLM judge rate its own fabricated claim 86/100.
- **Score inflation.** The paper's reviewer prompt explicitly warns against "uniformly high ratings
  (4–5) unless fully justified". If final ratings still cluster at 4–5, treat that as a failed
  review rather than as strong ideas.
- **Novelty is not checked by the method.** The Originality and Innovativeness ratings are the
  reviewer's opinion, with no literature search. Every submitted idea needs a manual novelty check,
  starting with the base paper's own authors' later work — both preemptions this project has
  suffered came from there.
- **The gaps block frames the problems.** D1 passes the team's dismissal of all twelve gaps into
  the prompt, so generated problems inherit that framing.

**Selection change (2026-10-08): P1 + P2 only.** P3 is dropped. The brief asks for "one or two"
research problems, and the gate in `run_pipeline.py` is restored to ≤2 (it had been loosened to
≤4 to admit three). P3 (candidate count × AR verifier) overlaps P2 (noise schedule) as a second
quality–efficiency study, and none of its work had started, so no cache is discarded. Budget falls
from 1,344 to **864 calls**; 210 distinct steps were already cached (96 problem-stage + 114
idea-stage; a further 98 response files are orphaned forks from earlier regenerations), leaving
**654**. Watch point: P2 lists "a simple learned schedule" among its variants; any method that
*trains* a schedule conflicts with the inference-only constraint, which every Feasibility review
is given.

