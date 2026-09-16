# Citation Agent — grounding verdicts

Retrieval: OpenAlex (8 queries, keyless, full results) + arXiv (blocked by HTTP 429 after the
OpenAlex+arXiv combined run — arXiv rate-limits aggressively and the script's query volume tripped
it; documented here rather than silently retried until it worked, per this project's own standard
of recording real infra constraints rather than hiding them).

**A1 (annealing at scale):** No paper found directly testing attention-mask annealing (or an
equivalent causal-to-bidirectional attention transition) at 7B-parameter scale for full-sequence
diffusion adaptation. One contextually relevant hit: **"Large Language Diffusion Models"
(LLaDA, 2025)** — trained from scratch rather than adapted from an AR checkpoint, which sidesteps
the annealing question entirely rather than answering it. Verdict: **gap confirmed by absence** —
9 of 10 retrieved results were topically unrelated (antibody LMs, smart grids, time-series
forecasting, prompt-engineering surveys), and the one relevant model in the field takes a different
approach that doesn't test this specific claim either.

**A8 (adaptation generalization beyond GPT2/LLaMA2):** Same LLaDA hit is the most relevant result —
confirms the field has a real *from-scratch* alternative, but does not test whether DiffuLLaMA's
specific adaptation recipe transfers to other AR backbones. Verdict: **gap confirmed**, LLaDA cited
as the contextual alternative, not as evidence closing the gap.

**A6 (instruction tuning for diffusion LMs) and A4 (inference-efficiency/compute-parity):**
Retrieval returned no genuinely relevant results for either (image-domain instruction tuning,
generic LLM-inference-efficiency surveys, unrelated domains). Given OpenAlex's demonstrated
weakness at this narrow a topic (the same problem litreview/REPORT.md already documents for the
SOTA assignment's own retrieval), a null result here is inconclusive rather than confirmatory —
recorded as such. Both A4 and A6 remain grounded directly in the anchor paper's own text regardless
(the paper states its own uncertainty diagnosis for A6, and never reports FLOPs/forward-pass counts
for A4), so external grounding was supplementary, not load-bearing, for either.

**Limitation of this pass, stated plainly:** query phrasing sensitivity is real — a prior, differently-
phrased search (outside this assignment's scope) previously surfaced a directly relevant paper
(NBDiff, testing annealing-like ablations at 7B for a different diffusion target) that this pass's
queries did not reproduce. Retrieval recall depends heavily on exact phrasing, a limitation of this
implementation worth stating rather than glossing over.
