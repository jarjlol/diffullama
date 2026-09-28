# Judge + Master Agent — scoring, taxonomy, provenance, consolidation

Per the Multi-Agent paper's Judge role (score depth/originality/actionability/groundedness, filter
below threshold) and Master role (dedupe, merge, tag provenance). Taxonomy: LimitGen's four aspects
(arXiv:2507.02694 Table 1).

## Merges (Extractor + Analyzer items describing the same underlying limitation)

- **E4 + A6** → one item: the paper explicitly defers instruction tuning (E4) *and* independently
  reports that this exact absence measurably hurts CoT performance (A6) — two angles on one gap.
- **E5 + A3** → one item: the paper explicitly admits the mask-token treatment is a workaround (E5)
  *and* that workaround is inconsistently applied in a way that confounds with model size and best
  performance (A3) — cause and unexamined consequence of the same design choice.

## Filtered out (Judge score below threshold)

- **E1** (sampled rather than integral loss, for efficiency) — standard Monte Carlo practice for
  this objective family, near-universal in the diffusion literature; low depth/originality as a
  "limitation," more a routine implementation detail than a genuine gap. Dropped.

## Final 10 items — taxonomy + provenance

| # | Item | LimitGen aspect | Provenance |
|---|---|---|---|
| L1 | A1: Annealing ablation contradicts its own "minimal impact" justification | Experimental Design → Lack of Ablation Studies | Inferred (from paper's own table) |
| L2 | A2: Shift-operation ablation is ~5x larger than annealing's, examined far less | Experimental Design → Lack of Ablation Studies | Inferred |
| L3 | E5+A3: Mask-token treatment is an admitted workaround, confounded with model size | Methodological → Inappropriate Method | Author-stated + Inferred |
| L4 | E4+A6: Instruction tuning deferred despite diagnosed, demonstrated need | Experimental Design → Insufficient Baselines | Author-stated + Inferred |
| L5 | E3: Infilling comparison self-flagged as potentially unfair | Experimental Design → Inappropriate Datasets | Author-stated |
| L6 | E2: Model reported as undertrained, no convergence shown | Experimental Design → Limited Datasets (training budget) | Author-stated |
| L7 | A4: No compute/FLOPs parity accounting despite no KV-cache | Result Analysis → Insufficient Metrics | Inferred, literature-grounding inconclusive |
| L8 | A5: High answer uncertainty reported, not investigated further | Result Analysis → Limited Analysis | Inferred |
| L9 | A7: "Code generation" claimed; only fill-in-the-middle infilling evaluated | Literature Review → Inaccurate Description (claim vs. evaluated scope) | Inferred |
| L10 | A8: Validated only on GPT2/LLaMA2 families, up to 7B | Literature Review → Limited Scope | Inferred, citation-grounded (LLaDA as contextual alternative) |

Ranked by Judge-assessed strength (groundedness x actionability x depth): **L1 > L4 > L3 > L2 > L10
> L8 > L5 > L7 > L9 > L6.** L1 (annealing) is the strongest single item: quantitatively grounded in
the paper's own table, directly contradicts the paper's own stated justification, is independently
citation-checked (no external paper found testing it at the relevant scale), and is directly
actionable (run the missing ablation).
