**Review:**

The method is exceptionally well-organized, presented in a structured table format that cleanly separates goals, concrete actions, and resource constraints. Each step contains explicit mathematical formulas (preservation scores, gain ratios, efficiency metrics), precise hyperparameter values, and reproducibility measures (fixed seeds, reference sets, environment archiving). The rationale section thoughtfully addresses confounds (scorer-model family matching), statistical validity (bootstrap at prompt level, descriptive framing for n=5), and construct validation (behavioral AR-likelihood check). The ablation plan is comprehensive and the timeline is realistic.

However, several minor ambiguities prevent a top-tier clarity rating:

1. **Layer indexing**: "ℓ=1 is the first layer" conflicts with standard 0-indexed PyTorch implementations — this could cause off-by-one errors in hook registration.
2. **"Linear kernel CKA"**: This reduces to centered bilinear correlation; the description should clarify this equivalence or justify the kernel formulation.
3. **Probing corpus split**: "10,000 sentences from WikiText-103 + C4 combined" doesn't specify the per-corpus proportion or whether analyses are stratified.
4. **Normalization reference set**: "First 10% of prompts" is ambiguous for benchmarks with indivisible counts (e.g., HumanEval has 164 prompts → 16.4). Rounding method should be specified.
5. **Scorer ambiguity for LLaMA-based DLMs**: "LLaMA-7B (or a distilled 1B version if memory is tight)" introduces uncertainty about the primary scorer choice.
6. **Curve fitting with 4 data points**: Fitting smooth monotonic curves to only 4 (step, UQS) pairs per DLM is underdetermined — the sensitivity of results to functional form choice should be more explicitly acknowledged.
7. **Normalization data leakage risk**: Using benchmark prompts for both the reference set and evaluation should clarify that the reference set is excluded from all experimental conditions.

**Feedback:**

Clarify the layer indexing convention (0- vs. 1-based) and specify exact hook positions in terms of module names (e.g., `transformer.h.{ℓ}.ln_1` post-attention). Replace "linear kernel CKA" with explicit notation showing it reduces to centered correlation. Fix the normalization reference set to use integer rounding (e.g., floor) and confirm it is disjoint from experimental prompts. Specify the exact LLaMA scorer variant used as primary. Add a sensitivity analysis note for the curve-fitting step, comparing at least two functional forms. Finally, explicitly state the probing corpus split ratio between WikiText-103 and C4.

**Rating (1-5): 4**