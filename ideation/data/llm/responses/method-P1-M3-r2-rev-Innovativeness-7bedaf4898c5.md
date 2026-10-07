**Review:**

TARAC proposes token-level adaptive refinement for diffusion language models, using an AR scorer as a confidence monitor to allocate extra denoising steps selectively to uncertain tokens. The method is well-structured, clearly articulated, and respects the inference-only constraint. The experimental design—comparing TARAC against uniform-S, uniform-N, and standard reranking baselines with compute-matched budgets and bootstrap significance testing—is rigorous and directly addresses the research problem.

However, the innovativeness is **incremental rather than transformative**. The core idea—using uncertainty estimates to allocate compute—is a well-established principle in ML (early exiting, adaptive computation, curriculum learning). The specific instantiation (per-token AR confidence → selective diffusion refinement) is a novel *configuration*, but it builds on existing components (AR scoring as in TESS 2, adaptive scheduling as in Jacobi Forcing, masking strategies common in diffusion) without introducing a fundamentally new mechanism or theoretical insight. The token-wise masking within a diffusion trajectory, while clever, raises implementation questions (how partial updates interact with the denoising dynamics) that are under-explained. The method's differentiation from prior inference-time guidance approaches (TESS 2 reward guidance, Jacobi Forcing) could be stated more sharply.

**Feedback:**

1. **Sharpen the novelty claim:** Explicitly contrast TARAC with TESS 2's reward guidance and Jacobi Forcing's trajectory distillation—clarify what is fundamentally different (scheduling vs. guidance vs. distillation).
2. **Address implementation feasibility:** Explain how token-wise selective refinement interacts with the diffusion sampler's latent dynamics; provide ablation evidence that partial updates don't degrade output quality.
3. **Validate the confidence signal:** Include an analysis showing that AR confidence on the base output actually correlates with token-level error rates, justifying the adaptive allocation strategy.
4. **Strengthen the baseline:** The "uniform-S" and "uniform-N" baselines are simplistic; include Jacobi Forcing and TESS 2 as direct competitors to better position the contribution.
5. **Consider theoretical grounding:** A brief analysis of *why* token-level adaptation should help (e.g., error propagation, token dependency structure) would elevate the method beyond a heuristic.

**Rating (1-5): 3**