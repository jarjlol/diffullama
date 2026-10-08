**Review:**

The ATAR method presents a well-structured and thoughtfully motivated approach to improving the quality–compute trade-off in diffusion language models. Its central insight — using AR-model entropy to identify uncertain tokens and selectively focusing denoising computation on those tokens — is logically sound and experimentally tractable. The method is clearly described, the algorithm is precise, and the experimental design (Pareto fronts, bootstrap significance testing, generalization checks) is rigorous.

However, assessed strictly on **innovativeness**, the method falls into the moderate category. Each individual component — entropy-based uncertainty estimation, attention masking for partial denoising, candidate generation with AR reranking, and compute reallocation — draws on established techniques from the very literature cited (Jacobi Forcing's selective token processing, TESS 2's reward guidance, PreDiff-LM's hybrid attention masks, and the broader use of AR scorers). The specific combination is coherent and well-justified, but it does not introduce a fundamentally new mechanism or paradigm. The innovation is primarily in the **systematic framework and empirical rigor** rather than in a novel technical contribution. A theoretical analysis (e.g., why token-wise adaptive refinement should provably outperform uniform refinement) or a comparison with alternative adaptive strategies (gradient-based uncertainty, margin-based selection) would strengthen the novelty claim considerably.

**Feedback:**

1. **Strengthen the novelty argument**: Explicitly articulate what is *not* known from existing work that ATAR contributes. Currently, the rationale frames ATAR as attacking the headroom "from a different angle," but this angle (entropy-guided selective denoising) is conceptually close to Jacobi Forcing's selective decoding and TESS 2's guidance. Distinguish ATAR more sharply — e.g., by showing that AR entropy captures a different signal than reward guidance or trajectory-based selection.

2. **Add a theoretical or empirical ablation that isolates the key mechanism**: Demonstrate that the token-wise masking itself (not just the candidate diversity or additional denoising steps) drives the improvement. For instance, compare ATAR with a "shuffled mask" baseline where the same fraction of tokens is masked randomly — if ATAR still wins, this validates the adaptivity as the source of gain.

3. **Deepen the generalization checks**: Testing alternative scorers is good, but also test whether the AR model's entropy is actually informative — e.g., correlate AR entropy with diffusion model's own uncertainty estimates (if available) or with final token correctness. This would validate the core assumption rather than just assuming it.

4. **Consider a more nuanced compute model**: The FLOP approximation treats all tokens equally, but in practice, attention to inactive tokens still incurs some cost (masked attention is not free). A more realistic model would strengthen the compute-normalization claim.

5. **Address the gradient-based guidance baseline more thoroughly**: The method mentions repeating the pipeline with λ > 0 guidance, but doesn't specify how guidance interacts with the entropy masking. This interaction needs careful treatment to avoid conflating two forms of guidance.

Rating (1-5): 3