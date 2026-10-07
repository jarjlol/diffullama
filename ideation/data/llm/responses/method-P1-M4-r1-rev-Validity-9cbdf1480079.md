Review:
The proposed ATAR method directly targets the two identified gaps (accuracy headroom L6 and compute-normalization L9) with a conceptually sound approach: using AR-derived token-wise entropy to focus diffusion refinement on uncertain tokens. The overall structure—model preparation, unified quality metric, compute-normalized efficiency, iterative refinement, Pareto analysis, and significance testing—is well-organized and largely feasible. However, several validity concerns undermine confidence in the method's correctness:

1. **FLOP savings from token-wise activity are likely overestimated.** Standard diffusion forward passes compute over all tokens regardless of an attention mask; masking prevents information flow but does not skip computation. Unless sparse operations are explicitly implemented, the claimed FLOP reduction proportional to the active-token fraction is inaccurate, invalidating the compute-normalized comparison.

2. **The conditional partial denoising step is not correctly specified.** Setting inactive tokens to clean embeddings and running a standard diffusion step does not correspond to the correct conditional distribution q(x_active | x_inactive, x_0). Proper treatment requires adjusting the noise schedule or using inpainting-specific sampling, which is non-trivial for discrete diffusion.

3. **AR-model entropy may misalign with diffusion-model uncertainty.** The frozen AR scorer's token-level entropy is used as a proxy for where the diffusion model needs refinement, but these two models have different inductive biases and failure modes. This proxy validity is assumed but not empirically justified.

4. **Fixed vs. adaptive masking trade-off is unresolved.** Keeping the mask fixed after round 1 means high-confidence tokens are never re-evaluated, potentially missing later-stage refinement opportunities. Recomputing the mask every k rounds introduces inconsistency. Neither option is clearly justified.

5. **AR counterparts for all five DLMs are not established.** Dream-7B, LLaDA-8B, and DiffuCoder-7B do not have obvious AR base models of the same architecture, yet the method assumes one per DLM.

6. **UQM's min-max normalization is outlier-sensitive**, and equal weighting across heterogeneous benchmarks may mask domain-specific effects despite per-benchmark reporting.

The method is innovative and the research question is well-framed, but the technical correctness issues—particularly around FLOP accounting and conditional diffusion sampling—must be resolved before the results can be considered valid.

Feedback:
- Revise the FLOP model to account for actual computation (full forward pass over all tokens even with masking) or implement sparse/selective computation that genuinely skips inactive tokens.
- Derive the correct conditional sampling procedure for partial denoising in discrete diffusion, or justify why the simplified approach is a valid approximation.
- Validate that AR-model entropy correlates with diffusion-model uncertainty on a small pilot dataset before committing to the full pipeline.
- Establish clear AR counterparts for Dream, LLaDA, and DiffuCoder, or exclude them and justify the reduced model set.
- Consider robust normalization (e.g., rank-based) for UQM and justify the equal-weighting scheme or use task-specific weighting.
- Specify the gradient-based guidance baseline precisely (method, λ range, selection criterion).

Rating (1-5): 3