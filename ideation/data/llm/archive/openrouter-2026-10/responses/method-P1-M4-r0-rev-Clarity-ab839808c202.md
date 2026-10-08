**Review:**

The AGSD method is well-organized and presents a coherent two-stage selective refinement strategy. The overall structure—from model preparation through Pareto construction—is logically sequenced and the mathematical notation is generally appropriate. However, several clarity gaps undermine replicability:

1. **Critical FLOP accounting flaw**: The total FLOP formula counts S₀ + S₁ = S diffusion steps per candidate regardless of masking, which is identical to uniform denoising. This fails to capture the actual compute reduction from selective refinement (only masking a fraction of tokens should reduce FLOPs), making the quality-per-FLOP comparison against uniform baselines misleading. The formula must be revised to reflect that masked-token-only refinement costs less than full-sequence diffusion.

2. **Mechanistic vagueness in Step 2**: The description of "zeroing gradients w.r.t. unmasked positions" is technically imprecise for diffusion sampling. It is unclear whether this means: (a) running the denoising network only on masked positions with cross-attention to frozen clean embeddings for unmasked ones, (b) a full forward pass with subsequent selective noise application, or (c) something else. A precise computational graph or pseudocode is needed.

3. **AR scorer overhead understated**: Entropy computation requires a full AR forward pass per candidate (N × prompt_length), which for N=8 and long prompts may exceed the claimed ≤1% overhead. This should be quantified or bounded more carefully.

4. **Ambiguous compute comparison**: It is unclear whether AGSD and the uniform baseline are compared at equal total FLOPs or equal wall-clock time. If FLOPs are equal (both use S steps), the advantage must come purely from better allocation—but the FLOP formula doesn't reflect any savings, muddying the interpretation.

5. **Vague references**: "a 3B-parameter model from the dLLM zoo" and "DiffuGPT-S/M" (without specifying exact checkpoints) reduce replicability.

6. **Statistical methodology gaps**: Piecewise-linear regression details (breakpoint selection, confidence intervals) and bootstrap resampling procedure (stratified by prompt?) are underspecified.

**Feedback:**
- Revise the FLOP formula to account for selective refinement savings (e.g., multiply by fraction of masked tokens during S₁ phase).
- Provide a precise algorithmic description or pseudocode for the selective denoising step, clarifying how unmasked token representations are handled during the reverse diffusion process.
- Quantify AR scorer overhead empirically across different N values.
- Explicitly state whether comparisons are FLOP-normalized or latency-normalized, and ensure the formula matches the claimed comparison.
- Specify exact model checkpoint identifiers and bootstrap resampling strategy.

**Rating (1-5):** 3