**Review:**

The TADAG method presents a clear motivation—adapting compute allocation at the token level rather than uniformly across all positions—and the overall algorithmic flow is well-structured with step-by-step procedures and mathematical notation. The connection to the research problem (quality–compute trade-off, accuracy headroom, compute normalization) is explicit and well-argued. The Pareto front construction, marginal gain analysis, and statistical testing framework are described with reasonable precision.

However, several clarity gaps materially affect replicability:

1. **Partial decoding (Step 2b)** is described ambiguously. Diffusion models produce noise-level predictions, not direct token logits for x₀ at intermediate steps. The method for extracting discrete token probabilities from the diffusion model's output at each denoising step is not specified—this is a critical implementation detail that a replicator would need to resolve.

2. **Confidence computation (Step 2d)** uses c = exp(s) where s is a log-likelihood (typically negative). This yields very small values, making the threshold τ unintuitive and potentially numerically unstable. The formulation appears problematic and requires clarification or correction (e.g., using log-likelihood directly, or a normalized confidence measure).

3. **Mask propagation logic** during backward iteration (t = T → 1) needs more precise specification: when a token is frozen at step t, the mask must remain zero for all subsequent (earlier) steps, but the interaction between multiple candidates' masks and the overall computation graph is not fully articulated.

4. **AR scorer overhead**: The method evaluates the AR model at every denoising step for every candidate (N × S_max calls), which is computationally substantial. The FLOP accounting includes this but doesn't discuss whether this overhead undermines the efficiency claims relative to uniform-S baselines that also use N-candidate reranking (without per-step AR evaluation).

5. **Tokenizer alignment** between DLM and AR scorer is mentioned in Model Preparation but the algorithm doesn't address how AR log-likelihoods are computed for tokens produced by a potentially different tokenizer during partial decoding.

**Feedback:**

The core idea—token-level adaptive freezing guided by AR likelihood—is well-motivated and genuinely novel relative to the baselines. To improve clarity and replicability, the authors should: (a) explicitly specify how discrete token probabilities are extracted from the diffusion model at intermediate denoising steps; (b) reformulate or clarify the confidence metric to avoid the exp(negative log-likelihood) issue; (c) provide a precise pseudocode-style description of mask update rules during backward iteration; (d) discuss the computational cost of per-step AR evaluation relative to the uniform-S baseline more transparently; and (e) address tokenizer mismatch handling in the scoring step.

**Rating (1-5):** 3