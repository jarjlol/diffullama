Review:
The ATAR method is well-organized with clear sectioning, a concise algorithm table, and a logical flow from preparation through Pareto analysis. The high-level idea—use AR entropy to focus diffusion refinement on uncertain tokens—is well-motivated and genuinely innovative. However, several critical implementation details are vague or potentially misleading, undermining replicability: (1) the attention-mask mechanism for freezing tokens during partial denoising is hand-wavy and conflates gradient-based and forward-pass reasoning; (2) the construction of the partially noisy latent during refinement rounds is ambiguous regarding how noise levels and timesteps are managed for mixed active/inactive tokens; (3) the claim that a tokenizer can be swapped "with no weight change" ignores vocabulary/embedding dimension mismatches; (4) the optional recomputation of entropy masks is left underspecified for the main experiments; (5) the wall-clock linearity validation and baseline guidance integration lack concrete procedures. These gaps leave a competent researcher with real uncertainty about how to implement the core refinement loop faithfully.

Feedback:
- Clarify the exact mechanism for freezing tokens during denoising (specify attention-mask application, layer-norm handling, and whether inactive tokens are injected as clean embeddings or kept at their current latent value).
- Define the noise schedule for partially active sequences at each refinement round (how is the timestep decrement applied when only a fraction of tokens is updated?).
- Justify or replace the tokenizer-alignment claim; if vocabularies differ, describe the actual remapping procedure.
- Specify default behavior for entropy-mask recomputation (fixed vs. every k rounds) and the choice of k.
- Concrete the wall-clock validation protocol (what range of conditions is used to establish linearity?) and the guidance-based baseline integration.
- Fix the "gradient" language in the inference procedure to forward-pass terminology.

Rating (1-5): 3