Review:

The proposed AR-SMCDS method presents a theoretically grounded Sequential Monte Carlo framework for guiding diffusion language model sampling with an autoregressive scorer, which is a creative and well-motivated contribution. The method is tested across five different DLM architectures and includes ablations for alternative scorers, sequence lengths, and scoring frequency—demonstrating a reasonable awareness of generalizability concerns.

However, several significant limitations constrain the method's generalizability:

1. **Tokenizer alignment hack**: Replacing diffusion model tokenizers with AR base tokenizers when they differ is a non-trivial intervention that could introduce semantic misalignment between the diffusion model's learned representations and the AR scorer's vocabulary, limiting applicability to model pairs with compatible tokenization schemes.

2. **Weight degeneracy in high dimensions**: The SMC importance weighting scheme is fundamentally susceptible to weight collapse in high-dimensional sequence spaces (L=128 tokens). While ESS-based resampling mitigates this, the problem is model-dependent and may worsen with longer sequences or different noise schedules—conditions not fully explored.

3. **Fixed hyperparameters**: A single temperature β and ESS threshold are used across all models and conditions, despite likely requiring per-model tuning. This reduces practical generalizability to unseen architectures without re-tuning.

4. **Narrow benchmark scope**: The quality metric is restricted to code (HumanEval), math (GSM8K), and commonsense (SIQA/WinoGrande). Generalization to open-ended generation, dialogue, or other task domains is not addressed.

5. **Superficial generalization checks**: Testing one additional DLM and two alternative scorers is a start but does not comprehensively establish broad applicability—continuous diffusion models, multimodal DLMs, different noise schedules, and alternative resampling strategies remain untested.

6. **Hardware constraint**: The method is optimized for a single RTX 6000 Pro Blackwell, limiting scalability to larger models or production environments with different memory/latency budgets.

7. **AR scorer assumption**: The method presumes the AR model provides meaningful likelihood guidance, but when the AR model and DLM have divergent capability profiles, the guidance may be misleading—a failure mode not analyzed.

Feedback:

The method's core SMC formulation is sound and the cross-model evaluation strategy is commendable. To strengthen generalizability, the authors should: (a) address the tokenizer incompatibility problem more rigorously—perhaps by learning a projection layer or using a shared tokenizer during DLM training rather than post-hoc replacement; (b) explore per-model hyperparameter tuning and report sensitivity analyses; (c) test on longer sequences and different task domains; (d) analyze failure modes when the AR scorer and DLM capabilities diverge; (e) consider alternative resampling strategies beyond systematic resampling; and (f) validate the method on hardware configurations beyond the single GPU constraint. The current generalization checks are a reasonable first step but should be expanded to more thoroughly establish the method's broad applicability.

Rating (1-5): 3