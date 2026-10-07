**Review:**

The proposed ACE method introduces an inference-time gating mechanism that uses an AR scorer to decide when to stop generating candidates, which is a conceptually modular departure from prior static grids and gradient-based guidance. The method does incorporate some generalizability-oriented elements: testing across five DLM architectures, alternative scorers, and held-out benchmarks. However, the generalizability evidence remains shallow and confined within the same paradigm. All tested scorers are still AR-style language models; the "unseen" DLM is another discrete diffusion transformer; and the sequence length is rigidly fixed at 128 tokens, leaving long-form generation untested. The FLOP estimation model is a rough approximation that may not transfer across architectures with different computational patterns (e.g., MoE, Mamba). The threshold mechanism assumes AR NLL is a reliable quality signal, which may not hold across domains or with noisier/cheaper scorers. While the modular separation of scorer and generator is a design choice that *could* support broader applicability, no evidence is provided for cross-paradigm extension (e.g., continuous diffusion, non-LM generators, non-AR scorers such as reward models or classifiers). The generalizability checks feel like compliance with a checklist rather than integral to the method's design, and the core hypothesis—whether AR-guided early acceptance outperforms step-budget increases—is tightly coupled to the discrete DLM setting.

**Feedback:**

1. **Deepen generalizability evidence**: Test with genuinely different scorer types (e.g., a frozen reward model, a lightweight classifier, or a semantic similarity metric) to determine whether the AR NLL signal is special or whether any reasonable quality proxy suffices. This would strengthen claims about scorer-agnosticism.

2. **Vary sequence length**: Fixing L=128 limits applicability to long-form tasks. A brief ablation across L ∈ {64, 256, 512} would reveal whether the ACE threshold and FLOP model scale with length.

3. **Address the FLOP approximation**: The linear FLOP model (α × |θ| × L) ignores architectural differences (MoE sparsity, Mamba state-space dynamics). Validate with actual measured FLOPs per step across architectures, or acknowledge this as a limitation.

4. **Explore cross-paradigm potential**: Briefly discuss whether the ACE gating pattern could apply to continuous diffusion, flow matching, or non-LM generators, even if not experimentally tested. This would clarify the method's broader relevance.

5. **Justify N_max = 8**: Provide reasoning or a sensitivity analysis for the candidate ceiling. For harder prompts or larger generation budgets, the optimal N_max may differ.

6. **Threshold initialization strategy**: Starting τ_p at −∞ means the first candidate is always accepted, which biases early compute allocation. Consider comparing with a warm-up period or percentile-based initial threshold.

**Rating (1-5): 2**