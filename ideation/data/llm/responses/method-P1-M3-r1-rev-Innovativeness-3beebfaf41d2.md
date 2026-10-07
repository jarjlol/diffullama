**Review:**

The AD‑AES method proposes using the AR scorer as a real‑time monitor of improvement during diffusion denoising, triggering early stopping on a per‑candidate basis when the AR score plateaus. This creates a prompt‑adaptive effective denoising budget that is genuinely distinct from the uniform (N, S) grid and from gradient‑based guidance (e.g., TESS 2). The framing of the diffusion trajectory as an any‑time process whose compute is allocated by an external scorer is a fresh perspective on the quality–compute trade‑off.

However, several concerns limit the innovativeness assessment:

1. **Conceptual proximity to standard early stopping:** Monitoring a validation metric and halting when improvement falls below a threshold is a decades‑old technique in optimization. The novelty lies in its application context (inference‑time, per‑candidate, within diffusion sampling), but the underlying mechanism is not fundamentally new.

2. **Signal reliability concern:** The method uses argmax over the vocabulary at intermediate diffusion steps to obtain a token sequence, then computes AR NLL. Intermediate diffusion states (especially at early steps with high noise) may produce incoherent token sequences, making the AR score a noisy and potentially misleading stopping signal. No ablation or analysis of this noise is proposed.

3. **Overclaimed distinction from prior work:** The method characterizes itself as "fundamentally different" from Jacobi Forcing and TESS 2, but the line between "monitoring improvement" and "guidance" is blurry—TESS 2's reward guidance also operates at inference time without retraining. The distinction needs sharper articulation.

4. **Lack of theoretical motivation:** There is no argument for why AR score plateau should correlate with diffusion convergence, leaving the heuristic foundation weak.

5. **Incremental rather than transformative:** The method adds a third option (adaptive steps) between the two extremes (more steps vs. more candidates), but this is a natural interpolation rather than a reframing of the problem.

**Feedback:**

The AD‑AES idea is a reasonable and implementable inference‑time strategy, but its innovativeness is moderate at best. To strengthen the contribution, the authors should: (a) analyze the reliability and noise characteristics of the AR score at intermediate diffusion steps, (b) provide a theoretical or empirical justification for why AR score plateau is a valid stopping criterion, (c) more carefully distinguish the method from existing guidance and early‑stopping literature, and (d) consider whether the early‑stopping threshold ε and acceptance percentile τ_p could be learned or tuned more systematically rather than treated as fixed hyperparameters. The method's practical value may still be meaningful even if its conceptual novelty is incremental.

**Rating (1-5): 3**