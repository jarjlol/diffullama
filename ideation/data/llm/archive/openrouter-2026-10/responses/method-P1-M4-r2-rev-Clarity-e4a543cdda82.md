**Review:**
The AGADES method is well-structured with a clear algorithmic outline, explicit FLOPs formulas, and a detailed week-by-week implementation plan. The symbol table and stepwise procedure aid comprehension. However, several technical ambiguities undermine replicability: (1) the initial marginal gain for s_i=0 is set to 0 with an undefined "optimistic prior," making first-step candidate selection ambiguous; (2) decoding the latent to compute AR-NLL after each step is underspecified (argmax vs. expected vs. sampled tokens); (3) the absolute-value convergence check |ΔNLL| < ε does not clarify whether improvement direction matters (NLL could worsen); (4) the mapping from budget B to uniform-baseline FLOPs is imprecise ("0.5× to 4×" without a concrete reference); (5) the marginal-gain regression holds "N and S_max constant" while S_max is only a ceiling, creating confusion about what is actually varied; (6) Section 7 introduces generalization checks (unseen DLM, alternative scorers, L=256) that strain the stated 10-week / single-GPU constraint without discussing feasibility trade-offs.

**Feedback:**
- Define the optimistic prior numerically and specify tie-breaking for the first step.
- Explicitly state the decoding scheme used to obtain a concrete token sequence for AR-NLL computation at each intermediate step, and justify its choice.
- Replace the absolute-value convergence check with a one-sided improvement check (delta < ε and delta > 0) to avoid stopping when NLL worsens.
- Provide an exact formula linking B to the uniform baseline FLOPs (e.g., FLOPs_uniform = N·S_max·F_diff + N·F_AR).
- Clarify the regression variables in Section 6 to reflect that actual steps ∑s_i, not S_max, drive quality.
- Either justify the expanded generalization scope within the 10-week plan or prioritize the core experiments.

**Rating (1–5):** 3