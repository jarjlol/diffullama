Review:
AGIR‑DLM presents a well‑structured inference‑time method that addresses the research problem with appropriate rigor. The importance‑resampling framing is the key novel contribution, offering a principled way to allocate compute between denoising steps and candidate diversity. The method is clearly described, reproducible, and stays within the inference‑only constraint. The Pareto front analysis with AUPC and marginal‑gain regressions provides a quantitative framework that directly targets the identified gaps (L6, L9).

Feedback:
1. Clarify the importance‑sampling justification: The AR log‑likelihood is used as resampling weights, but true importance sampling requires a known proposal–target ratio. The AR score is a heuristic proxy — acknowledge this or reframe as "score‑based resampling."
2. Compare against stronger baselines: Include Jacobi Forcing‑style multi‑block decoding and TESS 2 reward guidance to contextualize AGIR‑DLM's position.
3. Address diversity collapse risk: With R > 1, resampling with replacement may collapse diversity if the scorer is overconfident — add a diversity penalty or temperature‑scaled weights and report pairwise overlap.
4. Validate FLOP estimates empirically: Use torch.profiler rather than relying solely on the analytical approximation, as different diffusion architectures have varying per‑step costs.
5. Test scorer calibration: Investigate whether well‑calibrated probabilities improve resampling quality over raw log‑likelihood.
6. Clarify duplicate handling: When resampling selects the same candidate multiple times, clarify whether computation is duplicated or merged.

Rating (1-5): 3