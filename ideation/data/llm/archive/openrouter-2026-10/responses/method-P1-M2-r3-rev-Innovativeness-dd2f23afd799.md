**Review:**

The proposed TADAG method introduces a token-level adaptive compute mechanism for diffusion language models, claiming to allocate denoising steps per-token based on AR-model confidence scores. While the specific instantiation (freezing confident tokens during diffusion sampling) is a concrete contribution not present in the cited prior work, the innovation sits at the level of a novel *combination* rather than a fundamentally new principle. Several concerns merit attention:

1. **Limited novelty of core idea**: Confidence-based early stopping / adaptive computation is well-established in other ML contexts. The application to DLMs is new but builds directly on existing concepts (AR guidance as in Jacobi Forcing, reward guidance as in TESS 2).

2. **Practical effectiveness concerns**: At early denoising steps, the sequence is dominated by noise, making AR log-likelihoods unreliable confidence signals. Freezing tokens based on these noisy estimates may trap errors early rather than improve quality. The method's benefit is contingent on the AR model producing meaningful scores at high-noise timesteps—an assumption not empirically validated in the proposal.

3. **Compute overhead tension**: The method requires AR scoring at *every* denoising step for *every* candidate to make freezing decisions. The FLOP budget explicitly includes N × S_max × F_AR, which substantially increases total compute. The savings from freezing must outweigh this overhead—a non-trivial claim that remains unverified.

4. **Circularity in the freezing logic**: The AR score is computed on the diffusion model's current (tentative) output, which is itself being shaped by the freezing decisions. This feedback loop is not analyzed, and its stability/convergence properties are unclear.

5. **Orthogonality claim overstated**: The token-adaptive trade-off axis is presented as orthogonal to S and N, but in practice, the effective denoising steps per token are still bounded by S_max and correlated with N (more candidates → more AR scoring → more freezing decisions).

**Feedback:**
The method is a reasonable engineering contribution that could yield empirical insights, but the innovation is moderate rather than high. To strengthen the proposal: (a) provide a theoretical or empirical analysis of when AR likelihoods at intermediate denoising steps are reliable confidence signals; (b) address the compute overhead tension more rigorously—consider whether the AR scoring cost at every step undermines the efficiency claim; (c) analyze the freezing feedback loop; (d) consider simpler alternatives (e.g., freezing only at the final step based on AR score) as baselines to isolate the incremental value of per-step token-adaptivity. The generalization checks (alternative scorers, sequence lengths) are a strength that partially compensates for the conceptual limitations.

**Rating (1-5): 3**