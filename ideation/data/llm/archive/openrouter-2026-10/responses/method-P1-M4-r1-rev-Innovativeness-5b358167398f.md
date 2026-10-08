**Review:**

The PEAS method proposes an inference-time adaptive strategy for diffusion language models that allocates denoising compute dynamically across a candidate pool based on early evaluations from a lightweight AR scorer. While the method is well-structured and addresses the research problem rigorously, its innovativeness is moderate rather than high.

The core idea—generating candidates, scoring them with a cheap proxy, and terminating strong performers early—is a combination of well-established techniques: multi-candidate generation with reranking is standard in NLP, early-exit/early-stopping mechanisms are common in deep learning, and adaptive compute allocation has been explored in various forms (e.g., dynamic depth networks). TESS 2's reward guidance and Jacobi Forcing's trajectory distillation already demonstrate that inference-time compute can be reallocated in DLMs. PEAS's specific contribution—the progressive blockwise evaluation with a quality-dependent acceptance threshold and compute reallocation—is a sensible and concrete instantiation, but it does not introduce a fundamentally new principle or mechanism.

The method's strengths lie in its systematic empirical framework: the unified quality metric, FLOP-normalized efficiency measurement, Pareto-front analysis, and thorough sensitivity/hyperparameter exploration. These methodological rigor improvements are valuable but pertain more to experimental design than to algorithmic novelty. The threshold calibration via percentile sweep is a practical contribution, and the formal marginal-gain analysis (ΔUQM/ΔS, ΔUQM/Δlog N) provides useful quantification. However, these are analytical refinements rather than new techniques.

The method does not fundamentally transform how DLMs are used or understood; it offers an improved inference-time policy built from existing building blocks. The novelty is in the specific combination and formalization, not in the individual components.

**Feedback:**

1. **Sharpen the novelty claim:** Identify precisely which component is new—the progressive acceptance mechanism, the threshold calibration, or the compute reallocation formula—and justify why this specific combination hasn't been tried before (e.g., by discussing why TESS 2's reward guidance doesn't subsume PEAS).

2. **Compare more directly to existing adaptive inference methods:** Early-exit networks, adaptive computation time, and dynamic depth models have explored similar "stop early when confident" logic. Explicitly discuss what distinguishes PEAS from these and why they haven't been applied to DLMs.

3. **Clarify the failure mode of the AR scorer:** If the AR scorer is poorly calibrated (e.g., NLL doesn't correlate with actual DLM quality), the early acceptance could discard good candidates. A brief discussion of scorer mismatch and robustness would strengthen the method.

4. **Consider a ablation of the acceptance mechanism itself:** Test whether simple "generate N, score all at full S_max, pick best" (i.e., no early acceptance, just reranking) already achieves most of the gain. If so, PEAS's marginal value is primarily in efficiency, not quality—which should be stated honestly.

**Rating (1-5): 3**