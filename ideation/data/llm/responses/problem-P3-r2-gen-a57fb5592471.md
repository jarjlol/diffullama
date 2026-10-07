**Problem:**  
How does varying the attention‑mask schedule (from fully causal to fully bidirectional) during diffusion sampling affect the quality‑efficiency trade‑off of scaled diffusion language models adapted from autoregressive checkpoints, and can a simple mask‑annealing schedule recover the performance loss incurred when the original attention‑mask annealing is removed at scale?

**Rationale:**  

1. **Underexplored inference‑time knob with architectural relevance**  
   The target paper ablates the denoising‑step budget and the shift operation but does not examine how the *temporal pattern of attention masking* (i.e., how much bidirectional context is allowed at each denoising step) interacts with model scale and architecture. Recent work (Dream 7B, UNIFUSION, dLLM) shows that context‑adaptive noise rescheduling can improve perplexity and planning ability, yet the analogous idea of adapting the attention mask over time has not been systematically studied. By manipulating the mask schedule we can directly test whether the dropped attention‑mask annealing (L1) is truly harmless or merely masked by a sub‑optimal masking pattern.

2. **Link to reported limitations**  
   - **L1 (attention‑mask annealing):** The paper removes annealing at the 7B scale, claiming minimal impact, yet its own ablation shows a growing benefit with scale. Testing alternative mask schedules provides a low‑cost, inference‑only way to assess whether the dropped annealing is truly innocuous.  
   - **L4 (proxy‑task validation):** The adaptation recipe was selected using a cheap proxy (GSM8K‑symbolic). Evaluating mask schedules directly on the adapted checkpoints offers an inexpensive validation of whether the proxy truly reflects the full adaptation objective across scales and families.  
   - **L9 (efficiency claims not compute‑normalized):** By measuring the average number of forward passes per generated token (equivalent to the denoising‑step budget) and wall‑clock time per token under each mask schedule, we obtain a compute‑normalized efficiency metric that complements the original latency‑only comparison and reveals whether a schedule can recover efficiency lost due to the missing annealing.  
   - **L8 (infilling claims broader than evaluation):** Mask schedules directly affect the model’s ability to perform bidirectional infilling. We can evaluate whole‑function infilling (e.g., HumanEval multi‑line) as well as single‑line infilling to see whether a schedule that restores bidirectional context narrows the gap between the advertised infilling capability and the narrow evaluation used in the paper.  

3. **Operational definition of the quality‑efficiency trade‑off**  
   - **Quality:**  
     * Language modeling perplexity (or bits‑per‑character) on a held‑out corpus (WikiText‑103 or C4).  
     * Infilling accuracy: pass@k on HumanEval multi‑line infilling and on the FIM (fill‑in‑the‑middle) benchmark.  
     * Reasoning accuracy: average score on a subset of GSM8K and BBB (Big‑Bench Hard) tasks.  
   - **Efficiency:**  
     * Average number of forward passes required to produce one token (i.e., effective denoising‑step budget).  
     * Wall‑clock time per token at batch size 1 on the RTX 6000 Pro Blackwell (with 4‑bit quantization where needed).  
   The trade‑off will be visualized as quality versus efficiency curves; a schedule that yields higher quality for a given number of forward passes (or higher tokens‑per‑second for a given quality) is considered superior.

4. **Feasibility under the given constraints**  
   - **Inference‑only:** All required checkpoints (DiffuGPT‑S/M, DiffuLLaMA 6.74B, Dream‑7B, DiffuCoder‑7B, LLaDA‑8B) are publicly available. Changing the attention mask entails only modifying the mask tensor fed to the transformer at each denoising step—no retraining or gradient updates are needed.  
   - **Compute:** A single RTX 6000 Pro Blackwell (96 GB) can accommodate batched sampling for the 6–8 B‑parameter models when using 4‑bit quantization or careful batch sizing; smaller models allow extensive sweeps of mask‑schedule hyperparameters.  
   - **Timeline:** Implementing the mask‑schedule logic, running perplexity/efficiency/infilling/reasoning evaluations, and analyzing results fits comfortably within a ten‑week window with three GPU‑enabled members for sampling and four CPU‑only members for data preparation, metric computation, and statistical analysis.  

5. **Originality and significance**  
   - Rather than merely confirming that noise schedules matter, we hypothesize *why* attention‑mask schedules might interact differently with model scale: early denoising steps benefit from causal masking to preserve the autoregressive inductive bias learned during pretraining, while later steps can safely incorporate bidirectional context to refine coherence.  
   - We propose to evaluate a **simple mask‑annealing schedule** (e.g., linear increase of bidirectional token proportion from 0% at step 0 to 100% at the final step) as a straightforward, architecture‑agnostic remedy for the removed attention‑mask annealing. Demonstrating that this schedule narrows the quality‑efficiency gap between adapted DLMs and their autoregressive baselines would provide actionable guidance for practitioners and suggest a direction for future adaptation recipes (e.g., pairing specific mask schedules with architectural tweaks).  
   - By connecting mask‑schedule choice to scale‑dependent performance, the study moves beyond an empirical ablation toward a mechanistic understanding of how diffusion processes interact with the causal structure learned during autoregressive pretraining, thereby offering a contribution that is more than a routine hyperparameter sweep.  

In summary, this work will systematically examine how attention‑mask schedule, model scale, and architecture jointly determine the quality‑efficiency trade‑off of adapted diffusion language models, test whether a simple mask‑annealing schedule can compensate for the dropped attention‑mask annealing, and provide concrete, inference‑only recommendations for deploying scalable, efficient text generators.