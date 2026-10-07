**Problem:**  
How does the shape of the noise schedule (e.g., linear, cosine, or context‑adaptive token‑level rescheduling) used at inference time affect the quality‑efficiency trade‑off of scaled diffusion language models that have been adapted from autoregressive checkpoints, and does this relationship differ across model families (GPT‑2‑based vs. LLaMA‑based) and model scales (127 M – 7 B)?

**Rationale:**  
The target paper shows that continual denoising‑training of autoregressive (AR) checkpoints yields competitive diffusion language models (DLMs) such as DiffuGPT and DiffuLLaMA, but it leaves several under‑examined factors that influence the denoising process.  

1. **Underexplored inference‑time knob:** While the paper ablates the denoising‑step budget (L1–L2) and the shift operation, it does not study how the *temporal distribution* of noise (the noise schedule) impacts generation quality or sampling efficiency. Recent work (Dream 7B, UNIFUSION, dLLM) highlights that context‑adaptive or uniform‑noise schedules can substantially improve perplexity, planning ability, and inference flexibility.  

2. **Connection to reported limitations:**  
   - **L1 (attention‑mask annealing):** The paper drops annealing at scale, claiming minimal impact. A different noise schedule could mitigate the loss of annealing by allocating more noise to early timesteps where bidirectional context matters most.  
   - **L4 (proxy‑task validation):** The adaptation recipe was chosen on a cheap proxy (GSM8K‑symbolic). Evaluating alternative schedules directly on the adapted models provides a low‑cost way to test whether the proxy truly reflects the full adaptation objective.  
   - **L9 (efficiency claims not compute‑normalised):** By measuring wall‑clock time or number of forward passes under different schedules, we can obtain a compute‑normalised view of efficiency that complements the original latency‑only comparison.  

3. **Feasibility under constraints:**  
   - **Inference‑only:** All required checkpoints (DiffuGPT‑S/M, DiffuLLaMA 6.74B, LLaDA‑8B, Dream‑7B, DiffuCoder‑7B) are publicly available. Changing the noise schedule entails only modifying the timestep sampler (e.g., using a linear βₜ schedule, a cosine schedule, or the token‑level rescheduling described in Dream 7B) and re‑running the diffusion sampling loop—no retraining or gradient updates are needed.  
   - **Compute:** A single NVIDIA RTX 6000 Pro Blackwell can accommodate batched sampling for the 6–8 B‑parameter models; smaller models (127 M–355 M) allow extensive ablation sweeps.  
   - **Timeline:** Implementing alternative schedulers, running perplexity/efficiency evaluations on WikiText‑103 or C4, and analysing results fits comfortably within a ten‑week window with a seven‑person team (three GPU‑enabled members for sampling, four CPU‑only members for data prep, metric computation, and analysis).  

4. **Significance:**  
   - Clarifying how noise‑schedule shape interacts with model scale and architecture will guide practitioners in selecting schedules that maximize quality per compute unit, potentially closing the gap between adapted DLMs and their AR counterparts.  
   - Findings could inform the design of future adaptation recipes (e.g., pairing a specific schedule with reduced annealing or shift adjustments) and provide a concrete, low‑cost ablation that validates or refutes the proxy‑task decisions reported in the target paper.  

By systematically comparing linear, cosine, and context‑adaptive noise schedules across the released diffusion checkpoints, this study directly addresses an underexplored yet pivotal dimension of diffusion language modeling, offering actionable insights for both researchers and engineers aiming to deploy scalable, efficient text generators.