Problem:  
How does the choice of noise schedule (e.g., linear, cosine, or learned schedules) at inference time affect the quality‑efficiency trade‑off of scaled diffusion language models adapted from autoregressive checkpoints, and does this relationship differ across model families (GPT‑2‑based vs. LLaMA‑based) and scales (127 M – 7 B)?

Rationale:  
The target paper shows that continual pre‑training of open‑source autoregressive (AR) checkpoints yields competitive diffusion language models (DLMs) such as DiffuGPT and DiffuLLaMA. However, several limitations remain that can be probed without additional training:

1. **Undertrained models (L7).** The adapted DLMs are reported to be undertrained, leaving a quality gap to their AR counterparts. A well‑chosen noise schedule can improve the denoising trajectory and thus sample quality without changing model weights, offering a training‑free way to narrow this gap.

2. **Efficiency characterization (L9).** The original work reports wall‑clock latency without normalizing for compute. Different noise schedules alter the number and placement of denoising steps, directly affecting the number of transformer forward passes and the associated FLOPs. By measuring both quality and compute‑normalized efficiency (e.g., perplexity per GFLOP), we can obtain a more faithful quality‑efficiency frontier.

3. **Architectural generality (L10).** The adaptation recipe was applied only to GPT‑2 and LLaMA families up to 7 B. Evaluating multiple noise schedules across these families and scales will reveal whether certain architectures benefit more from specific schedules (e.g., cosine schedules may better match the curvature of the loss landscape in larger LLaMA models), informing future adaptation strategies.

4. **Downstream capabilities.** The paper claims proficiency in infilling, instruction following, and reasoning. Noise schedules influence the balance between exploration (early steps) and refinement (later steps); we can assess whether particular schedules improve specific abilities such as fill‑in‑the‑middle accuracy or instruction‑following performance without additional training.

**Feasibility under the given constraints:**  
- **Inference‑only:** All experiments use the released checkpoints (DiffuGPT‑S/M, DiffuLLaMA‑6.74 B, Dream‑7B, DiffuCoder‑7B, LLaDA‑8B). No pre‑training, continual pre‑training, or AR‑to‑diffusion adaptation is required.  
- **Compute:** One NVIDIA RTX 6000 Pro Blackwell (96 GB) suffices for batched inference. We will sweep over a small set of schedules (e.g., uniform linear, cosine, and a simple learned schedule obtained by fitting a monotonic function to the training timesteps) and a range of total denoising steps (e.g., 8, 16, 32, 64). Approximate FLOPs are computed as model size × steps × sequence length, enabling compute‑normalized metrics.  
- **Timeline:** Implementing a schedule‑switching harness, running sweeps, collecting automated metrics (perplexity, MAUVE, distinct‑n, HumanEval‑style infilling, instruction‑following accuracy), and analysis fits within ten weeks with three GPU‑enabled team members handling parallel runs; the remaining members manage CPU‑heavy tasks (metric computation, write‑up).  
- **Team:** CPU‑heavy tasks (metric computation, analysis, write‑up) can be performed by the remaining four members.

**Significance:**  
Systematically quantifying how noise‑schedule choice reshapes the quality‑efficiency frontier of scaled DLMs directly addresses limitations L7 (undertrained models) and L9 (non‑compute‑normalized efficiency) by providing a controllable, training‑free lever to improve generation quality and alignment. It extends the adaptation paradigm introduced in the target paper, offering practitioners a simple knob to enhance performance across model families and scales without costly retraining. Moreover, by revealing schedule‑dependent architectural sensitivities, the work can guide future DLM design (e.g., selecting schedules that better match the curvature of the loss landscape for larger models), thereby contributing to more efficient and effective diffusion‑based text generation.