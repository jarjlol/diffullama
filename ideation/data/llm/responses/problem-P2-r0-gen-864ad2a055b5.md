**Problem:**  
What is the impact of classifier‑free guidance on the quality‑efficiency trade‑off of scaled diffusion language models adapted from autoregressive checkpoints, and how does this impact differ across model families (GPT‑2‑based vs. LLaMA‑based) and scales (127 M – 7 B)?

**Rationale:**  
The target paper demonstrates that continual pre‑training of open‑source autoregressive (AR) checkpoints yields competitive diffusion language models (DLMs) such as DiffuGPT and DiffuLLaMA. However, the study leaves several open questions:

1. **Quality control without retraining** – The paper notes that the adapted models are still undertrained (L7) and that performance gaps remain relative to their AR counterparts. Classifier‑free guidance (CFG) is a training‑free technique that can markedly improve sample quality and alignment in diffusion models by scaling the conditional prediction toward the unconditional one. Its effect on large‑scale text diffusion models has not been systematically examined.

2. **Efficiency characterization** – The original work reports wall‑clock decoding latency without normalizing for compute (L9). CFG allows us to trade extra compute (additional forward passes for the unconditional branch) for higher quality, providing a natural axis to study compute‑normalized quality‑efficiency curves.

3. **Generality across architectures and scales** – The adaptation recipe was applied only to GPT‑2 and LLaMA families up to 7 B parameters (L10). Evaluating CFG across these families and scales will reveal whether the guidance mechanism behaves similarly or whether certain architectures benefit more, informing future adaptation strategies.

4. **Applicability to downstream capabilities** – Beyond raw language modeling, the paper claims proficiency in infilling, instruction following, and reasoning. CFG can be applied to conditional generation (e.g., prompting with an instruction or a prefix‑suffix infill mask) to assess whether guidance improves these specific abilities without additional training.

**Feasibility under the given constraints:**  
- **Inference‑only:** All experiments use the released checkpoints (DiffuGPT‑S/M, DiffuLLaMA‑6.74 B, Dream‑7B, DiffuCoder‑7B, LLaDA‑8B, etc.). No pre‑training, continual pre‑training, or AR‑to‑diffusion adaptation is required.  
- **Compute:** One NVIDIA RTX 6000 Pro Blackwell (96 GB) suffices for batched inference. We will measure wall‑clock time, approximate FLOPs (model size × denoising steps × sequence length), and throughput.  
- **Timeline:** Implementing a CFG evaluation harness, running sweeps over guidance scales (e.g., 0.0, 0.5, 1.0, 1.5, 2.0) and denoising‑step budgets (e.g., 8, 16, 32, 64 steps), and collecting automated metrics (perplexity, MAUVE, distinct‑n, instruction‑following accuracy, HumanEval‑style infilling) can be completed within ten weeks with three GPU‑enabled team members handling parallel runs.  
- **Team:** CPU‑heavy tasks (metric computation, analysis, write‑up) can be performed by the remaining four members.

**Significance:**  
Systematically quantifying how classifier‑free guidance reshapes the quality‑efficiency frontier of scaled DLMs will directly address limitations L9 (efficiency not compute‑normalized) and L7 (undertrained models) by providing a controllable, training‑free method to close the gap with AR baselines. It will also expand the understanding of the adaptation paradigm introduced in the target paper, offering practitioners a simple knob to improve generation quality, alignment, and task‑specific performance without costly retraining. This work bridges the gap between the theoretical promise of diffusion language models and practical, deployable systems.