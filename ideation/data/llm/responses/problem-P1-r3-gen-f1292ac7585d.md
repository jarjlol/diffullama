**Problem:**  
Does the extent to which a diffusion language model (DLM) preserves the autoregressive (AR) representations of its pretrained backbone determine whether allocating additional inference compute to (a) more denoising steps or (b) generating more candidates and reranking them with a tiny AR scorer yields a larger improvement in the quality‑efficiency trade‑off, and does this relationship vary across model families (GPT‑2‑based vs. LLaMA‑based) and denoising‑step budgets?

**Rationale:**  
The target paper demonstrates that continual‑pre‑training of open‑source AR models yields competitive DLMs (DiffuGPT, DiffuLLaMA) but leaves a measurable “accuracy headroom” (L6) that could be closed either by running more denoising steps or by better exploiting the model’s inherent candidate diversity. Recent inference‑time methods such as Jacobi Forcing (self‑distillation of parallel decoding trajectories) and TESS 2’s reward guidance show that quality can be traded for efficiency, yet it remains unclear *which* compute allocation—more denoising versus more sampling with lightweight reranking—is more effective for a given DLM, and whether this depends on how much AR knowledge the model retains after adaptation.  

If a DLM closely preserves the AR backbone’s representations, its internal denoising process may already be well‑aligned with the AR prior, suggesting that additional denoising steps yield diminishing returns while sampling more candidates and reranking with a lightweight AR model could recover the headroom more efficiently. Conversely, if representation preservation is low, extra denoising may be needed to compensate for misaligned representations. By quantifying representation preservation and systematically measuring the quality‑efficiency outcomes of the two compute‑allocation strategies across multiple released DLMs, we can test this hypothesis and provide practitioners with a principled rule for allocating inference budget between denoising and selection when deploying scaled diffusion language models.

**Methodology (inference‑only, fits a single RTX 6000 Pro Blackwell and a 10‑week timeline):**  

1. **Model selection & representation‑preservation measurement**  
   - Choose three representative released checkpoints: DiffuGPT‑S (127 M, GPT‑2‑based), DiffuLLaMA (6.74 B, LLaMA‑based), and Dream‑7B (7 B, LLaMA‑based).  
   - For each DLM, obtain its AR base model (GPT‑2‑small for DiffuGPT‑S, LLaMA‑2‑7B for the two 7B models).  
   - On a held‑out text corpus (e.g., WikiText‑103 validation split), compute centered kernel alignment (CKA) between the hidden‑state representations of each DLM layer and the corresponding layer of its AR base model. Aggregate across layers (e.g., average CKA) to obtain a single **representation‑preservation score** per DLM.  
   - (Optional) Validate CKA with a lightweight linear‑probe task (next‑token prediction on a small probe set) to ensure robustness.

2. **Candidate generation across denoising‑step budgets**  
   - For each DLM, generate text samples using the model’s native diffusion sampler at three denoising‑step budgets: **low (16 steps)**, **medium (64 steps)**, and **high (256 steps)**.  
   - At each step budget, produce **N = 1, 4, 8** independent samples per prompt (using different random seeds).  
   - Prompts: a balanced mix of (i) HumanEval function signatures (for code infilling), (ii) GSM8K math word problems, and (iii) SIQA/WinoGrande commonsense questions (≈200 prompts total, stratified).  
   - All generation is performed with **fp16 precision** and **activation checkpointing** to keep GPU memory ≤ 96 GB; where necessary, apply **4‑bit quantization** (via bitsandbytes) to the largest model (DiffuLLaMA) – this does not affect the representation‑preservation measurement because CKA is computed in fp16 before quantization.

3. **Lightweight AR reranking**  
   - For every set of N samples per prompt, compute a score using a tiny AR scorer (GPT‑2‑small, 124 M parameters) via average negative log‑likelihood of the full generated sequence.  
   - Select the top‑ranked sample as the final output for that prompt under the “sampling + reranking” strategy.  
   - The “denoising‑only” baseline uses the single sample (N=1) generated at the given step budget without reranking.

4. **Quality and efficiency metrics**  
   - **Quality:** For each strategy (denoising‑only, sampling + reranking) compute three task‑specific scores:  
        * HumanEval pass@k (k=1,10) → averaged to a code score.  
        * GSM8K exact‑match accuracy.  
        * SIQA/WinoGrande accuracy (averaged).  
     - Normalize each score to [0,1] across all conditions, then take the **unweighted mean** to obtain a unified **Quality** metric.  
   - **Efficiency:** Measure **wall‑clock latency per generated token** (including scorer overhead for the reranking condition) averaged over all prompts and runs. Convert to **FLOPs‑equivalent** by multiplying latency by the GPU’s theoretical peak FP16 throughput (≈ 140 TFLOPs) to obtain an **Estimated Compute** value.  
   - Define the **Quality‑Efficiency Trade‑off (QET)** as **Quality / (Estimated Compute)**; higher QET indicates better quality per unit compute.

5. **Analysis**  
   - For each DLM, plot QET as a function of denoising steps (for denoising‑only) and as a function of sample count N (for sampling + reranking) at each step budget.  
   - Compute the **marginal QET gain** from (a) increasing denoising steps from low→medium→high (keeping N=1) and (b) increasing N from 1→4→8 (keeping steps fixed at the medium budget).  
   - Test whether higher representation‑preservation scores predict **larger marginal QET gains from sampling + reranking** and **smaller gains from additional denoising steps** using Spearman correlation across the three DLMs.  
   - Examine whether the correlation differs between GPT‑2‑based and LLaMA‑based models and whether it changes with the denoising‑step budget (interaction analysis).  
   - Perform a non‑parametric bootstrap (1 000 resamples) to obtain confidence intervals for correlation coefficients.

**Feasibility considerations:**  
- **GPU memory:** Activation checkpointing + optional 4‑bit quantization keeps the 6.74 B model within 96 GB; the two smaller models fit comfortably in fp16.  
- **Throughput:** Generating at most 8 samples per prompt with 256 steps is the most expensive configuration; with prompt‑level batching (batch size = 4) and parallel CPU‑side scoring, the total generation time is estimated at ≈ 30 GPU‑hours, well within a 10‑week window when distributed across the three GPU‑enabled team members.  
- **Checkpoint access:** All listed checkpoints are publicly released (DiffusionFamily, Dream, LLaDA, DiffuCoder) and have been verified for download.  
- **Evaluation harnesses:** HumanEval, GSM8K, SIQA, and WinoGrande scripts are pre‑built and CPU‑tested before GPU generation begins.  

**Expected impact:**  
The study will clarify whether the “accuracy headroom” in adapted DLMs is better addressed by spending more compute on denoising or by leveraging candidate diversity with a lightweight AR reranker, and it will link this decision to a measurable property of the model—its retention of AR representations. This provides a concrete, inference‑time guideline for practitioners deploying scaled diffusion language models and bridges the representation‑alignment literature (REPR‑ALIGN, PreDiff‑LM) with inference‑time efficiency work (Jacobi Forcing, TESS 2), offering a principled, diagnostic tool rather than a new training method.