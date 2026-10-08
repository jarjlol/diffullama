**Method:** Adaptive Candidate Generation via AR‑Guided Early Acceptance (ACE)  

**Rationale:**  
The target paper identifies two open issues: (L6) a sizable “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, leaving it unclear whether DLMs truly beat AR models when cost is accounted for. Existing inference‑time work (e.g., Jacobi Forcing, TESS 2 reward guidance) shows that compute can be shifted between denoising and selection, but it remains unknown whether the headroom is best addressed by (i) raising the denoising‑step budget *S* or (ii) generating and reranking more samples *N* using a tiny AR scorer.  

The two previously proposed approaches tackle this question by (1) static/dynamic *N*,*S* grids with a two‑stage refinement step, and (2) gradient‑based guidance that injects AR scores into the denoising dynamics. Both methods either pre‑specify the number of candidates or modify the diffusion trajectory itself.  

We propose a fundamentally different angle: **let the AR scorer decide, on‑the‑fly, whether a newly generated candidate is good enough to stop further sampling**. In other words, the AR scorer is used not to rerank a fixed set of candidates, but to gate the *generation process* itself. This yields an *effective* number of candidates that adapts to the difficulty of each prompt and the quality of the early samples. By measuring the resulting quality–compute curve we can directly answer whether allocating compute to more AR‑guided candidates (via early acceptance) improves the Pareto frontier more effectively than simply increasing the denoising‑step budget *S*.  

The ACE procedure respects all constraints: it is inference‑only, uses only publicly released checkpoints, requires no training or adaptation, and can be implemented within the ten‑week timeline on a single RTX 6000 Pro Blackwell by fixing sequence length to 128 tokens and parallelizing the lightweight AR scoring across team members.

---

### 1. Model and Checkpoint Preparation  
* Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
* Ensure tokenizer compatibility; if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
* Load models in FP16; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.

### 2. Unified Quality Metric (UQM)  
* For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
* Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
* UQM = average of the five normalized scores → dimensionless quality in \([0,1]\) with equal benchmark contribution.

### 3. Compute‑Normalized Efficiency Measurement  
* **Diffusion FLOPs per denoising step:** \(F_{\text{diff}} = \alpha \times |\theta| \times L\) (α≈2 for multiply‑add, \(|\theta|\) = diffusion parameter count, L=128).  
* **AR scorer overhead:** a single forward pass of the frozen AR base model over the same sequence length; empirically ≤ 1 % of diffusion FLOPs and added only when scoring a candidate.  
* **Total FLOPs for a candidate generated with S steps:** \(F_{\text{cand}}(S) = S \times F_{\text{diff}} + F_{\text{AR}}\).  
* **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including candidate generation and scoring) using CUDA events; verify linearity with FLOP estimate (R² > 0.95).  
* Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.

### 4. Adaptive Candidate Generation via AR‑Guided Early Acceptance (ACE)  
For each prompt *p* and each base denoising‑step budget \(S_0 \in \{8,16,32,64,128\}\):  

1. **Initialize** an empty list of accepted candidates, a running score threshold \(\tau_p\) (initially set to \(-\infty\)), and a maximum candidate budget \(N_{\max}=8\).  
2. **Iterate** up to \(N_{\max}\) times:  
   a. Sample random noise \(\mathbf{z}_T\) and run the diffusion model for exactly \(S_0\) denoising steps to obtain a candidate sequence \(\mathbf{x}^{(i)}\).  
   b. Score \(\mathbf{x}^{(i)}\) with the AR base model: compute average negative log‑likelihood (NLL); define the *AR score* as \(-\text{NLL}\) (higher = better).  
   c. If the AR score \(\geq \tau_p\), **accept** \(\mathbf{x}^{(i)}\) as the final output for this prompt and **terminate** the loop.  
   d. Otherwise, update the running threshold: \(\tau_p \leftarrow \max(\tau_p, \text{AR score}^{(i)})\) (i.e., keep the best score seen so far) and continue to the next iteration.  
3. If the loop ends without acceptance (i.e., all \(N_{\max}\) candidates scored below the final \(\tau_p\)), select the candidate with the highest AR score as the output.  

*The ACE strategy yields an **effective** number of candidates \(N_{\text{eff}} \le N_{\max}\) that depends on how quickly a high‑scoring sample appears. Easy prompts tend to stop early (low compute), hard prompts consume the full budget (high compute).*  

### 5. Pareto Front Construction  
* For each model family (GPT‑2‑based vs. LLaMA‑based) and each \(S_0\) value, collect the pair \((\text{total FLOPs per token}, \text{UQM})\) across all prompts (total FLOPs = \(\sum_i F_{\text{cand}}(S_0)\) for the generated candidates + scoring overhead).  
* Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
* Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  

### 6. Trade‑off Analysis  
* **Marginal quality gain per denoising step:** For each fixed \(N_{\text{eff}}\) (observed per prompt), regress UQM against \(S_0\) and report the slope \(\Delta\text{UQM}/\Delta S\).  
* **Marginal quality gain per additional candidate:** For each fixed \(S_0\), regress UQM against \(\log_2(N_{\text{eff}})\) and report \(\Delta\text{UQM}/\Delta\log_2 N\).  
* **Effect of ACE vs. static reranking:**  
  - Run a baseline where we generate a fixed number of candidates \(N \in \{1,2,4,8\}\) with step budget \(S_0\), score them with the AR model, and pick the highest‑scoring candidate (static reranking).  
  - Compare the ACE frontier to the static‑reranking frontier at matched compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts).  
  - Declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  

### 7. Generalizability Checks  
* Hold‑out evaluation on HumanEval‑plus and MBPP to confirm observations are not benchmark‑specific.  
* Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
* Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the ACE experiment to assess scorer‑agnosticism.  

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  
* **Weeks 1‑2:** environment setup, checkpoint download, tokenizer alignment, FLOP‑profiling harness, implement basic diffusion sampling loop.  
* **Weeks 3‑4:** implement ACE acceptance logic, collect UQM and FLOP data for all \(S_0\) values and both diffusion families.  
* **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions.  
* **Week 6:** bootstrap significance testing and baseline (static reranking) comparison.  
* **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers).  
* **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, ACE‑threshold sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Why this method is substantively different:**  
Unlike the prior static/dynamic *N*,*S* grid (which pre‑fixes the number of candidates) and the gradient‑guidance approach (which continuously perturbs the denoising trajectory with AR scores), ACE treats the AR scorer as a **gate** that decides when to stop generating new candidates. This yields a **prompt‑adaptive effective candidate count** without altering the diffusion dynamics or requiring a two‑stage refinement step. It directly probes the hypothesis that “better exploitation of candidate diversity” (via AR‑guided early acceptance) can close the accuracy headroom more efficiently than simply adding denoising steps, thereby addressing L6 and L9 in a novel, inference‑only fashion.