**Method: AR‑Logit Fusion Denoising (ALFD) – a lightweight, inference‑only way to inject autoregressive knowledge into each diffusion step**  

**Rationale**  
The target paper shows that continual pre‑training of AR models yields strong diffusion language models (DLMs) but leaves two open questions: (L6) a sizable “accuracy headroom’’ that could be closed by more denoising steps or by better exploiting candidate diversity, and (L9) a lack of compute‑normalized efficiency claims that prevents a fair quality‑vs‑compute comparison with AR baselines. Recent inference‑time tricks (Jacobi Forcing, TESS 2 reward guidance) demonstrate that compute can be shifted between denoising and selection, yet it is still unclear whether the headroom is best addressed by (i) uniformly raising the denoising‑step budget **S**, (ii) generating more candidates **N** and reranking them with a tiny AR scorer, or (iii) a synergistic strategy that uses the AR model *during* the denoising process to steer each step toward higher‑quality tokens.  

ALFD proposes a third, synergistic mechanism: at every denoising step we query the frozen AR base model for token‑level log‑likelihoods (or probabilities) and fuse this information directly into the diffusion model’s prediction distribution. No gradients are back‑propagated through the AR model; the AR scorer acts as a static, lightweight “expert’’ that adjusts the diffusion logits via a simple log‑linear combination (product of probabilities). This yields a **guided‑like** effect without the overhead of a backward pass, keeps the method strictly inference‑only, and lets us study three orthogonal axes of compute allocation:  

1. **Denoising steps (S)** – how many diffusion iterations we run.  
2. **Number of candidates (N)** – how many independent trajectories we sample (each trajectory uses the same ALFD procedure).  
3. **Fusion strength (λ)** – how strongly the AR logits influence the diffusion prediction at each step (λ=0 recovers vanilla diffusion).  

By sweeping S, N, and λ we can construct Pareto fronts of quality versus compute and directly answer whether a lightweight AR reranker (here used as a per‑step fusion expert) improves the frontier more effectively than simply increasing S or N. The method respects the inference‑only constraint, uses only publicly released checkpoints, and fits within the ten‑week, single‑GPU budget because the AR scorer adds only a single forward pass per denoising step (≈ 1 % of diffusion FLOPs).  

---

### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
- Ensure tokenizer compatibility: if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load models in FP16; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → dimensionless quality in \([0,1]\) with equal benchmark contribution.  

### 3. Compute‑Normalized Efficiency Measurement (including ALFD overhead)  
- **Diffusion FLOPs per denoising step (full model):**  
  \[
  F_{\text{diff}} = \alpha \times |\theta_{\text{diff}}| \times L
  \]  
  where \(\alpha\approx2\) (multiply‑add), \(|\theta_{\text{diff}}|\) is the diffusion parameter count, and \(L=128\) is the fixed sequence length.  
- **AR scorer FLOPs per step:** a single forward pass of the frozen AR base model over the same sequence length to obtain token‑level log‑likelihoods. Empirically this is ≤ 1 % of \(F_{\text{diff}}\). No backward pass is performed.  
- **Total FLOPs per denoising step with ALFD:**  
  \[
  F_{\text{step}} = F_{\text{diff}} + F_{\text{AR}}
  \]  
- **Total FLOPs for a prompt:**  
  \[
  F_{\text{total}} = N \times S \times F_{\text{step}} \;+\; N \times F_{\text{AR}}^{\text{final}}
  \]  
  where the final term is a single AR forward pass over the completed sequence to score each candidate for selection (≤ 1 % of diffusion FLOPs per candidate).  
- **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including diffusion steps, AR forward passes, and final scoring) using CUDA events; verify linearity with the FLOP estimate (\(R^{2}>0.95\)).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

### 4. AR‑Logit Fusion Denoising (ALFD) Procedure  
For each prompt *p* we generate *N* independent candidates. Each candidate is produced by the following iterative denoising loop (steps indexed from *T* down to 1, truncated after *S* steps):  

1. **Diffusion prediction:**  
   - Run the diffusion model’s denoising network on the current latent \(\mathbf{z}_t\) to obtain raw logits \(\mathbf{g}_t\in\mathbb{R}^{V}\) (V = vocab size) for each token position.  

2. **AR scoring:**  
   - Convert \(\mathbf{g}_t\) to a probability distribution \(\mathbf{p}^{\text{diff}}_t = \text{softmax}(\mathbf{g}_t)\).  
   - Feed the *current* token sequence estimate (obtained by taking the argmax of \(\mathbf{p}^{\text{diff}}_t\) at each position) to the frozen AR base model and compute token‑level log‑likelihoods \(\ell_{t,i}= \log p_{\text{AR}}(x_i \mid x_{<i})\) for every possible token *i* at each position *t*.  
   - Convert to AR probabilities \(\mathbf{p}^{\text{AR}}_t = \text{softmax}(\boldsymbol{\ell}_t)\) (temperature = 1).  

3. **Logit fusion (product of experts):**  
   - Fuse the two distributions via a weighted product (equivalently, add logits):  
     \[
     \mathbf{p}^{\text{fused}}_t \propto \bigl(\mathbf{p}^{\text{diff}}_t\bigr)^{1-\lambda}\;\bigl(\mathbf{p}^{\text{AR}}_t\bigr)^{\lambda}
     \]  
     where \(\lambda\in[0,1]\) controls the strength of the AR expert (λ=0 → pure diffusion, λ=1 → pure AR).  
   - Convert back to logits for the diffusion update: \(\mathbf{g}^{\text{fused}}_t = \log \mathbf{p}^{\text{fused}}_t\).  

4. **Latent update:**  
   - Use the fused logits \(\mathbf{g}^{\text{fused}}_t\) to compute the predicted clean token embedding (or directly the predicted noise) and perform the standard diffusion update (e.g., Euler or DDIM step) to obtain \(\mathbf{z}_{t-1}\).  
   - Note: the diffusion network itself is **not** re‑run; we only replace its raw logits with the fused logits before the update. This keeps the computational cost identical to a vanilla step plus the AR forward pass.  

5. **Iterate** until *t = 0*, then decode \(\mathbf{z}_0\) to obtain the final token sequence for that candidate.  

After generating *N* candidates, score each with a **single** AR forward pass (average negative log‑likelihood) and retain the candidate with the highest AR score (lowest NLL).  

**Explored hyper‑parameters:**  
- Denoising‑step budget \(S \in \{8,16,32,64,128\}\) (by truncating the schedule).  
- Number of candidates \(N \in \{1,2,4,8\}\).  
- Fusion strength \(\lambda \in \{0.0, 0.2, 0.5, 0.8, 1.0\}\).  

### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based) and each triple \((S,N,\lambda)\), compute the pair (**average total FLOPs per token**, **UQM**) across all prompts.  
- Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**:  
  - (i) Vary \(\lambda\) at fixed \(S,N\) to see the effect of AR fusion strength.  
  - (ii) Vary \(S\) at fixed \(\lambda,N\) to isolate denoising‑step contribution.  
  - (iii) Vary \(N\) at fixed \(\lambda,S\) to isolate candidate‑selection contribution.  

### 6. Trade‑off Analysis  
- **Marginal quality gain per denoising step:** For each fixed \((N,\lambda)\), regress UQM against \(S\) and report \(\Delta\text{UQM}/\Delta S\).  
- **Marginal quality gain per additional candidate:** For each fixed \((S,\lambda)\), regress UQM against \(\log_{2}(N)\) and report \(\Delta\text{UQM}/\Delta\log_{2}N\).  
- **Marginal quality gain per unit fusion strength:** For each fixed \((S,N)\), regress UQM against \(\lambda\) and report \(\Delta\text{UQM}/\Delta\lambda\).  
- **Effect of ALFD vs. uniform strategies at equal compute:**  
  - Define three compute‑matched strategies at a given budget B (e.g., 1×, 2×, 4× the FLOPs of the base AR model):  
    1. **Uniform‑S:** increase \(S\) while keeping \(N=1\) and \(\lambda=0\) (vanilla diffusion).  
    2. **Uniform‑N:** increase \(N\) while keeping \(S=S_{\min}\) (e.g., 8) and \(\lambda=0\).  
    3. **ALFD:** use a low base \(S_{\min}\) (e.g., 8) and a moderate \(\lambda\) (e.g., 0.5), optionally with \(N>1\).  
  - Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM difference between ALFD and each uniform strategy at the same compute budget.  
  - Declare a strategy superior if the 95 % bootstrap CI of the UQM difference does **not** contain zero.  
- **Baseline comparison:** Repeat the entire pipeline with the standard reranking approach (generate \(N\) candidates with \(\lambda=0\), score with AR model, pick best) to quantify how much ALFD shifts the Pareto frontier relative to candidate‑only reranking.  

### 7. Generalizability Checks  
- **Hold‑out evaluation:** Repeat the full pipeline on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
- **Unseen DLM:** Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- **Alternative lightweight scorers:** Replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the ALFD experiment to assess scorer‑agnosticism.  
- **Sequence‑length ablation:** Briefly evaluate ALFD at \(L\in\{64,256,512\}\) to confirm that the adaptive fusion heuristic scales with longer generations (re‑compute FLOPs accordingly).  
- **Fusion‑strength sensitivity:** Vary \(\lambda\) across a finer grid (0.0–1.0 in steps of 0.1) and the number of refinement rounds (if we ever decide to re‑apply AR scoring after a few steps) to ensure robustness.  
- **Cross‑family scoring:** Use a GPT‑2 scorer on LLaMA‑based DLM outputs (and vice‑versa) to test whether the confidence signal transfers across model families.  

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  
- **Weeks 1‑2:** environment setup, checkpoint download, tokenizer alignment, implement diffusion step loop with AR forward pass and logit fusion, FLOP‑profiling harness.  
- **Weeks 3‑4:** implement candidate generation loop (N>1) and final AR scoring; collect UQM and FLOP data for all \((S,N,\lambda)\) combos for both diffusion families.  
- **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions.  
- **Week 6:** bootstrap significance testing and baseline (uniform‑S, uniform‑N, standard reranking) comparisons.  
- **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, sequence‑length ablation, λ‑sensitivity, cross‑family scoring).  
- **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, λ‑sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Why ALFD is substantively different**  
- Unlike the **TARAC** method, which spends extra denoising steps *only on uncertain tokens* via a masking mechanism that is underspecified for discrete diffusion, ALFD modifies the *prediction distribution* at **every** step using a lightweight AR expert, requiring no token‑wise masking or gradient computation.  
- Unlike the **Guidance‑Driven Diffusion Sampling** method, which perturbs the latent via the gradient of the AR log‑likelihood (forward + backward pass per step), ALFD uses only a **forward** pass of the AR model to obtain token‑level probabilities and fuses them with the diffusion logits via a product‑of‑experts rule. This halves the AR overhead and avoids any modification of the diffusion dynamics beyond logit replacement.  
- ALFD treats the AR model as a **static expert** that can be blended with the diffusion predictor at each iteration, enabling a clean investigation of three compute axes (S, N, λ) and directly answering whether a lightweight AR reranker (used as a per‑step expert) improves the Pareto frontier more effectively than uniformly raising S or N.  

By providing a clear, implementable, inference‑only protocol that isolates the contribution of autoregressive knowledge at each denoising step, ALFD offers a novel angle on the quality‑compute trade‑off of released diffusion language models.