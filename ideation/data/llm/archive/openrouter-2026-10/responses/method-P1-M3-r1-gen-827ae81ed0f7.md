**Method:** Adaptive Denoising‑Step Allocation via AR‑Guided Early Stopping (AD‑AES)  
**Rationale:** The target paper highlights two gaps: (L6) an “accuracy headroom’’ that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) a lack of compute‑normalized efficiency claims that prevents a fair comparison between DLMs and their AR counterparts. Existing inference‑time work (e.g., Jacobi Forcing, TESS 2 reward guidance) shows that compute can be shifted between denoising and selection, but it remains unclear whether the headroom is best addressed by (i) raising the denoising‑step budget *S* uniformly for all candidates or (ii) generating and reranking more samples *N* using a tiny AR scorer.  

We propose a fundamentally different inference‑time mechanism: **let the AR scorer decide, on‑the‑fly, when further denoising of a given candidate yields diminishing returns and therefore stop early**. Instead of fixing a global *S* or a fixed number of candidates *N*, we treat each diffusion trajectory as an *any‑time* process whose compute expenditure is guided by the AR scorer’s estimate of improvement. This yields a *prompt‑adaptive effective denoising budget* that automatically allocates more steps to hard‑to‑generate tokens (where the AR score improves slowly) and fewer steps to easy tokens (where the AR score quickly plateaus). By measuring the resulting quality–compute curve we can directly answer whether allocating compute via AR‑guided early stopping improves the Pareto frontier more effectively than simply increasing a uniform *S* or increasing *N*.  

The AD‑AES procedure respects all constraints: it is inference‑only, uses only publicly released checkpoints, requires no training or adaptation, and can be implemented within the ten‑week timeline on a single RTX 6000 Pro Blackwell by fixing sequence length to 128 tokens and parallelizing candidate generation and scoring across team members.

---

### 1. Model and Checkpoint Preparation
- Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
- Verify tokenizer compatibility; if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load models in FP16; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.

### 2. Unified Quality Metric (UQM)
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → dimensionless quality in \([0,1]\) with equal benchmark contribution.

### 3. Compute‑Normalized Efficiency Measurement
- **Diffusion FLOPs per denoising step:** \(F_{\text{diff}} = \alpha \times |\theta| \times L\) (α≈2 for multiply‑add, \(|\theta|\) = diffusion parameter count, L=128).  
- **AR scorer overhead per evaluation:** a single forward pass of the frozen AR base model over the same sequence length; empirically ≤ 1 % of diffusion FLOPs and added each time the AR scorer is queried.  
- **Total FLOPs for a candidate:** \(F_{\text{cand}} = (\text{actual denoising steps used}) \times F_{\text{diff}} + (\#\text{AR evaluations}) \times F_{\text{AR}}\).  
- **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including candidate generation, AR evaluations, and early‑stopping checks) using CUDA events; verify linearity with FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.

### 4. Adaptive Denoising‑Step Allocation via AR‑Guided Early Stopping (AD‑AES)
For each prompt *p* and each **maximum** denoising‑step budget \(S_{\max} \in \{8,16,32,64,128\}\):

1. **Initialize** an empty list of candidates, a step counter \(s = 0\), and a previous AR score \(prev\_score = -\infty\).  
2. **Repeat** up to \(N_{\max}=8\) times (or until a stopping condition described below):  
   a. Sample random noise \(\mathbf{z}_T\).  
   b. **Iterative denoising:** while \(s < S_{\max}\):  
      i. Run the diffusion denoising network for **one** step to obtain \(\mathbf{z}_{s}\).  
      ii. Convert \(\mathbf{z}_{s}\) to a token sequence \(\hat{\mathbf{x}}_s\) (e.g., by taking the argmax over the vocabulary).  
      iii. Compute the AR score of \(\hat{\mathbf{x}}_s\): \(score_s = -\text{NLL}_{\text{AR}}(\hat{\mathbf{x}}_s)\) (higher = better).  
      iv. If \(s>0\) and \(\Delta = score_s - prev\_score < \epsilon\) (where \(\epsilon\) is a small improvement threshold, e.g., \(10^{-3}\) in NLL units) **for two consecutive steps**, break the denoising loop (early stop).  
      v. Otherwise, set \(prev\_score = score_s\) and increment \(s\).  
   c. The denoising process yields a final candidate sequence \(\mathbf{x}^{(i)}\) and a concrete step count \(s_i\) (the number of steps actually executed).  
   d. Store \((\mathbf{x}^{(i)}, s_i, score_{s_i})\).  
   e. **Optional early‑acceptance check:** if the current candidate’s AR score exceeds a dynamic acceptance threshold \(\tau_p\) (initialized to the 75th percentile of scores observed on a small validation set), accept it as the final output for this prompt and stop generating further candidates.  
   f. If not accepted, continue to the next iteration (up to \(N_{\max}\)).  
3. If the loop ends without acceptance, select the candidate with the highest AR score as the output.  
4. **Total compute for the prompt:** sum of \(F_{\text{cand}}(s_i)\) over all generated candidates plus the AR scorer overhead for each evaluation and for the final acceptance check (if used).  

*Key properties:*  
- The AR scorer is used **only as a monitor of improvement**, not to modify the diffusion dynamics.  
- Early stopping yields a **variable effective denoising budget** per candidate that reflects the difficulty of the prompt.  
- The method still explores candidate diversity (via multiple trajectories) but allocates compute where the AR scorer indicates it is most needed.

### 5. Pareto Front Construction
- For each model family (GPT‑2‑based vs. LLaMA‑based) and each \(S_{\max}\) value, collect the pair (**total FLOPs per token**, **UQM**) across all prompts (total FLOPs = Σ F_cand over generated candidates + AR‑scorer overhead).  
- Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  

### 6. Trade‑off Analysis
- **Marginal quality gain per denoising step:** For each fixed observed effective step count (averaged over candidates per prompt), regress UQM against the mean steps used and report \(\Delta\text{UQM}/\Delta S\).  
- **Marginal quality gain per additional candidate:** For each fixed \(S_{\max}\), regress UQM against \(\log_2(N_{\text{eff}})\) (where \(N_{\text{eff}}\) is the actual number of candidates generated before acceptance or exhaustion) and report \(\Delta\text{UQM}/\Delta\log_2 N\).  
- **Effect of AR‑guided early stopping vs. uniform step budget:**  
  - Run a baseline where we generate a fixed number of candidates \(N \in \{1,2,4,8\}\) with a uniform step budget \(S \in \{8,16,32,64,128\}\), score them with the AR model, and pick the highest‑scoring candidate (static reranking).  
  - Compare the AD‑AES frontier to the static‑reranking frontier at matched compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts).  
  - Declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  
- **Effect of candidate‑only increase:** Repeat the baseline with \(N\) varied but \(S\) fixed at a low value (e.g., 8) to isolate the benefit of more candidates versus better exploitation of each candidate’s denoising trajectory.  

### 7. Generalizability Checks
- **Hold‑out evaluation:** Repeat the entire pipeline on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
- **Unseen DLM:** Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- **Alternative lightweight scorers:** Replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the AD‑AES experiment to assess scorer‑agnosticism.  
- **Sequence‑length ablation:** Briefly evaluate AD‑AES at \(L \in \{64, 256\}\) to confirm that the early‑stopping heuristic scales with longer generations.  
- **Threshold sensitivity:** Vary the improvement threshold \(\epsilon\) (e.g., \(\{10^{-4},10^{-3},10^{-2}\}\)) and the acceptance percentile \(\tau_p\) to ensure robustness of the adaptive behavior.  

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)
- **Weeks 1‑2:** environment setup, checkpoint download, tokenizer alignment, implement diffusion step‑wise loop with AR scoring after each step, FLOP‑profiling harness.  
- **Weeks 3‑4:** implement AD‑AES early‑stopping logic, candidate generation loop, and optional acceptance threshold; collect UQM and FLOP data for all \(S_{\max}\) values and both diffusion families.  
- **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions.  
- **Week 6:** bootstrap significance testing and baseline (static reranking) comparison.  
- **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, sequence‑length ablation).  
- **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, early‑stopping sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Why this method is substantively different:**  
Unlike the prior static/dynamic *N*,*S* grid (which pre‑fixes the number of candidates or uses a two‑stage refinement step) and unlike the gradient‑guidance approach (which continuously perturbs the denoising trajectory with AR scores), AD‑AES treats the AR scorer as a **monitor of improvement** that decides **when to halt denoising of an individual trajectory**. This yields a **prompt‑adaptive effective denoising budget** without altering the diffusion dynamics, without requiring a predetermined number of candidates, and without relying on gradient‑based guidance. It directly investigates whether better exploitation of each candidate’s denoising trajectory (via AR‑guided early stopping) can close the accuracy headroom more efficiently than simply adding uniform denoising steps or generating more candidates, thereby addressing L6 and L9 in a novel, inference‑only fashion.