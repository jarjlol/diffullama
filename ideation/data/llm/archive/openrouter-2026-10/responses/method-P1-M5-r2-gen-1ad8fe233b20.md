**Method:**  
**AR‑Guided Progressive Denoising with Early Acceptance (AG‑PDEA)** – a lightweight, inference‑only strategy that lets the autoregressive (AR) scorer decide, *on‑the‑fly*, how many denoising steps each diffusion candidate should receive.  

**Rationale:**  
The target paper highlights two gaps: (L6) a sizable “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) a lack of compute‑normalized efficiency claims that prevents a fair comparison with AR baselines. Existing inference‑time tricks either (i) add gradient‑based guidance (Method 2) or (ii) prune/refine candidates using token‑wise uncertainty (Method 3). Both approaches still tie the denoising budget to a *uniform* per‑token schedule or to a binary mask, which can waste compute on tokens that are already easy to predict.  

AG‑PDEA treats each diffusion trajectory as an *any‑time* proposal: after a few denoising steps we can already query the frozen AR base model for a quick quality estimate (autoregressive log‑likelihood of the current token sequence). If the estimate is high enough, we **stop** denoising that candidate early; otherwise we continue until a maximum step budget is reached. By allocating more steps only to the candidates that the AR scorer deems promising, we (a) exploit the diversity of the sampled set (we keep many low‑cost proposals alive) and (b) spend the denoising budget where it is most likely to improve quality. Because the AR model is used only for a forward pass (no gradients, no updates), the method stays strictly within the inference‑only constraint and fits the same FLOP‑bookkeeping used in the baseline plans.  

The method therefore directly answers the research question: it reveals whether the accuracy headroom is better closed by (i) uniformly increasing denoising steps, (ii) simply generating more candidates and picking the best, or (iii) letting the AR scorer *gate* the denoising budget per candidate, thereby achieving a higher‑quality point on the Pareto frontier at the same compute.

---

### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
- Verify that each diffusion checkpoint uses the same tokenizer as its AR base; if a mismatch exists, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load all models in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = (average of the five normalized scores). This yields a dimensionless quality score in [0, 1] where each benchmark contributes equally. (Per‑benchmark results are also reported for transparency.)  

### 3. Compute‑Normalized Efficiency Measurement  
- **Diffusion FLOPs per denoising step:**  
  \[
  F_{\text{diff}} = \alpha \times |\theta_{\text{diff}}| \times L,
  \]  
  where \(\alpha\approx2\) (multiply‑add), \(|\theta_{\text{diff}}|\) is the diffusion parameter count, and \(L=128\) is the fixed sequence length.  
- **AR‑scorer FLOPs per forward pass:**  
  \[
  F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L.
  \]  
  Empirically \(F_{\text{AR}} \le 0.01 \times F_{\text{diff}}\) (≤ 1 % of diffusion cost).  
- **Total FLOPs for an AG‑PDEA condition:**  
  For each candidate \(i\) we run a variable number of steps \(s_i\) (determined by the early‑acceptance rule). The total cost is  
  \[
  \text{FLOPs}_{\text{total}} = \sum_{i=1}^{N}\bigl[ s_i \times F_{\text{diff}} + F_{\text{AR}} \bigr] .
  \]  
  The extra \(F_{\text{AR}}\) term accounts for the *single* autoregressive score computed after each denoising step (see §4). Because the scorer is invoked at most once per step per candidate, its overhead remains negligible (< 1 % of the diffusion term).  
- **Wall‑clock validation:** run a 100‑token warm‑up, then measure average latency per generated token (including candidate generation, intermittent AR scoring, and early‑stop logic) using CUDA events; verify linearity between measured latency and FLOP estimate (target \(R^2>0.95\)).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

### 4. AG‑PDEA Sampling Procedure  
For each evaluation prompt we generate \(N\) independent diffusion trajectories. Each trajectory proceeds as follows:

1. **Initialize** a random noise tensor \(\mathbf{z}_T\) (standard diffusion schedule).  
2. **Iterate denoising steps** \(t = T, T-1, \dots, 1\):  
   a. Run the diffusion denoising network to obtain the predicted clean‑token logits \(\mathbf{p}_\theta(\mathbf{z}_t)\).  
   b. Convert logits to a probability distribution and obtain the **expected token embedding** \(\hat{\mathbf{e}}_t = \sum_v p_\theta(v| \mathbf{z}_t) \cdot \mathbf{e}_v\) (where \(\mathbf{e}_v\) is the embedding of token \(v\)).  
   c. Form the **current token sequence** \(\hat{\mathbf{x}}_{1:t}\) by taking the arg‑max of \(\hat{\mathbf{e}}_k\) for each position \(k\le t\) (or, equivalently, by feeding the expected embeddings through the diffusion model’s embedding‑to‑logit head; any deterministic mapping that yields a concrete token sequence works).  
   d. **AR scoring:** feed \(\hat{\mathbf{x}}_{1:t}\) to the frozen AR base model and compute the autoregressive log‑likelihood  
      \[
      \ell^{(i)}_t = \log p_{\text{AR}}(\hat{\mathbf{x}}_{1:t}) .
      \]  
      (This is a standard forward pass; the AR model processes the sequence token‑by‑token using its KV cache, so the cost is exactly \(F_{\text{AR}}\) per call.)  
   e. **Early‑acceptance decision:** if \(\ell^{(i)}_t \ge \tau\) (a pre‑set quality threshold), **stop** denoising this candidate and record its current sequence as the final output for trajectory \(i\). Otherwise continue to the next step.  
   f. If the loop reaches \(t=0\) without early acceptance, the fully denoised sequence \(\hat{\mathbf{x}}_{1:L}\) is taken as the output.  

3. **After all \(N\) candidates have finished**, select the candidate with the highest final AR log‑likelihood \(\ell^{(i)}_{\text{final}}\) (equivalently, lowest NLL) as the model’s output for the prompt.  

**Hyper‑parameter grid:**  
- Maximum denoising‑step budget \(S_{\max} \in \{8,16,32,64,128\}\) (the scheduler is truncated after this many steps if no early acceptance occurs).  
- Number of candidates \(N \in \{1,2,4,8\}\).  
- AR‑score threshold \(\tau \in \{-\infty, -2.0, -1.5, -1.0, -0.5\}\) (where \(\tau=-\infty\) disables early acceptance, i.e., uniform \(S_{\max}\) steps for all candidates).  

The baseline AR reranker corresponds to \(\tau=-\infty\) with \(N>1\) and selection of the highest‑scoring candidate after the full \(S_{\max}\) steps (i.e., standard “best‑of‑N”).  

### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based DLMs) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((S_{\max}, N, \tau)\) condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both higher or equal quality and lower or equal compute.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**: (i) varying \(\tau\) at fixed \(S_{\max},N\) to see the effect of the acceptance threshold; (ii) varying \(S_{\max}\) at fixed \(\tau,N\) to isolate the denoising‑step contribution; (iii) varying \(N\) at fixed \(\tau,S_{\max}\) to isolate the candidate‑selection contribution.  

### 6. Trade‑off Analysis  
- **Marginal quality gain per denoising step:** regress UQM against the *average* number of steps actually used per candidate (holding \(N,\tau\) constant) and report \(\Delta\text{UQM}/\Delta S\).  
- **Marginal quality gain per additional candidate:** regress UQM against \(\log_2(N)\) (holding \(S_{\max},\tau\) constant).  
- **Marginal quality gain per unit threshold:** regress UQM against \(\tau\) (holding \(S_{\max},N\) constant) to capture the effect of tightening or loosening the early‑acceptance criterion.  
- **Comparison with stronger baselines:** for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by:  
  1. **Increase denoising steps only** (\(N=1,\tau=-\infty\), varying \(S_{\max}\)).  
  2. **Increase candidates only** (\(S_{\max}=8,\tau=-\infty\), varying \(N\)).  
  3. **Standard AR reranking** (\(\tau=-\infty\), varying \(N,S_{\max}\)).  
  4. **AG‑PDEA** (jointly varying \(N,S_{\max},\tau\) under the same FLOP ceiling).  
  5. **Jacobi Forcing‑style multi‑block decoding** (implemented as described in the related paper, using the same AR scorer for scoring blocks).  
  6. **TESS 2 reward guidance** (gradient‑based guidance with the AR scorer, as in Method 2).  
  Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences. Apply a **Benjamini‑Hochberg FDR correction** across the six pairwise comparisons to control for multiple testing. Declare a strategy superior if the 95 % corrected confidence interval of the UQM difference does not contain zero.  
- **Diversity measurement:** report the average pairwise token‑level entropy across the \(N\) candidates *before* early stopping, and the effective number of candidates \(\exp\!\bigl(-\sum_i w^{(i)}\log w^{(i)}\bigr)\) where \(w^{(i)}\) is the normalized AR score of candidate \(i\) after its final step (this quantifies weight spread and detects collapse).  

### 7. Generalizability Checks  
- Hold‑out evaluation on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
- Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the AG‑PDEA experiment to assess scorer‑agnosticism.  
- Validate the FLOP‑latency correlation across sequence lengths (L = 128, 256, 512) on the same GPU to assess scalability.  
- Run a small multilingual probe (e.g., XNLI) using the same AR scorer; if the scorer is unavailable for a language, note the limitation and skip that language.  

### 8. Resource‑aware Implementation Plan (10‑week timeline)  
- **Weeks 1‑2:** environment setup, checkpoint download, tokenizer alignment, implement FLOP‑profiling harness and AR‑scorer forward pass (with KV caching for incremental scoring).  
- **Weeks 3‑4:** build the sampling loop that supports variable \(S_{\max}\), \(N\), and \(\tau\); implement early‑acceptance logic and per‑candidate step counting; collect baseline UQM and FLOP data for all \((S_{\max},N,\tau)\) conditions.  
- **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions, and diversity metrics.  
- **Week 6:** bootstrap significance testing with FDR correction, and comparisons against Jacobi Forcing and TESS 2 reward guidance.  
- **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, sequence‑length scaling, multilingual probe).  
- **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, threshold‑sensitivity curves, diversity‑vs‑quality scatter), prepare reproducibility package (Dockerfile, scripts, seeded RNG, pre‑computed noise seeds for rapid re‑scoring).  

**Why AG‑PDEA is substantively different:**  
- Unlike Method 2, it does **not** compute gradients or modify the denoising update; the AR scorer is used only for a *forward‑pass* quality check.  
- Unlike Method 3, it does **not** rely on token‑wise uncertainty masks; decisions are made at the *candidate level* based on a sequence‑level AR score, preserving the full token distribution of each trajectory.  
- Unlike Method 1, it does **not** form a weighted average of candidates; it retains the *discrete* nature of diffusion outputs and selects a single high‑scoring trajectory after adaptive compute allocation.  
- The method introduces a novel **any‑time** diffusion sampling paradigm where the AR scorer gates the denoising budget, directly addressing the accuracy headroom by spending compute where the AR model predicts it will be most effective, while still exploring a diverse set of low‑cost proposals.  

By quantifying how the early‑acceptance threshold \(\tau\) reshapes the Pareto frontier, AG‑PDEA provides a clear, compute‑normalized answer to the research question: it reveals whether the quality‑compute trade‑off of released DLMs is best improved by uniform step increases, by simply generating more candidates, or by letting a lightweight AR scorer dynamically allocate denoising steps per candidate.