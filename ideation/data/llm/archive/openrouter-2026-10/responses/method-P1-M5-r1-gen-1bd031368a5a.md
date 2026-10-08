**Method: AR‑Score Weighted Candidate Fusion (AWCF) for Diffusion Language Models**  
**Rationale:** The target paper identifies two gaps: (L6) a sizable “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) a lack of compute‑normalized efficiency claims that prevents a fair comparison with autoregressive (AR) baselines. Existing inference‑time tricks either (i) add gradient‑based guidance (Method 2) or (ii) prune/refine candidates using token‑wise uncertainty (Method 3). Both approaches still treat the denoising process as a uniform per‑token budget or as a binary mask‑based refinement.  

AWCF takes a different angle: it treats the set of diffusion trajectories as a **sampled proposal distribution** and uses the lightweight AR scorer to **produce a soft, score‑based weighting** of those proposals. By forming a token‑wise weighted average of the candidates (instead of selecting a single winner or applying a gradient), AWCF **preserves the full diversity of the sampled set** while **biasing the output toward high‑scoring regions**. This directly attacks the accuracy headroom by making better use of candidate diversity, and it provides a clear, compute‑normalized way to trade off denoising steps against the number of candidates (via the N‑S grid and a temperature‑controlled weighting scheme). Because the AR model is only used for a forward pass (scoring) and never updated, the approach stays strictly within the inference‑only constraint and can be implemented with the same FLOP‑bookkeeping used in the baseline plans.  

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
- **Diffusion FLOPs per denoising step:** \(F_{\text{diff}} = \alpha \times |\theta| \times L\) (α≈2 for multiply‑add, \(|\theta|\) = diffusion parameter count, L=128).  
- **AR‑scorer FLOPs per forward pass:** \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\) (empirically ≤ 1 % of \(F_{\text{diff}}\)).  
- For an AWCF condition with **N** candidates and **S** denoising steps per candidate, total FLOPs per token are:  
  \[
  \text{FLOPs}_{\text{total}} = N \times \bigl[ S \times F_{\text{diff}} + F_{\text{AR}} \bigr].
  \]  
  (The weighting and token‑wise averaging operations are negligible (< 0.1 % of diffusion FLOPs) and are omitted from the formula.)  
- **Wall‑clock validation:** run a 100‑token warm‑up, then measure average latency per generated token (including candidate generation, scoring, and weighting) using CUDA events; verify linearity between measured latency and FLOP estimate (target \(R^2>0.95\)).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

### 4. AWCF Sampling Procedure  
For each evaluation prompt:  

1. **Candidate generation** – draw \(N\) independent noise vectors \(\{\mathbf{z}^{(i)}_T\}_{i=1}^N\) and run the diffusion denoising network for a fixed budget \(S\) steps (e.g., \(S\in\{8,16,32,64,128\}\)), obtaining raw candidates \(\{\mathbf{x}^{(i)}\}\).  

2. **Scoring** – compute the AR‑model log‑likelihood (or equivalently, the negative log‑likelihood) of each *complete* candidate:  
   \[
   s^{(i)} = \log p_{\text{AR}}(\mathbf{x}^{(i)}).
   \]  

3. **Weighting** – convert scores to normalized weights using a temperature‑scaled softmax:  
   \[
   w^{(i)} = \frac{\exp(s^{(i)}/\tau)}{\sum_{j=1}^{N} \exp(s^{(j)}/\tau)},
   \]  
   where \(\tau>0\) controls the sharpness of the weighting (we explore \(\tau\in\{0.2,0.5,1.0,2.0\}\); \(\tau\to\infty\) yields uniform weighting, \(\tau\to0\) approaches hard selection).  

4. **Token‑wise fusion** – for each token position \(t\) (1…L), compute the weighted average of the candidate token‑distribution logits (or embeddings) produced by the diffusion model at the final step:  
   \[
   \bar{\mathbf{h}}_t = \sum_{i=1}^{N} w^{(i)} \, \mathbf{h}^{(i)}_t,
   \]  
   where \(\mathbf{h}^{(i)}_t\) is the hidden‑state/logit vector for token \(t\) in candidate \(i\).  

5. **Decoding** – obtain the final output token sequence by taking the arg‑max (or sampling) from the fused distribution \(\{\bar{\mathbf{h}}_t\}_{t=1}^L\).  

6. **Hyper‑parameter grid** – we vary:  
   - Denoising‑step budget \(S \in \{8,16,32,64,128\}\)  
   - Number of candidates \(N \in \{1,2,4,8\}\)  
   - Temperature \(\tau \in \{0.2,0.5,1.0,2.0\}\)  
   (The baseline AR reranker corresponds to \(\tau\to0\) with hard selection of the highest‑scoring candidate.)  

### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based DLMs) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((S,N,\tau)\) condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both higher or equal quality and lower or equal compute.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**: (i) varying \(\tau\) at fixed \(S,N\) to see the effect of weighting sharpness; (ii) varying \(S\) at fixed \(\tau,N\) to isolate the denoising‑step contribution; (iii) varying \(N\) at fixed \(\tau,S\) to isolate the candidate‑selection contribution.  

### 6. Trade‑off Analysis  
- **Marginal quality gain per denoising step:** regress UQM against \(S\) (holding \(N,\tau\) constant) and report \(\Delta\text{UQM}/\Delta S\).  
- **Marginal quality gain per additional candidate:** regress UQM against \(\log_2(N)\) (holding \(S,\tau\) constant).  
- **Marginal quality gain per temperature unit:** regress UQM against \(1/\tau\) (holding \(S,N\) constant) to capture the effect of moving from uniform weighting toward hard selection.  
- **Comparison with stronger baselines:** for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by:  
  1. **Increase denoising steps only** (\(N=1,\tau\to\infty\), varying \(S\)).  
  2. **Increase candidates only** (\(S=8,\tau\to\infty\), varying \(N\)).  
  3. **Standard AR reranking** (\(\tau\to0\), varying \(N,S\)).  
  4. **AWCF** (jointly varying \(N,S,\tau\) under the same FLOP ceiling).  
  5. **Jacobi Forcing‑style multi‑block decoding** (implemented as described in the related paper, using the same AR scorer for scoring blocks).  
  6. **TESS 2 reward guidance** (gradient‑based guidance with the AR scorer, as in Method 2).  
  Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences. Apply a **Benjamini‑Hochberg FDR correction** across the six pairwise comparisons to control for multiple testing. Declare a strategy superior if the 95 % corrected confidence interval of the UQM difference does not contain zero.  
- **Diversity measurement:** report the average pairwise self‑BLEU (or token‑level entropy) across the \(N\) candidates before weighting, and the effective number of candidates \(\text{exp}(-\sum_i w^{(i)}\log w^{(i)})\) to quantify weight spread and detect collapse.  

### 7. Generalizability Checks  
- Hold‑out evaluation on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
- Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the AWCF experiment to assess scorer‑agnosticism.  
- Validate the FLOP‑latency correlation across sequence lengths (L = 128, 256, 512) on the same GPU to assess scalability.  
- Run a small multilingual probe (e.g., XNLI) to see whether the weighting scheme transfers beyond English (using the same AR scorer; if unavailable, skip and note the limitation).  

### 8. Resource‑aware Implementation Plan (10‑week timeline)  
- **Weeks 1‑2:** environment setup, checkpoint download, tokenizer alignment, implement FLOP‑profiling harness and AR‑scorer forward pass.  
- **Weeks 3‑4:** build the candidate‑generation loop, scoring, temperature‑scaled softmax weighting, and token‑wise fusion; collect baseline UQM and FLOP data for all \((S,N,\tau)\) conditions.  
- **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions, and diversity metrics.  
- **Week 6:** bootstrap significance testing with FDR correction, and comparisons against Jacobi Forcing and TESS 2 reward guidance.  
- **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, sequence‑length scaling, multilingual probe).  
- **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, weighting‑temperature curves, diversity‑vs‑quality scatter), prepare reproducibility package (Dockerfile, scripts, seeded RNG, pre‑computed candidate caches for rapid re‑scoring).  

**Why AWCF is substantively different:**  
- It does **not** rely on gradient‑based guidance (unlike Method 2) nor on token‑wise uncertainty masking (unlike Method 3).  
- It treats the AR scorer as a **weighting function** for a soft fusion of multiple diffusion trajectories, thereby explicitly leveraging the *distribution* of candidate scores rather than only the top‑1 score or a binary mask.  
- The compute allocation is **transparent**: each additional candidate incurs a linear FLOP increase, and the temperature parameter lets us smoothly interpolate between uniform averaging (maximum diversity) and hard selection (maximum exploitation) without extra denoising steps.  
- The method remains fully inference‑only, uses only publicly released checkpoints, and fits within the stipulated hardware and timeline constraints.  

By quantifying how score‑weighted candidate fusion moves the Pareto frontier, AWCF directly answers the research question: it reveals whether the accuracy headroom is better closed by spending compute on more denoising steps, on more candidates, or on a principled combination of both via lightweight AR‑guided weighting.