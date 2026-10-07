**Method: AR‑Guided Tokenwise Adaptive Refinement (ATAR)**  

---

### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs: **DiffuGPT‑S/M**, **DiffuLLaMA‑6.74B**, **Dream‑7B**, **LLaDA‑8B**, **DiffuCoder‑7B**.  
- For each DLM, obtain its corresponding AR base model (GPT‑2‑small/medium for the GPT‑2‑based DLMs; LLaMA‑7B for the LLaMA‑based DLMs).  
- Verify that the diffusion checkpoint and the AR base model share the same tokenizer; if they differ, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load **one diffusion model and its AR scorer** in FP16 on the RTX 6000 Pro Blackwell at a time; this fits comfortably within the 96 GB memory budget.  

### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply **min‑max normalization within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → a dimensionless quality score in [0, 1] where each benchmark contributes equally.  
- In addition to the scalar UQM, report **per‑benchmark normalized scores** to avoid masking domain‑specific failures.  

### 3. Compute‑Normalized Efficiency Measurement  
- **Diffusion FLOPs per denoising step (full sequence)**:  
  \[
  F_{\text{diff}} = \alpha \times |\theta| \times L
  \]  
  where \(\alpha\approx2\) (multiply‑add), \(|\theta|\) = diffusion parameter count, \(L=128\) (fixed sequence length).  
- **Token‑wise active fraction**: during a refinement step only a subset \(a\in[0,1]\) of tokens is processed (the rest are held fixed). The effective diffusion FLOPs for that step become \(a \times F_{\text{diff}}\).  
- **AR scorer FLOPs per token**: a single forward pass of the frozen AR base model over the sequence (≈ 1 % of \(F_{\text{diff}}\)). Entropy computation for masking requires the same forward pass, so we count it once per candidate per refinement iteration.  
- **Total FLOPs for a condition** (see §4 for the iterative procedure):  
  \[
  \text{FLOPs}_{\text{total}} = 
  \sum_{c=1}^{N}\Bigl[
      S_{0}\,F_{\text{diff}} \;+\;
      \sum_{r=1}^{R} a_{c,r}\,F_{\text{diff}} \;+\;
      (S_{0}+R)\,F_{\text{AR}}
  \Bigr]
  \]  
  where  
  * \(S_{0}\) = initial low‑step budget (uniform denoising for all tokens),  
  * \(R\) = number of refinement rounds,  
  * \(a_{c,r}\) = fraction of tokens actively denoised in refinement round \(r\) for candidate \(c\) (determined by the AR‑based uncertainty mask).  
- **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including all denoising passes, masking logic, and AR scoring) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

### 4. AR‑Guided Tokenwise Adaptive Refinement Procedure  

| Symbol | Meaning |
|--------|---------|
| \(N\) | number of initial candidates (∈ {1,2,4,8}) |
| \(S_{0}\) | initial denoising steps applied to **all** tokens (∈ {4,8}) |
| \(R\) | maximum number of refinement rounds (set so that total steps ≤ \(S_{\max}\)) |
| \(\tau\) | entropy threshold (percentile of token‑wise entropy distribution) |
| \(a_{c,r}\) | active‑token fraction for candidate \(c\) in round \(r\) |

**Algorithm (per prompt):**  

1. **Initial low‑step generation**  
   - For each of the \(N\) candidates, run the diffusion model for \(S_{0}\) denoising steps **with full token activity** (\(a=1\)).  
   - Store the resulting token sequences \(\{\mathbf{x}^{(i)}_{0}\}_{i=1}^{N}\).  

2. **Token‑wise uncertainty estimation** (once per candidate)  
   - Feed each \(\mathbf{x}^{(i)}_{0}\) to the frozen AR base model and obtain the token‑level probability distribution \(p^{(i)}_{t}(v)\).  
   - Compute token‑wise entropy:  
     \[
     H^{(i)}_{t} = -\sum_{v} p^{(i)}_{t}(v)\log p^{(i)}_{t}(v)
     \]  
   - For each candidate, derive a binary mask:  
     \[
     M^{(i)}_{t} = \begin{cases}
     1 & \text{if } H^{(i)}_{t} > \tau \\
     0 & \text{otherwise}
     \end{cases}
     \]  
     where \(\tau\) is set **globally** (not per‑batch) as the 75‑th percentile of entropy across **all tokens of all candidates** for the current prompt. This yields an active‑token fraction  
     \[
     a^{(i)} = \frac{1}{L}\sum_{t} M^{(i)}_{t}.
     \]  

3. **Iterative refinement rounds**  
   - For round \(r = 1\) to \(R\):  
     * For each candidate \(i\):  
       - Construct a partially noisy latent \(\mathbf{z}^{(i)}_{T}\) where tokens with \(M^{(i)}_{t}=0\) (high‑confidence) are set to the clean embedding (or a low‑variance Gaussian) and tokens with \(M^{(i)}_{t}=1\) are sampled from the standard diffusion noise.  
       - Run the diffusion denoising network for **one denoising step** (i.e., decrement the diffusion timestep by 1) **only on the active tokens**; the network still processes the full sequence but the gradient w.r.t. inactive tokens is zero‑ed by fixing their noise to the clean value (this is achievable with a standard attention mask that blocks information flow from inactive positions).  
       - Update the token sequence \(\mathbf{x}^{(i)}_{r}\) from the denoised latent.  
     * After the step, optionally recompute entropy masks (every \(k\) rounds, e.g., \(k=2\)) to adapt to changing uncertainties; otherwise keep the mask from step 2 fixed for simplicity.  
   - The total number of refinement steps per candidate is \(R\); the overall denoising budget is \(S = S_{0} + R\).  

4. **Scoring and selection**  
   - Compute the AR‑model negative log‑likelihood (NLL) of each final refined candidate \(\mathbf{x}^{(i)}_{R}\).  
   - Select the candidate with the **lowest NLL** as the model’s output for that prompt.  

**Conditions explored**  

- \(N \in \{1,2,4,8\}\)  
- \(S_{0} \in \{4,8\}\) (kept small to ensure a diverse, cheap pool)  
- Total denoising budget \(S \in \{8,16,32,64,128\}\) → implies \(R = S - S_{0}\) refinement rounds.  
- Entropy mask percentile \(\tau \in \{60\%,70\%,80\%\}\) (to test sensitivity).  

### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based) plot **UQM (y‑axis)** against **total FLOPs per token (x‑axis)** for every \((N,S_{0},S,\tau)\) condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both **higher or equal quality** and **lower or equal compute**.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**:  
  * Vary \(S\) at fixed \(N,S_{0},\tau\) to see the effect of refinement steps.  
  * Vary \(N\) at fixed \(S,S_{0},\tau\) to isolate the candidate‑selection contribution.  
  * Vary \(\tau\) at fixed \(N,S,S_{0}\) to assess sensitivity of the uncertainty mask.  

### 6. Trade‑off Analysis  
- **Marginal quality gain per refinement step**: fit a piecewise‑linear regression of UQM versus \(R\) (holding \(N,S_{0},\tau\) constant) and report \(\Delta\text{UQM}/\Delta R\).  
- **Marginal quality gain per additional candidate**: regress UQM against \(\log_{2}(N)\) (holding \(S,S_{0},\tau\) constant).  
- **Marginal quality gain per unit entropy‑mask aggressiveness**: regress UQM against the active‑token fraction \(\bar{a}\) (average across candidates) to obtain \(\Delta\text{UQM}/\Delta\bar{a}\).  
- **Effect of ATAR vs. baselines**: for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by:  
  1. **Uniform denoising** (\(S_{0}=0\), \(R=S\), full‑token activity each step).  
  2. **Plain candidate reranking** (generate \(N\) candidates with uniform \(S\) steps, score with AR model, pick best).  
  3. **ATAR** (as described).  
  Use **paired bootstrap** (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences.  
- **Statistical significance**: declare a strategy superior if the 95 % bootstrap CI of the UQM difference does **not** contain zero.  
- **Baseline comparison with gradient‑based guidance**: repeat the entire pipeline with the guidance‑driven method (λ > 0) to quantify how much ATAR shifts the frontier relative to that existing inference‑time technique.  

### 7. Generalizability Checks  
- **Hold‑out evaluation** on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
- **Unseen DLM**: test a 3B‑parameter diffusion model from the dLLM zoo (not used in the main analysis) to verify that trends extrapolate across architectures.  
- **Alternative lightweight scorers**: replace the AR base model with (i) a distilled 60M‑parameter Transformer, (ii) a frozen BERT‑style MLM, and (iii) a small reward‑model (e.g., a pretrained classifier fine‑tuned on human preferences). Run ATAR with each scorer to assess scorer‑agnosticism.  
- **Per‑benchmark reporting**: alongside UQM, present per‑benchmark normalized scores to confirm that gains are not confined to a single task type.  

### 8. Resource‑aware Implementation Plan (10‑week timeline)  

| Week | Activities |
|------|------------|
| 1‑2 | Environment setup, checkpoint download, tokenizer alignment, implement entropy computation and masking logic; verify that the diffusion model respects an attention mask that freezes selected tokens (no weight change). |
| 3‑4 | Build the two‑stage denoising loop (initial uniform \(S_{0}\) steps → iterative refinement rounds with token‑wise masks); collect baseline UQM and FLOP data for all \((N,S_{0},S,\tau)\) combos. |
| 5 | Implement Pareto front extraction, AUPC calculation, and marginal‑gain regressions. |
| 6 | Perform bootstrap significance testing and baseline comparisons (uniform denoising, plain reranking, gradient‑based guidance). |
| 7‑8 | Generalization experiments (held‑out prompts, extra DLM, alternative scorers). |
| 9‑10 | Write‑up, visualizations (Pareto plots, marginal‑gain bar charts, entropy‑mask sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

---

## Rationale  

The target paper’s open issues are:  

1. **(L6) Accuracy headroom** – either more denoising steps or better exploitation of candidate diversity can close the gap.  
2. **(L9) Compute‑normalized efficiency claims** – current reports do not clarify whether DLMs truly beat AR models when FLOPs are accounted for.  

Existing inference‑time tricks (Jacobi Forcing, TESS 2 reward guidance) shift compute between denoising and selection but treat the **whole sequence uniformly**.  

**ATAR** attacks the headroom from a **different angle**:  

* It first creates a **diverse, cheap pool** of candidates with few denoising steps (high entropy, high variability).  
* Using the **frozen AR scorer**, it estimates **per‑token uncertainty** (entropy) – a proxy for where the diffusion model is still ambiguous.  
* It then **focuses the remaining denoising budget exclusively on the uncertain tokens**, leaving high‑confidence tokens fixed. Because the diffusion model can be instructed to denoise only a subset of positions via an attention mask, the **effective FLOPs per refinement step are reduced proportionally to the active‑token fraction**.  
* By iterating this process, ATAR **reallocates compute from easy tokens to hard tokens** within each candidate, extracting more quality from a fixed FLOP budget than either (i) uniformly increasing denoising steps for all tokens or (ii) simply generating more full‑length candidates and reranking them.  

The method respects the **strict inference‑only** constraint: no training, no weight updates, and the AR model is used only for forward passes (entropy and scoring). The masking mechanism relies solely on the diffusion model’s ability to accept an arbitrary attention mask – a feature present in the released checkpoints (they already support variable masking for in‑painting or partial denoising).  

By measuring **actual FLOPs** (model size × effective denoising steps × sequence length × active‑token fraction) and validating with wall‑clock latency, ATAR provides a **compute‑normalized efficiency** measurement that directly compares against uniform baselines and plain reranking. The Pareto‑front analysis and bootstrap significance testing quantify whether the adaptive token‑wise allocation **shifts the frontier upward** (higher quality at equal compute) more effectively than the alternatives.  

Finally, the plan is feasible within the ten‑week timeline on a single RTX 6000 Pro Blackwell: the extra overhead of entropy computation and masking is modest, the two‑stage denoising loop adds only a constant factor to the existing candidate‑generation pipeline, and all analysis steps can be parallelized across the three GPU‑enabled team members while the remaining four handle CPU‑only tasks (evaluation harness, result aggregation, write‑up).  

Thus, ATAR offers a **clear, innovative, rigorous, valid, and generalizable** methodology to answer the research question: *How does the quality–compute trade‑off of released diffusion language models change when we vary the denoising‑step budget versus the number of generation candidates, and can a lightweight autoregressive reranker improve the Pareto frontier of quality versus compute at fixed inference budgets?*