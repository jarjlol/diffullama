**Method:** Adaptive Denoising‑Budget Allocation via an Autoregressive Scorer (ADBAS)  
**Rationale:** The target paper identifies two open issues: (L6) a sizable “accuracy headroom’’ that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, making it unclear whether DLMs truly beat AR models when cost is accounted for. Existing inference‑time work (Jacobi Forcing, TESS 2 reward guidance) shows that compute can be reallocated between denoising and selection, but it remains unknown whether the headroom is best addressed by (i) raising the denoising‑step budget *S* or (ii) generating and reranking more samples *N* using a tiny AR scorer.  

ADBAS offers a **different inference‑time mechanism**: instead of fixing the number of full‑length candidates ahead of time (static reranking) or injecting AR scores into the denoising dynamics (guidance‑driven diffusion), we start with a modest pool of parallel diffusion trajectories, **evaluate them after each denoising step with a frozen AR base model**, and **dynamically allocate the remaining denoising budget to the highest‑scoring survivors**. Low‑scoring trajectories are allowed to terminate early, saving compute that is then reinvested in promising candidates. This step‑wise budget reallocation directly probes the trade‑off between investing compute in more denoising steps versus in more (but selectively refined) candidates, while keeping the scorer lightweight and inference‑only. By varying the initial pool size, the promotion fraction, and the base/maximum step budgets we can trace a quality‑compute curve that isolates the returns of denoising‑step investment from candidate‑selection investment, thereby answering the research question and addressing L6 and L9.  

---

### 1. Model and Checkpoint Preparation  
* Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
* Verify tokenizer compatibility: if a diffusion checkpoint’s tokenizer differs from its AR base, load the AR base **with the diffusion model’s tokenizer** (no weight change, inference‑only).  
* Load all models in FP16 on a single **NVIDIA RTX 6000 Ada Generation (96 GB)**; keep only one diffusion model and its AR scorer resident at a time to stay within the memory budget.  

### 2. Unified Quality Metric (UQM) – Fixed‑Reference Normalization  
* For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
* Establish a **reference score** for each benchmark using the **GPT‑2‑small AR baseline** (zero‑shot, greedy decoding) evaluated on the same prompts.  
* Normalize each raw score *r* as  
  \[
  \hat r = \frac{r - r_{\text{ref}}^{\text{min}}}{r_{\text{ref}}^{\text{max}} - r_{\text{ref}}^{\text{min}}},
  \]  
  where \(r_{\text{ref}}^{\text{min}}\) and \(r_{\text{ref}}^{\text{max}}\) are the minimum and maximum reference scores observed across all DLM families and conditions for that benchmark. This yields a dimensionless quality in \([0,1]\) where each benchmark contributes equally and the metric is **independent of the experimental condition set** (no min‑max shift when new conditions are added).  
* UQM = average of the five normalized scores.  
* **Sanity check:** compute the Spearman correlation between AR token‑level NLL (averaged over tokens) and UQM on a held‑out 5 % validation set. If \(|\rho|<0.25\) we will fall back to a hybrid score (AR NLL + length penalty) and report the fallback in the appendix.  

### 3. Compute‑Normalized Efficiency Measurement (FLOP‑accurate)  
* **Diffusion FLOPs per denoising step:** use `fvcore.nn.FlopCountAnalysis` on the diffusion model with a dummy input of shape \([1, L_{\text{seq}}=128, \text{hidden}]\). The analysis automatically accounts for attention, feed‑forward, layer‑norm, and any architecture‑specific modules (Transformer, Mamba, hybrid). Denote this value as \(F_{\text{diff}}\).  
* **AR‑scorer FLOPs per evaluation:** we avoid a full forward pass. Instead, we compute token‑level logits by projecting the expected embedding \(\mathbf{e}_t\) onto the token‑embedding matrix \(\mathbf{W}_E\) of the AR model:  
  \[
  \mathbf{z}_t = \mathbf{W}_E^\top \mathbf{e}_t \quad (\text{size }|\mathcal{V}|),
  \]  
  then apply log‑softmax to obtain log‑probabilities. The cost is a single matrix‑vector multiplication, i.e. \(F_{\text{AR}}^{\text{eval}} = 2 \times |\mathcal{V}| \times d_{\text{model}}\) FLOPs (multiply‑add). Empirically this is ≤ 0.5 % of \(F_{\text{diff}}\) for the model sizes considered.  
* **Total FLOPs for a condition:**  
  \[
  \text{FLOPs}_{\text{total}} = \sum_{t=1}^{S_{\max}} N_{\text{act}}(t) \times F_{\text{diff}} \;+\; \sum_{t=1}^{S_{\max}} N_{\text{act}}(t) \times F_{\text{AR}}^{\text{eval}} \;+\; F_{\text{AR}}^{\text{final}},
  \]  
  where \(N_{\text{act}}(t)\) is the number of candidates still active at step *t* and \(F_{\text{AR}}^{\text{final}}\) is a single forward pass of the AR model over the final decoded sequence (≤ 1 % of diffusion FLOPs).  
* **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including candidate generation, scoring, and any early‑termination logic) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95). Report efficiency as **quality per FLOP** (or quality per millisecond).  

### 4. Adaptive Denoising‑Budget Allocation via AR Scorer (ADBAS)  
For each prompt *p* we define four hyper‑parameters:  

| Symbol | Meaning | Values explored |
|--------|---------|-----------------|
| \(N_0\) | Initial pool size (number of trajectories started from independent Gaussian noise) | \(\{4,8,16,32\}\) |
| \(\alpha\) | Promotion fraction – after each step we keep the top \(\alpha\) fraction of active candidates for the next step; the rest are terminated early | \(\{0.25,0.5,0.75\}\) |
| \(S_0\) | Base denoising‑step budget given to **every** candidate before any promotion decision | \(\{4,8\}\) |
| \(S_{\max}\) | Maximum total denoising steps allowed for any candidate (the overall inference budget) | \(\{16,32,64,128\}\) |

**Algorithm (per prompt):**  

1. **Initialize** \(N_0\) candidates \(\{\mathbf{z}_{S_{\max}}^{(i)}\}_{i=1}^{N_0}\) with independent Gaussian noise at the maximum timestep \(T=S_{\max}\).  
2. **For** denoising step \(t = S_{\max}, S_{\max}-1, \dots, 1\):  
   a. Run the diffusion denoising network on **all currently active candidates** to obtain predicted clean token logits \(\mathbf{p}_\theta(\mathbf{z}_t^{(i)})\).  
   b. Convert logits to a probability distribution and compute the **expected embedding** \(\mathbf{e}_t^{(i)} = \sum_{v} p_{\theta}(v| \mathbf{z}_t^{(i)}) \cdot \mathbf{w}_v\) (where \(\mathbf{w}_v\) is the token‑embedding vector of the AR model).  
   c. **Score** each active candidate with the AR base model: compute token‑level logits \(\mathbf{z}_t^{(i),\text{AR}} = \mathbf{W}_E^\top \mathbf{e}_t^{(i)}\), obtain log‑probabilities via log‑softmax, and sum over the sequence length to get the sequence log‑likelihood \(\ell_t^{(i)}\). Define the AR score as \(\ell_t^{(i)}\) (higher = better).  
   d. **Promotion/termination:** sort candidates by AR score, keep the top \(\lceil \alpha \times N_{\text{act}}(t) \rceil\) candidates for the next step; terminate the rest (they will not receive further denoising). If the number of survivors drops to 1, keep that single candidate for all remaining steps.  
3. **After** the final step (\(t=0\)), decode the remaining candidates to token sequences, recompute their full‑sequence AR log‑likelihood (using a standard forward pass of the AR model), and select the highest‑scoring candidate as the model output for the prompt.  

*The total number of diffusion steps actually executed is*  
\[
\text{Steps}_{\text{executed}} = \sum_{t=1}^{S_{\max}} N_{\text{act}}(t),
\]  
*and the overall FLOP cost follows the formula in §3.*  

### 5. Pareto Front Construction  
* For each model family (GPT‑2‑based vs. LLaMA‑based) and each combination \((N_0,\alpha,S_0,S_{\max})\) we compute the pair **(total FLOPs per token, UQM)** averaged over all prompts.  
* Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
* Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
* Additionally, generate **slice‑wise frontiers**: (i) varying \(\alpha\) at fixed \(N_0,S_0,S_{\max}\) to see the effect of promotion aggressiveness; (ii) varying \(S_{\max}\) at fixed \(N_0,\alpha,S_0\) to isolate the denoising‑step contribution; (iii) varying the effective candidate‑budget \(\displaystyle B = \sum_{t} N_{\text{act}}(t)\) at fixed \(\alpha,S_0,S_{\max}\) to isolate the candidate‑selection contribution.  

### 6. Trade‑off Analysis  
* **Marginal quality gain per denoising step:** For each fixed observed effective candidate‑budget \(B\) (averaged over prompts), regress UQM against \(S_{\max}\) and report the slope \(\Delta\text{UQM}/\Delta S_{\max}\).  
* **Marginal quality gain per additional candidate‑budget:** For each fixed \(S_{\max}\), regress UQM against \(\log_2 B\) and report \(\Delta\text{UQM}/\Delta\log_2 B\).  
* **Marginal quality gain per unit promotion fraction:** Regress UQM against \(\alpha\) (holding \(N_0,S_0,S_{\max}\) constant) to obtain \(\Delta\text{UQM}/\Delta\alpha\).  
* **Effect of ADBAS vs. baselines:**  
  - **Static reranking baseline:** generate \(N\in\{1,2,4,8\}\) full‑length candidates each with step budget \(S_{\max}\), score them with the AR model (single forward pass per candidate), retain the highest‑scoring candidate.  
  - **Jacobi‑Forcing approximation (inference‑only):** at each denoising step, after obtaining the model’s predicted token distribution, compute the AR score of the predicted sequence and move the latent toward the AR‑guided direction using a small step size (no training required).  
  - **TESS 2‑style reward guidance:** treat the AR scorer as a reward and apply classic classifier‑guidance (gradient of AR log‑likelihood w.r.t. the latent) at every step with a fixed guidance weight \(\lambda\).  
  - For each baseline we sweep the same hyper‑parameter ranges (e.g., \(N\) and \(S_{\max}\) for static reranking; \(\lambda\) and \(S_{\max}\) for guidance; Jacobi‑Forcing uses a single guidance strength).  
  - Compare the ADBAS frontier to each baseline frontier at matched compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using **paired bootstrap** (10 000 resamples of prompts).  
  - Apply **Holm‑Bonferroni correction** for the multiple baseline comparisons; declare a strategy superior if the 95 % confidence interval of the UQM difference does not contain zero after correction.  
* **Diversity monitoring:** report the entropy of the candidate score distribution after each step to ensure that aggressive promotion does not collapse diversity prematurely; if entropy falls below a threshold (e.g., 0.5 nats) we note the condition as “low‑diversity’’ and discuss its impact on quality.  

### 7. Generalizability Checks  
* **Hold‑out evaluation:** repeat the entire pipeline on HumanEval‑plus and MBPP to confirm findings are not benchmark‑specific.  
* **Unseen diffusion model:** test an additional, unseen DLM (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
* **Alternative lightweight scorers:** replace the AR base model with (a) a distilled 60M‑parameter Transformer trained on the same corpus, or (b) a frozen BERT‑style MLM, and repeat the ADBAS experiment to assess scorer‑agnosticism.  
* **Sequence‑length scaling:** repeat a subset of experiments with \(L_{\text{seq}}=256\) and \(512\) tokens (using gradient‑checkpointing to fit memory) to examine whether the Pareto frontier shifts predictably with length.  
* **Cross‑language / code evaluation:** run on a multilingual reasoning benchmark (e.g., XLM‑R‑based reasoning) and a code‑only benchmark (HumanEval in Java) to gauge domain robustness.  

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  
| Week | Activities |
|------|------------|
| 1‑2 | Environment setup, checkpoint download, tokenizer verification, implement FLOP‑count harness (`fvcore`), basic diffusion sampling loop. |
| 3‑4 | Implement AR‑scorer projection (embedding‑matrix dot‑product) and early‑termination logic; collect UQM and FLOP data for all \((N_0,\alpha,S_0,S_{\max})\) combinations on the main evaluation suites for both diffusion families. |
| 5 | Implement baseline methods (static reranking, Jacobi‑Forcing approximation, TESS 2‑style guidance); gather their UQM/FLOP data. |
| 6 | Pareto front extraction, AUPC calculation, marginal‑gain regressions, bootstrap significance testing vs. baselines (with Holm‑Bonferroni). |
| 7 | Diversity monitoring (entropy tracking), ablation of promotion fraction, sensitivity to \(S_0\). |
| 8‑9 | Generalization experiments (held‑out prompts, extra DLM, alternative scorers, longer sequences, multilingual/code tests). |
| 10 | Write‑up, visualizations (Pareto plots, marginal‑gain bar charts, promotion‑fraction sensitivity curves, candidate‑budget histograms), prepare reproducibility package (Dockerfile, scripts, seeded RNG, calibration details). |

### 9. Substantive Difference from Prior Proposals  
* **ACE‑v2** decides **after a full candidate** whether to stop generating more full‑length samples.  
* **Guidance‑Driven Diffusion** injects AR scores **into the denoising dynamics at every step** via gradient updates that modify the latent.  
* **ADBAS** instead **starts with many parallel trajectories**, **scores them after each denoising step with a lightweight AR projection**, and **dynamically reallocates the remaining denoising budget to the top‑scoring survivors**, allowing low‑scoring trajectories to terminate early. This mechanism explores the trade‑off between denoising steps and candidate diversity **without** modifying the diffusion latents (as in guidance) or waiting for full‑length samples (as in ACE‑v2), while remaining strictly inference‑only and using only released checkpoints.  

---  

**Method** ends here.