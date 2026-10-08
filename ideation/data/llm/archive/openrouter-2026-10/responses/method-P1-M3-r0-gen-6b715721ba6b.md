**Method:**  
*Measuring the denoising‑trajectory alignment of a diffusion language model (DLM) to its autoregressive (AR) backbone and using that trajectory to predict whether extra denoising steps or extra sampling + lightweight reranking yields the larger quality‑efficiency gain.*

---

### 1. Model & Backbone Selection  
| DLM (released checkpoint) | Size | Family | AR backbone (frozen) |
|---------------------------|------|--------|----------------------|
| DiffuGPT‑S                | 127 M| GPT‑2  | GPT‑2‑small (124 M) |
| DiffuGPT‑M                | 355 M| GPT‑2  | GPT‑2‑medium (355 M)|
| DiffuLLaMA                | 6.74 B| LLaMA | LLaMA‑2‑7B          |
| Dream‑7B                  | 7 B  | LLaMA | LLaMA‑2‑7B          |

*All checkpoints are publicly available; no further training or fine‑tuning is performed (inference‑only).*

---

### 2. Quantifying **Denoising‑Trajectory Alignment**  

For each DLM‑backbone pair we compute a *step‑wise distributional similarity* between the model’s prediction after *t* denoising steps and the AR backbone’s prediction given the same conditioning prefix.

1. **Prompt subset for alignment measurement** – 200 prompts drawn uniformly from the same pool used later for quality evaluation (HumanEval, GSM8K, SIQA/WinoGrande).  
2. **Forward pass** – For each prompt we run the DLM’s native sampler and record the hidden state after each denoising step *t* ∈ {0, 16, 32, 64, 128, 256}.  
   * Step 0 corresponds to the pure noise input (the model’s initial prediction).  
   * At each *t* we also run the AR backbone in **teacher‑forcing mode** on the *same* prefix (the tokens generated so far by the DLM at step *t*) to obtain the AR next‑token distribution.  
3. **Similarity metric** – Compute the **symmetrised KL divergence** (Jensen‑Shannon divergence, JS) between the DLM’s next‑token distribution *pₜ* and the AR backbone’s distribution *qₜ*:  

   \[
   \text{JS}(pₜ\|qₜ)=\frac{1}{2} \text{KL}(pₜ\|m)+\frac{1}{2}\text{KL}(qₜ\|m),\quad m=\frac{pₜ+qₜ}{2}
   \]

   JS is bounded in [0, 1] and symmetric, making it easier to interpret than raw KL.  
4. **Trajectory summary** – For each model we derive two complementary scalars:  

   * **Alignment Area (AA)** – the average JS divergence across all timesteps (lower = closer overall).  
   * **Convergence Step (CS)** – the smallest *t* at which JS ≤ τ, where τ is a small threshold (e.g., 0.05). If the threshold is never reached, CS is set to the maximum step (256).  
   * **Alignment Slope (AS)** – the negative slope of a linear fit to JS vs. *t* (higher = faster alignment).  

   These three numbers capture **(i)** overall closeness, **(ii)** how quickly the denoising process brings the model into AR‑like territory, and **(iii)** the rate of alignment.  

   All JS computations are performed in **fp16** with activation checkpointing; the AR backbone forward pass is negligible (< 0.2 GPU‑hr per model).  

---

### 3. Candidate Generation & Lightweight AR Reranking  

Exactly as in the baseline protocol:  

* Denoising‑step budgets: **Low = 16**, **Medium = 64**, **High = 256** steps.  
* At each budget generate **N ∈ {1, 2, 4, 8}** independent samples per prompt (different seeds).  
* Prompt pool: 300 instances (100 each from HumanEval, GSM8K, SIQA/WinoGrande) stratified by difficulty (low/medium/high).  
* Tiny AR scorer: GPT‑2‑small (124 M) – average negative log‑likelihood (NLL) of the full sequence; rerank and select the top‑scoring sample.  
* “Denoising‑only” baseline uses the single sample (N = 1) at the given step budget.  

All generation uses **fp16 + activation checkpointing**; for DiffuLLaMA we additionally test a **4‑bit quantised** version *only* for the sampling phase (the full‑precision copy is retained for the alignment measurement so that quantisation does not contaminate JS/AA/CS/AS).

---

### 4. Quality & Efficiency Metrics  

* **Quality** – For each output compute:  
  * HumanEval pass@1 and pass@10 → averaged to a code score.  
  * GSM8K exact‑match accuracy.  
  * SIQA/WinoGrande accuracy (averaged).  
  Normalise each score to [0, 1] across *all* conditions (model × step × N) and take the unweighted mean → **Quality ∈ [0,1]**.  

* **Efficiency** – Measure wall‑clock latency per generated token (including scorer overhead for the reranking condition) via `torch.cuda.Event`. Convert to **Estimated Compute**:  

  \[
  \text{EstCompute} = \text{latency\_per\_token} \times \text{peak\_FP16\_throughput} \;(≈140\text{ TFLOPs})
  \]

* **Quality‑Efficiency Trade‑off (QET)** –  

  \[
  \text{QET} = \frac{\text{Quality}}{\text{EstCompute}}
  \]

  Higher QET = more quality per unit of compute.

---

### 5. Analysis Plan  

#### 5.1 Marginal QET Gains  

* **Denoising gain** – ΔQET₍denoise₎ = QET(high steps, N=1) − QET(low steps, N=1) (also low→medium, medium→high).  
* **Sampling gain** – ΔQET₍sample₎ = QET(medium steps, N=8) − QET(medium steps, N=1) (also 1→4, 4→8).  

#### 5.2 Relating Alignment Trajectory to Gains  

For each DLM we now have a triplet of alignment descriptors (AA, CS, AS) and two gain values (ΔQET₍denoise₎, ΔQET₍sample₎).  

* **Spearman correlation** – Test whether:  
  * Lower AA (better overall alignment) predicts **larger** ΔQET₍sample₎ and **smaller** ΔQET₍denoise₎.  
  * Smaller CS (earlier convergence) predicts the same pattern.  
  * Larger AS (faster alignment slope) predicts the same pattern.  

* **Multiple regression** – Fit a linear model (ordinary least squares) with the three alignment descriptors as predictors and each gain as the outcome, including **model family** (GPT‑2 vs. LLaMA) as a covariate.  

* **Interaction analysis** – Add interaction terms (e.g., AS × family) to see whether the relationship differs between families.  

* **Non‑parametric bootstrap** – Resample prompts (with replacement) 1 000 times, recompute AA/CS/AS, QET, gains, and correlations; report 95 % confidence intervals for each coefficient.  

#### 5.3 Decision‑Rule Simulation  

Using the fitted regression model, predict the expected gain from an **extra denoising step** (ΔQET per additional 16‑step increment) and from an **extra sample** (ΔQET per doubling of N) at a fixed compute budget (e.g., the cost of going from 64→256 steps vs. the cost of going from N=1→N=8 at 64 steps).  

*Allocate the extra compute to whichever predicted gain is larger.*  
Compare the resulting QET to two naïve baselines: (i) always spend extra compute on denoising, (ii) always spend extra compute on sampling + reranking. Report the relative improvement.

---

### 6. Feasibility Check (fits the constraints)  

| Item | Estimate | Reasoning |
|------|----------|-----------|
| **Alignment measurement** | ~ 4 GPU‑hrs | 200 prompts × 6 timesteps × 4 models; fp16 + activation checkpointing; AR backbone forward pass negligible. |
| **Generation & reranking** | ~ 22 GPU‑hrs | 300 prompts × 4 N values × 3 step budgets × 4 models ≈ 14 400 generations; batch‑size = 4, activation checkpointing → ≈ 22 GPU‑hrs (same as prior proposals). |
| **Scoring & metrics** | < 1 GPU‑hr (CPU‑heavy) | NLL scoring on CPU, quality metrics, latency timing. |
| **Bootstrap (1 000 resamples)** | ~ 6 GPU‑hrs | Lightweight recomputation of QET on sampled prompt sets; can run overnight on a single GPU. |
| **Total** | **≈ 32 GPU‑hrs** | Well below the 30‑40 GPU‑hr budget; leaves ample headroom for overhead and repetitions. |
| **Memory** | ≤ 96 GB | fp16 + activation checkpointing keeps 6.74 B model within limit; 4‑bit quantisation used only for sampling ablation. |
| **Timeline** | ~ 8 weeks | 1 wk setup, 2 wks alignment + generation, 2 wks analysis & bootstrap, 2 wks write‑up & validation; 2‑week buffer. |

All steps respect the **inference‑only** rule (no training, continual‑pre‑training, or AR‑to‑diffusion adaptation).

---

### 7. Rationale  

The two previously proposed methods both rely on *static* or *linearly‑dynamic* similarity measures (CKA, SVCCA, linear‑probe, or a simple CKA‑vs‑step slope) to quantify how much AR knowledge the DLM retains. Those metrics capture **geometric similarity** of hidden‑state subspaces but are indirect with respect to the *behavioural* consequence that matters for the quality‑efficiency trade‑off: **how close the model’s predictive distribution is to the AR backbone’s distribution during the denoising process**.

By measuring the **Jensen‑Shannon divergence** between the DLM’s next‑token distribution and the AR backbone’s distribution at multiple denoising timesteps, we obtain a **direct, probabilistic proxy** for the degree to which the model has already “learned” the AR prior at that point in its trajectory.  

* **Alignment Area (AA)** summarises overall distributional closeness across the whole trajectory.  
* **Convergence Step (CS)** tells us at what point the model’s predictions become AR‑like enough (JS ≤ τ) that further denoising yields diminishing returns.  
* **Alignment Slope (AS)** captures the *speed* at which the model approaches the AR prior—fast alignment suggests the denoising process is already efficiently injecting AR knowledge, whereas a shallow slope indicates the model is still far from the AR distribution even after many steps.

If a DLM exhibits **low AA, small CS, and high AS** (i.e., it quickly becomes AR‑like), we expect that **additional denoising steps will bring little extra quality** because the model’s internal denoising is already well‑aligned; instead, **exploiting the model’s inherent candidate diversity** (more samples + lightweight AR reranking) should recover the remaining “accuracy headroom” more efficiently. Conversely, a DLM with **high AA, large CS, low AS** is misaligned throughout the trajectory, implying that **more denoising steps are needed** to bring its predictions toward the AR prior before sampling can be effective.

This mechanistic hypothesis is **distinct** from prior work because it:

1. **Operates in output‑space (next‑token distribution)** rather than representation‑space, linking alignment directly to the quantity that determines generation quality.  
2. **Captures the entire denoising trajectory** (area, convergence point, slope) rather than a single static snapshot or a simple linear trend.  
3. **Provides actionable, inference‑time diagnostics** (AA, CS, AS) that can be computed once per model with a modest prompt subset and then used to decide where to allocate extra inference budget.  

Empirically validating this relationship across model families and denoising‑step budgets will give practitioners a **principled, model‑specific rule** for spending compute on denoising versus sampling + reranking, grounded in a measurable property of the DLM that reflects its retention of AR knowledge. This bridges the representation‑alignment literature (REPR‑ALIGN, PreDiff‑LM) with inference‑time efficiency work (Jacobi Forcing, TESS 2) while offering a novel, easy‑to‑apply tool for scaling diffusion language models.