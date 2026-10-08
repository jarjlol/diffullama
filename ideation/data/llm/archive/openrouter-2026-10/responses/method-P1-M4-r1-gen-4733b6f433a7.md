**Method:**  
**Progressive Early Acceptance via an Autoregressive Scorer (PEAS)** – an inference‑time strategy that allocates the denoising budget adaptively by generating a pool of candidates, scoring them with a lightweight AR model after short denoising blocks, and accepting (i.e., terminating denoising for) any candidate whose AR score exceeds a quality‑dependent threshold.  Remaining candidates continue to receive additional denoising steps until a global step budget is exhausted or all candidates are accepted.  The final output is the highest‑scoring accepted candidate (or, if none are accepted, the best‑scoring candidate after the maximum allowed steps).  

---

### Rationale  

The target paper identifies two gaps:  

* **(L6) Accuracy headroom** – quality can be improved either by more denoising steps or by better exploiting the diversity of generated candidates.  
* **(L9) Compute‑normalized efficiency** – existing efficiency claims are not normalized, so it is unclear whether DLMs truly beat AR models when cost is accounted for.  

PEAS attacks the headroom from a *candidate‑centric* angle: instead of uniformly spending the same number of denoising steps on every sample (baseline) or refining only a subset of tokens (AGSD), it decides **per‑candidate** whether further denoising is worthwhile, using the frozen AR model as a cheap proxy for eventual quality.  By accepting strong candidates early, PEAS reallocates the saved steps to the remaining weaker candidates, thereby extracting more value from a fixed compute budget than either (i) simply increasing the denoising step count for all candidates or (ii) generating more candidates and reranking them after a fixed, uniform number of steps.  

Because the AR scorer is only used for *evaluation* (a forward pass) and never for gradient‑based guidance, the method respects the strict inference‑only constraint, adds negligible overhead (≤ 1 % of diffusion FLOPs per evaluation), and works with any released DLM checkpoint without modification.  The unified quality metric and FLOP‑normalized efficiency measurement enable a fair, compute‑aware comparison across model families, and the Pareto‑front analysis directly quantifies whether PEAS shifts the frontier upward (higher quality at equal compute) relative to uniform‑step baselines and simple candidate‑reranking.  

---

## 1. Model and Checkpoint Preparation  

* Assemble the five released DLMs: **DiffuGPT‑S/M**, **DiffuLLaMA‑6.74B**, **Dream‑7B**, **LLaDA‑8B**, **DiffuCoder‑7B**.  
* For each DLM, retrieve its corresponding AR base model (GPT‑2‑small/medium for the GPT‑2 family; LLaMA‑7B for the LLaMA family).  
* Verify tokenizer compatibility; if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
* Load models in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB budget.  

---

## 2. Unified Quality Metric (UQM)  

For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
Apply **min‑max normalization within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
UQM = average of the five normalized scores → a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

---

## 3. Compute‑Normalized Efficiency Measurement  

*Let*  

* \(|\theta|\) – number of diffusion model parameters.  
* \(L = 128\) – fixed sequence length.  
* \(\alpha \approx 2\) – FLOPs per multiply‑add in a transformer layer.  
* \(F_{\text{diff}} = \alpha \times |\theta| \times L\) – FLOPs for **one denoising step** on a full sequence.  
* \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\) – FLOPs for a **single forward pass** of the AR scorer (≈ 1 % of \(F_{\text{diff}}\) for the model sizes considered).  

For a given condition we track the **actual** number of denoising steps executed for each candidate (some may stop early).  
If candidate \(i\) executes \(s_i\) steps, its diffusion FLOP cost is \(s_i \times F_{\text{diff}}\).  
Each time we evaluate a candidate with the AR scorer we add \(F_{\text{AR}}\).  
Total FLOPs per token = \(\displaystyle \frac{\sum_i \bigl(s_i \, F_{\text{diff}} + n^{\text{eval}}_i \, F_{\text{AR}}\bigr)}{L \times N_{\text{prompts}}}\), where \(n^{\text{eval}}_i\) is the number of AR evaluations performed for candidate i.  

Wall‑clock latency per token is measured on the GPU (CUDA events) after a 100‑token warm‑up; linearity with the FLOP estimate is verified (R² > 0.95) to confirm that FLOPs are a valid proxy for compute.  

Efficiency is reported as **quality per FLOP** (or quality per millisecond).  

---

## 4. Progressive Early Acceptance Procedure  

### 4.1 Hyper‑parameters  

| Symbol | Meaning | Typical values explored |
|--------|---------|------------------------|
| \(N\) | Initial number of candidates per prompt | \(\{1,2,4,8\}\) |
| \(S_{\max}\) | Maximum denoising steps allowed per candidate | \(\{8,16,32,64,128\}\) |
| \(B\) | Denoising block size (steps after which we evaluate) | \(\{4,8\}\) |
| \(\tau\) | AR‑score acceptance threshold (lower NLL = better) | set per model from a validation sweep (see below) |

### 4.2 Threshold calibration (inference‑only, no training)  

* Using a small held‑out validation set (e.g., 200 prompts from HumanEval), run the **baseline** method with \(N=1\) and \(S=S_{\max}\).  
* Record the AR NLL scores obtained.  
* Choose \(\tau\) as the **\(p\)-th percentile** (e.g., 80th percentile) of this score distribution.  
* Intuitively, \(\tau\) corresponds to a quality level that a typical candidate reaches after the full budget; candidates that surpass it early are deemed “good enough” to stop denoising.  

### 4.3 Algorithm (per prompt)  

```
Input: prompt p, model M, AR scorer A, hyper‑params (N, S_max, B, τ)
Output: selected sequence y

1. Initialise N candidates:
   For i = 1..N:
       z_i ← random noise (shape L)
       s_i ← 0          # steps executed so far
       active_i ← True

2. While ∃ i with active_i and Σ_i s_i < N * S_max:
       # Denoise a block of B steps (or fewer if near the limit)
       For each i with active_i:
            steps_to_run = min(B, S_max - s_i)
            z_i ← Denoise_M(z_i, steps_to_run)   # standard diffusion reverse process
            s_i ← s_i + steps_to_run

       # Evaluate each active candidate
       For each i with active_i:
            x_i ← Decode(z_i)   # argmax or expected token sequence from logits
            score_i ← -NLL_A(x_i)   # higher = better
            if score_i ≥ τ:
                  accepted_i ← True
                  active_i   ← False
                  store final sequence y_i ← x_i
            else:
                  accepted_i ← False

3. If any candidate was accepted:
       y ← the accepted candidate with highest score_i
   else:
       y ← the active candidate with highest score_i   # none passed τ

4. Return y
```

*The diffusion Denoise_M function executes the standard reverse‑process steps (no modification to the model). The only extra cost is the AR forward pass after each block.*  

### 4.4 Exploration of the hyper‑space  

We sweep over the Cartesian product of \(\{N\}\), \(\{S_{\max}\}\), \(\{B\}\), and the fixed \(\tau\) (derived per model).  Each point yields an empirical pair (UQM, total FLOPs per token).  

---

## 5. Pareto Front Construction  

* For each model family (GPT‑2‑based vs. LLaMA‑based) plot **UQM** (y‑axis) against **total FLOPs per token** (x‑axis) for every hyper‑parameter combination.  
* Derive the **empirical Pareto frontier** by retaining points where no other point has both **≥** quality and **≤** compute.  
* Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
* Additionally, produce **slice‑wise frontiers**:  
  - Vary \(S_{\max}\) at fixed \(N,B,\tau\) to isolate the effect of raising the step ceiling.  
  - Vary \(N\) at fixed \(S_{\max},B,\tau\) to isolate the effect of more candidates.  
  - Vary \(B\) at fixed \(N,S_{\max},\tau\) to assess the impact of evaluation frequency.  

---

## 6. Trade‑off Analysis  

### 6.1 Marginal quality gain per denoising step  

* Hold \(N,B,\tau\) constant.  
* Fit a **piecewise‑linear regression** of UQM versus \(S_{\max}\) (using the points on the Pareto frontier).  
* Report the slope \(\Delta\text{UQM}/\Delta S\) as the average quality increase per additional diffusion step when the early‑acceptance mechanism is active.  

### 6.2 Marginal quality gain per additional candidate  

* Hold \(S_{\max},B,\tau\) constant.  
* Regress UQM against \(\log_2(N)\).  
* Report \(\Delta\text{UQM}/\Delta\log_2(N)\).  

### 6.3 Effect of early acceptance vs. uniform baselines  

* For a set of compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model), compute:  
  - **UQM_uniform**: baseline method with uniform \(S=S_{\max}\) and \(N\) varied to match the budget.  
  - **UQM_PEAS**: PEAS with the same budget (actual FLOPs measured).  
* Use **paired bootstrap** (10 000 resamples of prompts) to obtain a 95 % confidence interval for the difference \(\Delta\text{UQM} = \text{UQM\_PEAS} - \text{UQM\_uniform}\).  
* Declare the improvement significant if the interval does **not** contain zero.  

### 6.4 Sensitivity to the acceptance threshold  

* Repeat the entire sweep for alternative percentile choices (e.g., 70th, 90th) to assess robustness of the Pareto shift.  

---

## 7. Generalizability Checks  

1. **Hold‑out prompts** – evaluate on HumanEval‑plus and MBPP (not used for threshold calibration or main sweeps) to ensure observations are not benchmark‑specific.  
2. **Unseen DLM** – test an additional diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that the PEAS advantage transfers across architectures.  
3. **Alternative lightweight scorers** – replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the PEAS experiment; compare AUPC shifts to confirm scorer‑agnosticism.  
4. **Longer sequence length** – run a subset of experiments with \(L=256\) (still fitting in GPU memory) to see whether the early‑acceptance benefit scales with context size.  

---

## 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  

| Week | Activities |
|------|------------|
| **1‑2** | Environment setup, download all checkpoints, tokenizer alignment, implement FLOP‑counting harness and latency measurement. |
| **3‑4** | Implement the PEAS sampling loop (blockwise denoising, AR evaluation after each block, early‑acceptance logic). Collect baseline UQM and FLOP data for the full hyper‑parameter sweep. |
| **5** | Pareto‑front extraction, AUPC calculation, marginal‑gain regressions (piecewise‑linear and log‑linear). |
| **6** | Bootstrap significance testing (uniform baseline vs. PEAS) and threshold‑sensitivity analysis. |
| **7‑8** | Generalization experiments: hold‑out prompts, extra DLM, alternative scorers, longer sequence length. |
| **9‑10** | Write‑up, visualisations (Pareto plots, marginal‑gain bar charts, threshold‑sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

All steps respect the **inference‑only** rule, use only a single RTX 6000 Pro Blackwell GPU, and can be parallelized across the three GPU‑enabled team members (different model families or hyper‑parameter slices) while the remaining four handle CPU‑only tasks such as evaluation harness construction, result aggregation, and analysis.

---

**In summary**, PEAS provides a clear, innovative, and rigorous answer to the research question: it quantifies how the quality‑compute trade‑off shifts when we vary denoising steps versus the number of candidates, and demonstrates that a lightweight autoregressive reranker can improve the Pareto frontier by allocating compute to the most promising candidates early, thereby closing the accuracy headroom (L6) and delivering a compute‑normalized efficiency advantage (L9).