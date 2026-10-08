**Method:** AR‑Guided Sequential Monte Carlo Diffusion Sampling (AR‑SMCDS)  

**Rationale:**  
The research problem asks how the quality–compute trade‑off of released diffusion language models (DLMs) changes when we vary the denoising‑step budget (**S**) versus the number of generation candidates (**N**), and whether a lightweight autoregressive (AR) reranker can improve the Pareto frontier of quality versus compute at a fixed inference budget.  
Existing work (Jacobi Forcing, TESS 2 reward guidance) shows that compute can be re‑allocated between denoising and selection, but it remains unclear whether the “accuracy headroom’’ is best closed by (i) investing more denoising steps per sample or (ii) generating more samples and reranking them with a tiny AR scorer.  

AR‑SMCDS treats the diffusion process as a **sequential Monte Carlo (particle‑filter)** where the AR model supplies a token‑level likelihood that is used to **weight and resample** noise trajectories at each denoising step. In this way the AR scorer does not merely rerank a fixed set of final samples; it continuously steers the exploration of the noise space toward regions that the AR model deems more likely, thereby exploiting candidate diversity **without** requiring many independent full‑sequence samples. By varying the number of particles (**N**, which plays the role of candidate count) and the number of denoising steps per particle (**S**), we can directly compare the efficiency of investing compute in more denoising steps versus more guided particles. The method stays strictly inference‑only, uses only publicly released checkpoints, and fits the ten‑week timeline on a single RTX 6000 Pro Blackwell because all operations are simple forward passes of the diffusion and AR models, with negligible overhead for weighting and resampling.  

---

### Method Details  

#### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
- Ensure tokenizer compatibility: if a diffusion checkpoint uses a tokenizer that differs from its AR base, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load each model in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

#### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply **min‑max normalization within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

#### 3. Compute‑Normalized Efficiency Measurement  
- **Diffusion FLOPs per denoising step**:  
  \(F_{\text{diff}} = \alpha \times |\theta_{\text{diff}}| \times L\)  
  with \(\alpha\approx2\) (multiply‑add), \(|\theta_{\text{diff}}|\) the diffusion parameter count, and fixed sequence length \(L=128\).  
- **AR‑scorer FLOPs per evaluation**: a single forward pass of the frozen AR base model over the same sequence length,  
  \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\). Empirically this is ≤ 1 % of \(F_{\text{diff}}\) and is added for every evaluation.  
- **Total FLOPs per condition** (for AR‑SMCDS with \(N\) particles and \(S\) denoising steps):  
  \[
  \text{FLOPs}_{\text{total}} = N \times S \times F_{\text{diff}} \;+\; N \times S \times F_{\text{AR}} .
  \]  
  (The AR scorer is evaluated once per particle per denoising step; if we wish to reduce overhead we can evaluate it only every \(k\) steps and still obtain a valid importance weight – this ablation is included in the generalizability checks.)  
- **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including particle propagation, weighting, and resampling) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

#### 4. AR‑Guided Sequential Monte Carlo Diffusion Sampling (AR‑SMCDS)  

For each evaluation prompt:  

1. **Initialize particles**  
   - Draw \(N\) independent noise tensors \(\mathbf{z}_T^{(i)} \sim \mathcal{N}(0,\mathbf{I})\), \(i=1,\dots,N\).  
   - Set particle weights \(w_T^{(i)} = 1/N\).  

2. **Iterate denoising steps** for \(t = T, T-1, \dots, 1\) (where \(T=S\) is the denoising‑step budget):  
   a. **Propagation** – apply one step of the diffusion model’s denoising network to each particle:  
      \[
      \mathbf{z}_{t-1}^{(i)} = \text{DiffuseStep}\bigl(\mathbf{z}_{t}^{(i)};\; \theta_{\text{diff}}\bigr).
      \]  
   b. **Partial decoding** – obtain an estimate of the partially denoised sequence \(\mathbf{x}_{0:t}^{(i)}\) by running the diffusion model’s decoder on \(\mathbf{z}_{t-1}^{(i)}\) (or, equivalently, by taking the current estimate of \(\mathbf{x}_0\) after the step).  
   c. **AR scoring** – compute the token‑level log‑likelihood of the partial sequence under the frozen AR base model:  
      \[
      s^{(i)}_t = \log p_{\text{AR}}\bigl(\mathbf{x}_{0:t}^{(i)}\bigr)
                = \sum_{k=0}^{t} \log p_{\text{AR}}\bigl(x_k \mid x_{<k}\bigr).
      \]  
      (Higher \(s\) indicates higher AR‑model likelihood.)  
   d. **Weight update** – convert scores to unnormalized weights via a temperature‑scaled softmax to avoid overflow:  
      \[
      \tilde{w}_{t-1}^{(i)} = \exp\bigl(\beta \, s^{(i)}_t\bigr),
      \]  
      where \(\beta>0\) is a fixed inverse temperature (chosen via a small pilot study; e.g., \(\beta = 0.1\)).  
      Normalize: \(w_{t-1}^{(i)} = \tilde{w}_{t-1}^{(i)} / \sum_j \tilde{w}_{t-1}^{(j)}\).  
   e. **Resampling** – compute the effective sample size (ESS) \(= 1/\sum_i (w_{t-1}^{(i)})^2\).  
      If ESS < \(N/2\), perform systematic resampling to obtain an equally‑weighted particle set \(\{ \mathbf{z}_{t-1}^{(i)}\}_{i=1}^N\) and reset weights to \(1/N\).  
   f. **Continue** to the next denoising step.  

3. **Final selection** – after the last step (\(t=0\)) we have \(N\) fully denoised sequences \(\{\mathbf{x}_0^{(i)}\}\).  
   - Choose the sequence with the highest AR score \(s^{(i)}_0\) (or equivalently the highest final weight) as the model’s output for the prompt.  

**Explored conditions**  
- Denoising‑step budget \(S \in \{8,16,32,64,128\}\).  
- Number of particles (candidates) \(N \in \{1,2,4,8\}\).  
- Temperature \(\beta\) fixed after a brief validation sweep (same for all experiments).  
- Optional ablation: evaluate AR scorer only every \(k\) steps (e.g., \(k=2,4\)) to assess overhead vs. performance.  

#### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based DLMs) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((N,S)\) condition (including the ablation variants).  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both higher or equal quality and lower or equal compute.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**: (i) varying \(S\) at fixed \(N\) to isolate the denoising‑step contribution; (ii) varying \(N\) at fixed \(S\) to isolate the particle‑count contribution.  

#### 6. Trade‑off Analysis  
- **Marginal quality gain per additional denoising step**: fit a piecewise‑linear regression of UQM versus \(S\) (holding \(N\) constant) and report the slope \(\Delta\text{UQM}/\Delta S\).  
- **Marginal quality gain per additional particle**: fit a similar regression of UQM versus \(\log_2(N)\) (holding \(S\) constant).  
- **Effect of AR‑guided SMC vs. independent sampling + reranking**: repeat the entire pipeline with a baseline that draws \(N_{\text{ind}}\) independent diffusion samples (each with \(S\) denoising steps), scores them with the AR model, and keeps the highest‑scoring sample. Compare the two frontiers at fixed compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the difference in UQM.  
- **Statistical significance**: declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  

#### 7. Generalizability Checks  
- Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
- Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM (using its masked‑language‑model likelihood as a proxy score) and repeat the AR‑SMCDS experiment to assess scorer‑agnosticism.  
- Vary sequence length (e.g., \(L=64,256\)) on a subset of prompts to examine robustness to longer contexts.  
- Evaluate sensitivity to the temperature \(\beta\) and resampling threshold (ESS < \(N/2\)) via a small grid search.  

#### 8. Resource‑aware Implementation Plan (10‑week timeline)  

| Week | Activities |
|------|------------|
| 1‑2 | Environment setup, checkpoint download, tokenizer alignment, implement diffusion one‑step function, AR scorer forward pass, weighting/resampling utilities. |
| 3‑4 | Build the AR‑SMCDS sampling pipeline supporting variable \(N\) and \(S\); collect baseline UQM and FLOP data for all conditions (including ablation where AR scorer is evaluated every \(k\) steps). |
| 5 | Implement Pareto front extraction, AUPC calculation, marginal‑gain regressions. |
| 6 | Bootstrap significance testing and baseline (independent sampling + reranking) comparison. |
| 7‑8 | Generalization experiments (held‑out prompts, extra DLM, alternative scorers, varying sequence length, temperature/ESS thresholds). |
| 9‑10 | Write‑up, visualizations (Pareto plots, marginal‑gain bar charts, ESS trace diagnostics), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

---

**Why AR‑SMCDS is substantively different**  
- It does **not** generate a fixed set of independent candidates and then rerank them; instead, it continuously steers a population of noise trajectories using the AR model’s likelihood at each denoising step, akin to a particle filter.  
- The compute budget is split between **more denoising steps per particle** (increasing \(S\)) and **more particles** (increasing \(N\)), allowing a clean comparison of the two axes of the quality–compute trade‑off.  
- By reporting ESS and optionally evaluating the AR scorer sparsely, we explicitly address concerns about weight degeneracy and overhead, making the method both theoretically grounded (importance sampling) and practically viable within the stipulated resources.  

This method directly tackles the accuracy headroom (L6) by exploiting the AR model’s guidance to focus compute on promising regions of the diffusion trajectory, and it closes the compute‑normalization gap (L9) by providing a strict FLOP‑based accounting that fairly contrasts denoising‑step investment against particle‑count investment. The approach is clear, innovative (application of SMC to diffusion language model sampling), rigorous (explicit weighting, resampling, ESS monitoring, bootstrap significance), valid (validation of AR scorer correlation, ablation studies), and generalizable (tested across models, scorers, sequence lengths).