**Method:**  
**AR‑Guided Shallow Fusion for Diffusion Language Models (ASF‑DLM)** – an inference‑time strategy that blends the next‑token distribution of a frozen autoregressive (AR) scorer with the denoising model’s prediction at each diffusion step via a log‑linear (shallow‑fusion) combination. The fusion weight λ controls how much the AR model influences the reverse process, allowing us to trade compute between denoising steps and AR‑guided steering. After a fixed denoising budget S we generate N independent trajectories (each using the same λ) and select the highest‑scoring candidate according to the AR scorer’s sequence‑level log‑likelihood. By varying λ, S, and N we map the quality‑compute trade‑off and test whether lightweight AR guidance can shift the Pareto frontier upward more effectively than simply increasing S or N alone.

---

**Rationale:**  
The target paper shows that continual pre‑training of AR models yields competitive diffusion language models (DLMs) but leaves two gaps: (L6) an “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, obscuring whether DLMs truly beat AR models when cost is accounted for. Existing inference‑time remedies (Jacobi Forcing, TESS 2 reward guidance, PEAS/ACE) either modify the latent via gradient‑based guidance or rely on post‑hoc reranking of fully denoised candidates.  

Shallow fusion offers a distinct middle ground: it injects AR knowledge directly into the denoising dynamics **without** gradient back‑propagation, incurs only a single forward pass of the AR model per diffusion step (≤ 1 % of diffusion FLOPs), and leaves the diffusion weights untouched—fully satisfying the inference‑only constraint. By conditioning each denoising step on AR predictions, the model can correct early‑stage errors that would otherwise require many extra denoising steps to fix, thereby addressing the accuracy headroom (L6) with far fewer steps. Simultaneously, because the AR scorer is cheap, we can afford to generate multiple candidates (N) and rerank them, testing whether candidate diversity or AR‑guided denoising yields a better quality‑compute trade‑off.  

Formally, ASF‑DLM lets us answer the research question:  
- How does the quality‑compute curve shift when we vary the denoising‑step budget S versus the number of generation candidates N?  
- Does a lightweight AR reranker (used here as a shallow‑fusion guide and/or final selector) improve the Pareto frontier of quality versus compute at fixed inference budgets?  

The method is innovative because it applies a well‑known technique from speech recognition (shallow fusion of an external language model) to the discrete diffusion reverse process—a combination that has not been explored in the DLM literature. It is rigorous through explicit FLOP accounting, wall‑clock validation, Pareto‑front construction, bootstrap significance testing, and generalization checks. It is valid because the AR scorer is frozen, no training occurs, and the quality metric is grounded in established benchmarks. Finally, it is generalizable across model families (GPT‑2‑based vs. LLaMA‑based DLMs), alternative lightweight scorers, longer sequences, and held‑out prompts.  

---  

### Detailed Procedure  

#### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs: **DiffuGPT‑S/M**, **DiffuLLaMA‑6.74B**, **Dream‑7B**, **LLaDA‑8B**, **DiffuCoder‑7B**.  
- For each DLM, retrieve its corresponding AR base model (GPT‑2‑small/medium for the GPT‑2 family; LLaMA‑7B for the LLaMA family).  
- Verify tokenizer compatibility; if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load models in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB budget.  

#### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply **min‑max normalization within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

#### 3. Compute‑Normalized Efficiency Measurement  
- Let \(|\theta|\) be the diffusion model parameter count, \(|\theta_{\text{AR}}|\) the AR scorer parameter count, \(L=128\) the fixed sequence length, and \(\alpha\approx2\) the FLOPs per multiply‑add in a transformer layer.  
- **Diffusion FLOPs per denoising step:** \(F_{\text{diff}} = \alpha \times |\theta| \times L\).  
- **AR scorer FLOPs per forward pass:** \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\) (empirically ≤ 1 % of \(F_{\text{diff}}\) for the model sizes considered).  
- **Shallow‑fusion overhead per step:** one forward pass of the AR model to obtain token‑level logits → \(F_{\text{AR}}\). No backward pass is needed.  
- **Total FLOPs per denoising step with fusion:** \(F_{\text{step}} = F_{\text{diff}} + F_{\text{AR}}\).  
- For a condition defined by denoising steps S, number of candidates N, and fusion weight λ, the total FLOPs to produce one output token are:  
  \[
  \text{FLOPs}_{\text{token}} = \frac{S \times F_{\text{step}} + N \times F_{\text{AR}}}{L}
  \]  
  (the final AR forward pass for scoring each candidate is added in the numerator).  
- **Wall‑clock validation:** run a 100‑token warm‑up, then measure average latency per generated token (including AR forward passes for fusion and final scoring) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond).  

#### 4. Shallow‑Fusion Sampling Procedure  
For each prompt:  

1. **Initialize** a random noise tensor \(\mathbf{z}_T\) (standard diffusion schedule).  
2. **For** denoising step \(t = T, T-1, \dots, 1\):  
   a. Run the diffusion denoising network to obtain predicted clean‑token logits \(\mathbf{p}_\theta(\mathbf{z}_t)\).  
   b. **AR guidance:** feed the current expected token sequence (obtained by taking the argmax of \(\mathbf{p}_\theta(\mathbf{z}_t)\) or using the embedding of the expected tokens) to the frozen AR base model, obtain its next‑token logits \(\mathbf{p}_{\text{AR}}(\mathbf{z}_t)\).  
   c. **Shallow‑fusion combination:** compute fused logits  
      \[
      \mathbf{p}_{\text{fused}} = \operatorname{softmax}\bigl(\log \mathbf{p}_\theta(\mathbf{z}_t) + \lambda \log \mathbf{p}_{\text{AR}}(\mathbf{z}_t)\bigr)
      \]  
      where \(\lambda \ge 0\) controls the strength of AR guidance (λ = 0 reduces to vanilla diffusion).  
   d. Sample (or take the expected value of) the next token from \(\mathbf{p}_{\text{fused}}\) and update the latent \(\mathbf{z}_{t-1}\) using the standard diffusion reverse‑process update (e.g., Eq. 11 of Ho et al., 2020) with the fused distribution as the prediction target.  
3. After the final step \(t=0\), decode \(\mathbf{z}_0\) to obtain the output sequence.  

- **Candidate generation:** repeat the above guided sampling process **N** times (independent noise seeds) to obtain N guided candidates.  
- **Selection:** score each candidate with the AR scorer (average negative log‑likelihood, lower = better) and retain the candidate with the highest AR score (equivalently, lowest NLL).  

- **Conditions explored:**  
  - Fusion weight \(\lambda \in \{0.0, 0.2, 0.5, 1.0, 2.0\}\) (λ = 0 corresponds to standard diffusion, i.e., no AR guidance).  
  - Denoising‑step budget \(S \in \{8, 16, 32, 64, 128\}\).  
  - Number of candidates \(N \in \{1, 2, 4, 8\}\).  

For each \((\lambda, S, N)\) tuple we record UQM and total FLOPs per token (as defined in §3).  

#### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based) plot **UQM** (y‑axis) against **total FLOPs per token** (x‑axis) for every \((\lambda, S, N)\) condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both **≥** quality and **≤** compute.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**:  
  - Vary \(\lambda\) at fixed \(S,N\) to see the effect of guidance strength.  
  - Vary \(S\) at fixed \(\lambda,N\) to isolate the denoising‑step contribution.  
  - Vary \(N\) at fixed \(\lambda,S\) to isolate the candidate‑selection contribution.  

#### 6. Trade‑off Analysis  
- **Marginal quality gain per denoising step:** hold \(\lambda,N\) constant, fit a piecewise‑linear regression of UQM versus \(S\); report slope \(\Delta\text{UQM}/\Delta S\).  
- **Marginal quality gain per additional candidate:** hold \(\lambda,S\) constant, regress UQM against \(\log_2(N)\); report \(\Delta\text{UQM}/\Delta\log_2(N)\).  
- **Marginal quality gain per unit guidance weight:** hold \(S,N\) constant, regress UQM against \(\lambda\); report \(\Delta\text{UQM}/\Delta\lambda\).  
- **Effect of guidance vs. pure step increase:** for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by:  
  (a) increasing \(S\) with \(\lambda=0\);  
  (b) increasing \(N\) with \(\lambda=0\);  
  (c) increasing \(\lambda\) while keeping \(S,N\) low.  
  Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences.  
- **Statistical significance:** declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does **not** contain zero.  
- **Baseline comparison:** repeat the entire pipeline with the standard reranking approach (generate N candidates with \(\lambda=0\), score with AR model, pick best) to quantify how much the shallow‑fusion method shifts the Pareto frontier relative to candidate‑only reranking.  

#### 7. Generalizability Checks  
1. **Hold‑out prompts:** evaluate on HumanEval‑plus and MBPP (not used for any hyper‑parameter sweep) to ensure observations are not benchmark‑specific.  
2. **Unseen DLM:** test an additional diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that the ASF‑DLM advantage transfers across architectures.  
3. **Alternative lightweight scorers:** replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the ASF‑DLM experiment; compare AUPC shifts to confirm scorer‑agnosticism.  
4. **Longer sequence length:** run a subset of experiments with \(L=256\) (still fitting in GPU memory) to see whether the guidance benefit scales with context size.  

#### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  

| Week | Activities |
|------|------------|
| **1‑2** | Environment setup, download all checkpoints, tokenizer alignment, implement FLOP‑counting harness and latency measurement; develop shallow‑fusion sampling loop (forward pass of AR model, logit combination). |
| **3‑4** | Build candidate‑generation and scoring infrastructure; collect baseline UQM and FLOP data for the full \((\lambda, S, N)\) sweep across all five DLMs. |
| **5** | Pareto‑front extraction, AUPC calculation, marginal‑gain regressions (piecewise‑linear, log‑linear, linear in λ). |
| **6** | Bootstrap significance testing (uniform baseline vs. ASF‑DLM, ASF‑DLM vs. shallow‑fusion‑only, ASF‑DLM vs. candidate‑only reranking) and sensitivity analysis for λ. |
| **7‑8** | Generalization experiments: hold‑out prompts, extra DLM, alternative scorers, longer sequence length. |
| **9‑10** | Write‑up, visualisations (Pareto plots, marginal‑gain bar charts, λ‑sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

All steps respect the **inference‑only** rule, use only a single RTX 6000 Pro Blackwell GPU, and can be parallelized across the three GPU‑enabled team members (different model families or hyper‑parameter slices) while the remaining four handle CPU‑only tasks such as evaluation harness construction, result aggregation, and analysis.  

---  

**In summary**, ASF‑DLM introduces a novel, gradient‑free mechanism for injecting lightweight autoregressive knowledge into the diffusion reverse process via shallow fusion. By systematically varying the fusion strength, denoising budget, and number of candidates, we obtain a fine‑grained map of the quality‑compute trade‑off and directly test whether AR‑guided denoising (with or without candidate reranking) can lift the Pareto frontier—thereby addressing the accuracy headroom (L6) and the compute‑normalization gap (L9) identified in the target paper. The method is clear, innovative, rigorous, valid, and generalizable, and fits comfortably within the stipulated resource constraints.