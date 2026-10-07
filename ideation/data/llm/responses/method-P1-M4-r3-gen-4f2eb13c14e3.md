**Method:** Adaptive Step‑Skipping with Autoregressive‑Scorer Guidance (ASS‑SG)  
**Rationale:**  

The target paper identifies two gaps in released diffusion language models (DLMs): (L6) an “accuracy headroom’’ that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, leaving it unclear whether DLMs truly beat autoregressive (AR) models when cost is accounted for. Existing inference‑time work either (i) adds gradient‑based guidance at every denoising step (e.g., TESS 2 reward guidance, Jacobi Forcing) or (ii) generates a fixed number of full‑trajectory candidates and reranks them with an AR scorer. Both approaches either spend a relatively constant amount of compute per step or ignore the opportunity to *save* compute on easy‑to‑generate tokens while still using the AR model’s knowledge.

ASS‑SG proposes a **different angle**: use the frozen AR scorer *only* as a cheap, token‑level confidence monitor that decides, **on‑the‑fly**, whether a denoising step is necessary for the current latent representation. When the AR model is highly confident about the next token (high predicted probability), we **skip** the expensive diffusion update for that step and retain the current latent; when confidence is low we perform the full diffusion step. This yields a **variable‑step** diffusion process where the number of actually executed denoising steps adapts to the perceived difficulty of each token, while the AR scorer remains frozen and is used only for a single forward pass per step (no gradients, no training).  

By coupling this adaptive step‑skipping with the generation of multiple independent candidates (N) and final AR‑based reranking, we can directly answer the research question:

* How does the quality–compute trade‑off change when we vary the **denoising‑step budget** (now an *effective* budget that emerges from the skipping policy) versus the **number of generation candidates**?  
* Does a lightweight AR reranker, when employed as an **online confidence guide**, shift the Pareto frontier upward more effectively than simply increasing the nominal step count S or the candidate count N?

Because the AR scorer is consulted at every diffusion step, its contribution to the total FLOP count is explicit and measurable, eliminating the hidden‑overhead problem noted in (L9). The method stays strictly inference‑only, uses only publicly released checkpoints, and can be executed within the ten‑week, single‑GPU budget by limiting sequence length to 128 tokens and parallelizing candidate generation across the three GPU‑enabled team members.

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
- **AR scorer FLOPs per forward pass:** \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\).  
- At each denoising step \(t\) we **always** run the AR scorer forward pass to obtain token‑level logits (needed for the confidence test).  
  - If the confidence test decides to **execute** the diffusion update, we also pay \(F_{\text{diff}}\).  
  - If the test decides to **skip** the diffusion update, we pay only \(F_{\text{AR}}\).  
- Hence, the expected FLOPs for a single step under a skipping probability \(p_{\text{skip}}\) are:  
  \[
  F_{\text{step}} = F_{\text{AR}} + (1-p_{\text{skip}})\,F_{\text{diff}} .
  \]  
- For a condition defined by a **maximum** denoising‑step budget \(S_{\max}\), a skipping threshold \(\tau\) (see §4), and a number of candidates \(N\), the total FLOPs to produce one output token are:  
  \[
  \text{FLOPs}_{\text{token}} = \frac{ \displaystyle\sum_{i=1}^{N}\sum_{t=1}^{S_{\max}} \bigl[ F_{\text{AR}} + (1-\mathbb{I}\{c_{i,t}\ge\tau\})F_{\text{diff}} \bigr] + N \times F_{\text{AR}} }{L},
  \]  
  where \(c_{i,t}\) is the AR scorer’s confidence (max softmax probability) for the predicted token at step \(t\) of candidate \(i\), and the final \(N \times F_{\text{AR}}\) term accounts for the AR scorer’s sequence‑level NLL computation used for reranking.  
- **Wall‑clock validation:** run a 100‑token warm‑up, then measure average latency per generated token (including AR forward passes for confidence testing and final reranking) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond).  

#### 4. Adaptive Step‑Skipping with AR‑Scorer Guidance  
For each prompt and each candidate (indexed by \(i=1\dots N\)):  

1. **Initialize** a random noise tensor \(\mathbf{z}_{S_{\max}}\) (standard diffusion schedule).  
2. **For** denoising step \(t = S_{\max}, S_{\max}-1, \dots, 1\):  
   a. Run the diffusion denoising network on \(\mathbf{z}_t\) to obtain predicted clean‑token logits \(\mathbf{p}_\theta(\mathbf{z}_t)\).  
   b. Derive the **expected token** \(\hat{x}_t = \operatorname{argmax}\mathbf{p}_\theta(\mathbf{z}_t)\) (or the embedding of the expected token distribution).  
   c. Feed \(\hat{x}_t\) (or its embedding) to the frozen AR base model and obtain the next‑token logits \(\mathbf{p}_{\text{AR}}(\hat{x}_t)\).  
   d. Compute **confidence** \(c_t = \max\bigl(\operatorname{softmax}(\mathbf{p}_{\text{AR}}(\hat{x}_t))\bigr)\).  
   e. **If** \(c_t \ge \tau\) (confidence threshold) → **skip** the diffusion update: set \(\mathbf{z}_{t-1} = \mathbf{z}_t\).  
      **Else** → perform the standard diffusion reverse‑process update (e.g., Eq. 11 of Ho et al., 2020) using \(\mathbf{p}_\theta(\mathbf{z}_t)\) as the prediction target to obtain \(\mathbf{z}_{t-1}\).  
3. After the final step \(t=0\), decode \(\mathbf{z}_0\) to obtain the output sequence for candidate \(i\).  
4. **Score** the candidate with the AR scorer (average negative log‑likelihood; lower = better).  

- **Conditions explored:**  
  - Confidence threshold \(\tau \in \{0.5, 0.7, 0.9\}\) (lower τ → more aggressive skipping).  
  - Maximum step budget \(S_{\max} \in \{8, 16, 32, 64, 128\}\).  
  - Number of candidates \(N \in \{1, 2, 4, 8\}\).  

For each \((\tau, S_{\max}, N)\) tuple we record UQM and total FLOPs per token (as defined in §3).  

#### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based) plot **UQM** (y‑axis teenager) against **total FLOPs per token** (x‑axis) for every \((\tau, S_{\max}, N)\) condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both **≥** quality and **≤** compute.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**:  
  - Vary \(\tau\) at fixed \(S_{\max},N\) to see the effect of confidence‑based skipping.  
  - Vary \(S_{\max}\) at fixed \(\tau,N\) to isolate the nominal step‑budget contribution.  
  - Vary \(N\) at fixed \(\tau,S_{\max}\) to isolate the candidate‑selection contribution.  

#### 6. Trade‑off Analysis  
- **Marginal quality gain per executed denoising step:** hold \(\tau,N\) constant, fit a piecewise‑linear regression of UQM versus the *average* number of executed diffusion steps (i.e., \(S_{\max}\times(1-p_{\text{skip}})\)); report slope \(\Delta\text{UQM}/\Delta S_{\text{exec}}\).  
- **Marginal quality gain per additional candidate:** hold \(\tau,S_{\max}\) constant, regress UQM against \(\log_2(N)\); report \(\Delta\text{UQM}/\Delta\log_2(N)\).  
- **Marginal quality gain per unit confidence threshold:** hold \(S_{\max},N\) constant, regress UQM against \(\tau\); report \(\Delta\text{UQM}/\Delta\tau\).  
- **Effect of adaptive skipping vs. plain step increase:** for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by:  
  (a) increasing \(S_{\max}\) with \(\tau=0.5\) (minimal skipping),  
  (b) increasing \(N\) with \(\tau=0.5\) (no skipping),  
  (c) decreasing \(\tau\) (more aggressive skipping) while keeping \(S_{\max},N\) low.  
  Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences.  
- **Statistical significance:** declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does **not** contain zero.  
- **Baseline comparison:** repeat the entire pipeline with the standard reranking approach (generate N candidates with \(\tau=0.0\) → i.e., no skipping, pure diffusion steps, then AR‑rerank) to quantify how much the adaptive‑skipping method shifts the Pareto frontier relative to candidate‑only reranking.  

#### 7. Generalizability Checks  
1. **Hold‑out prompts:** evaluate on HumanEval‑plus and MBPP (not used for any hyper‑parameter sweep) to ensure observations are not benchmark‑specific.  
2. **Unseen DLM:** test an additional diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that the ASS‑SG advantage transfers across architectures.  
3. **Alternative lightweight scorers:** replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the ASS‑SG experiment; compare AUPC shifts to confirm scorer‑agnosticism.  
4. **Longer sequence length:** run a subset of experiments with \(L=256\) (still fitting in GPU memory) to see whether the guidance‑based skipping benefit scales with context size.  

#### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  

| Week | Activities |
|------|------------|
| **1‑2** | Environment setup, download all checkpoints, tokenizer alignment, implement FLOP‑counting harness and latency measurement; develop the adaptive step‑skipping loop (AR forward pass for confidence, conditional diffusion update). |
| **3‑4** | Build candidate‑generation and scoring infrastructure; collect baseline UQM and FLOP data for the full \((\tau, S_{\max}, N)\) sweep across all five DLMs. |
| **5** | Pareto‑front extraction, AUPC calculation, marginal‑gain regressions (piecewise‑linear, log‑linear, linear in τ). |
| **6** | Bootstrap significance testing (baseline vs. ASS‑SG, ASS‑SG vs. candidate‑only reranking) and sensitivity analysis for τ. |
| **7‑8** | Generalization experiments: hold‑out prompts, extra DLM, alternative scorers, longer sequence length. |
| **9‑10** | Write‑up, visualisations (Pareto plots, marginal‑gain bar charts, τ‑sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

All steps respect the **inference‑only** rule, use only a single RTX 6000 Pro Blackwell GPU, and can be parallelized across the three GPU‑enabled team members (different model families or hyper‑parameter slices) while the remaining four handle CPU‑only tasks such as evaluation harness construction, result aggregation, and analysis.

---

### Why ASS‑SG Is Different  

- **Mechanism:** Unlike the first proposal (static N,S sweep with post‑hoc AR reranking) and the second proposal (gradient‑based guidance at every step), ASS‑SG uses the AR scorer **only as a confidence monitor** that decides whether to *skip* a costly diffusion update. No gradients are added, and the AR model is not fused into the denoising logits.  
- **Compute Model:** The FLOP accounting explicitly captures the trade‑off between skipped diffusion steps (saving \(F_{\text{diff}}\)) and the unavoidable AR forward pass (\(F_{\text{AR}}\)) each step, providing a fine‑grained, compute‑normalized view of the quality‑efficiency frontier.  
- **Research Question Focus:** By varying the confidence threshold \(\tau\) we directly manipulate the *effective* denoising‑step budget, allowing us to answer whether the accuracy headroom is better closed by **spending more compute on denoising** or by **leveraging the AR scorer to avoid unnecessary work** while still generating diverse candidates.  
- **Novelty Angle:** Adaptive compute allocation based on token‑level confidence from a frozen AR model has not been applied to discrete diffusion language model inference in the published literature, making this a distinct angle from both gradient‑guidance and static reranking approaches.  

Thus, ASS‑SG provides a clear, innovative, rigorous, valid, and generalizable methodology to investigate the quality–compute trade‑off of released DLMs and to test whether a lightweight AR reranker—when used as an online skipping guide—can improve the Pareto frontier of quality versus compute at fixed inference budgets.