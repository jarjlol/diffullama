**Method: Token‑Adaptive Diffusion Sampling with AR Guidance (TADAG)**  

**Rationale**  
The research problem asks how the quality–compute trade‑off of released diffusion language models (DLMs) changes when we vary the denoising‑step budget (**S**) versus the number of generation candidates (**N**), and whether a lightweight autoregressive (AR) reranker can improve the Pareto frontier of quality versus compute at a fixed inference budget. Existing work (Jacobi Forcing, TESS 2 reward guidance, simple N‑candidate reranking) treats compute as a uniform resource that is either spent equally on all denoising steps or on generating more independent samples.  

TADAG introduces a **token‑level adaptive compute mechanism**: the AR scorer is used not only to rank finished candidates but also to **detect, at each denoising step, which token positions are already sufficiently confident** and can be “frozen” (i.e., excluded from further denoising). By freezing confident tokens early, the model spends its denoising budget primarily on the uncertain positions, thereby obtaining higher quality for the same total FLOP budget or, equivalently, reducing the FLOPs needed to reach a given quality level.  

Because the adaptation happens **per token and per step**, the effective denoising investment is no longer a single scalar **S** applied uniformly to all tokens; instead, it is a distribution of step counts across token positions. This gives a new axis of trade‑off that is orthogonal to simply varying **N** (the number of independent candidates) or applying a uniform **S**. By sweeping the confidence threshold that controls freezing, we obtain a family of (N, τ) conditions that trace a Pareto frontier distinct from the uniform‑S frontier. The lightweight AR reranker is then applied to the final candidates to select the best‑scoring output, allowing us to quantify whether the token‑adaptive guidance yields a higher‑quality‑per‑FLOP frontier than merely increasing **N** or uniformly increasing **S**.  

TADAG stays strictly inference‑only, uses only publicly released checkpoints, and can be implemented within the ten‑week timeline on a single RTX 6000 Pro Blackwell because it requires only forward passes of the diffusion and AR models, plus inexpensive masking and token‑level likelihood computations.  

---

### Method Details  

#### 1. Model and Checkpoint Preparation  
- Gather the five released DLMs: DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B.  
- For each DLM, retrieve its corresponding AR base model (GPT‑2‑small/medium for the GPT‑2‑based DLMs; LLaMA‑7B for the LLaMA‑based DLMs).  
- Ensure tokenizer compatibility: if a diffusion checkpoint uses a tokenizer that differs from its AR base, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load each diffusion model and its AR scorer in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion‑AR pair resident at a time to stay within the 96 GB memory budget.  

#### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply **min‑max normalization within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

#### 3. Compute‑Normalized Efficiency Measurement  
- **Diffusion FLOPs per active token per denoising step**:  
  \(F_{\text{diff}} = \alpha \times |\theta_{\text{diff}}|\) with \(\alpha\approx2\) (multiply‑add).  
  The total diffusion FLOPs for a condition are obtained by summing \(F_{\text{diff}}\) over all token‑step pairs that are actually processed (i.e., tokens that are not frozen at that step).  
- **AR‑scorer FLOPs per evaluation**: a single forward pass of the frozen AR base model over the full sequence (length L = 128) yields logits for all positions; the per‑token log‑likelihood is extracted from these logits.  
  \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\). Empirically this is ≤ 2 % of the diffusion FLOPs for the models considered; we measure it on the hardware to obtain an exact correction.  
- **Total FLOPs per condition** (for TADAG with N candidates and confidence threshold τ):  
  \[
  \text{FLOPs}_{\text{total}} = 
  \sum_{i=1}^{N}\;\sum_{t=1}^{S_{\max}}\;\bigl(\#\text{active tokens}_{i,t}\bigr)\times F_{\text{diff}}
  \;+\; N \times S_{\max} \times F_{\text{AR}} .
  \]  
  (The AR scorer is evaluated once per candidate per denoising step to obtain the token‑level likelihoods used for freezing decisions.)  
- **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including candidate generation, token‑level scoring, masking, and selection) using CUDA events; verify linearity between measured latency and the FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

#### 4. Token‑Adaptive Diffusion Sampling with AR Guidance (TADAG)  

For each evaluation prompt:  

1. **Initialize candidates**  
   - For \(i = 1,\dots,N\): draw an independent noise tensor \(\mathbf{z}_T^{(i)} \sim \mathcal{N}(0,\mathbf{I})\).  
   - Set a binary mask \(\mathbf{m}_T^{(i)} = \mathbf{1}_{L}\) (all tokens active).  

2. **Iterate denoising steps** for \(t = T, T-1, \dots, 1\) (where \(T = S_{\max}\) is the maximal denoising‑step budget, e.g., 128):  
   a. **Propagation** – apply one step of the diffusion model’s denoising network **only to the active token positions**:  
      \[
      \mathbf{z}_{t-1}^{(i)} = \text{DiffuseStep}\bigl(\mathbf{z}_{t}^{(i)};\; \theta_{\text{diff}}\bigr) \odot \mathbf{m}_{t}^{(i)} \;+\; \mathbf{z}_{t}^{(i)} \odot (1-\mathbf{m}_{t}^{(i)}),
      \]  
      where \(\odot\) denotes element‑wise multiplication and the mask selects which coordinates are updated.  
   b. **Partial decoding** – obtain a tentative discrete sequence \(\hat{\mathbf{x}}_{0:t}^{(i)}\) by taking the argmax of the diffusion model’s logits for \(\mathbf{x}_0\) (or by a single‑step decoder if the model provides one) **only for the active positions**; frozen positions retain their previously decoded token IDs.  
   c. **AR token‑level scoring** – feed the full tentative sequence \(\hat{\mathbf{x}}_{0:t}^{(i)}\) to the frozen AR base model and obtain the log‑probability of each token:  
      \[
      s^{(i)}_{t,k} = \log p_{\text{AR}}\bigl(\hat{x}^{(i)}_k \mid \hat{x}^{(i)}_{<k}\bigr),\quad k=1,\dots,L .
      \]  
      Higher \(s\) indicates higher AR‑model likelihood (lower surprisal).  
   d. **Confidence computation & freezing** – convert log‑likelihoods to a confidence score, e.g., \(c^{(i)}_{t,k} = \exp(s^{(i)}_{t,k})\).  
      If \(c^{(i)}_{t,k} \ge \tau\) (a pre‑chosen confidence threshold), set the mask entry to zero for all **earlier** steps:  
      \[
      m^{(i)}_{t-1,k} = 0,\quad \text{and}\quad m^{(i)}_{t',k}=0\;\forall\,t' < t .
      \]  
      Tokens that meet the threshold are considered “confident” and are frozen for the remainder of the denoising trajectory.  
   e. **Continue** to the next denoising step with the updated mask.  

3. **Final selection** – after the last step (\(t=0\)) we have \(N\) fully denoised sequences \(\{\mathbf{x}_0^{(i)}\}\).  
   - Compute the **full‑sequence AR score** for each candidate:  
      \[
      S^{(i)}_0 = \frac{1}{L}\sum_{k=1}^{L} \log p_{\text{AR}}\bigl(x^{(i)}_k \mid x^{(i)}_{<k}\bigr).
      \]  
   - Choose the candidate with the highest \(S^{(i)}_0\) as the model’s output for the prompt.  

**Explored conditions**  
- Number of candidates \(N \in \{1,2,4,8\}\).  
- Confidence threshold \(\tau\) varied to produce a range of effective average denoising steps per token (we record the actual average steps \(\bar{S}\) for each condition).  
- For comparison, we also run the **uniform‑S baseline** (standard diffusion sampling with a fixed denoising‑step budget \(S \in \{8,16,32,64,128\}\) and the same N‑candidate reranking procedure).  

#### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based DLMs) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((N,\tau)\) condition of TADAG and for every uniform‑S baseline condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both higher or equal quality and lower or equal compute.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**: (i) varying \(\tau\) at fixed \(N\) to isolate the token‑adaptive denoising contribution; (ii) varying \(N\) at fixed \(\tau\) to isolate the candidate‑generation contribution.  

#### 6. Trade‑off Analysis  
- **Marginal quality gain per additional effective denoising step**: fit a piecewise‑linear regression of UQM versus the measured average steps \(\bar{S}\) (holding \(N\) constant) and report the slope \(\Delta\text{UQM}/\Delta\bar{S}\).  
- **Marginal quality gain per additional candidate**: fit a similar regression of UQM versus \(\log_2(N)\) (holding \(\tau\) constant).  
- **Effect of token‑adaptive guidance vs. uniform‑S + reranking**: compare the TADAG frontier to the uniform‑S baseline frontier at matched compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the difference in UQM.  
- **Statistical significance**: declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  

#### 7. Generalizability Checks  
- Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
- Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM (using its masked‑language‑model likelihood as a proxy score) and repeat the TADAG experiment to assess scorer‑agnosticism.  
- Vary sequence length (e.g., \(L=64,256\)) on a subset of prompts to examine robustness to longer contexts.  
- Evaluate sensitivity to the confidence threshold \(\tau\) and the freezing rule (e.g., using a moving‑average of confidence) via a grid search reported in the ablation section.  

#### 8. Resource‑aware Implementation Plan (10‑week timeline)  

| Week | Activities |
|------|------------|
| 1‑2 | Environment setup, checkpoint download, tokenizer alignment, implement diffusion one‑step function with token‑wise masking, AR scorer forward pass, token‑level likelihood extraction. |
| 3‑4 | Build the TADAG sampling pipeline supporting variable \(N\) and confidence threshold \(\tau\); collect baseline UQM and FLOP data for all TADAG conditions and for the uniform‑S baseline. |
| 5 | Implement Pareto front extraction, AUPC calculation, marginal‑gain regressions. |
| 6 | Bootstrap significance testing and comparison between TADAG and uniform‑S baselines. |
| 7‑8 | Generalization experiments (held‑out prompts, extra DLM, alternative scorers, varying sequence length, threshold/ablation studies). |
| 9‑10 | Write‑up, visualizations (Pareto plots, marginal‑gain bar charts, mask‑usage diagnostics), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

---

**Why TADAG is substantively different**  
- Unlike the previously listed methods (which either vary a uniform denoising‑step budget **S** or the number of independent candidates **N**, and optionally apply a static reranking step), TADAG **allocates compute adaptively at the token level** using the AR model’s likelihood as a confidence signal.  
- The core innovation is the **freezing mechanism**: tokens that the AR model deems likely are exempt from further denoising, allowing the diffusion model to concentrate its steps on uncertain positions. This yields a new trade‑off axis (token‑wise step distribution) that is not reducible to simple changes in **S** or **N**.  
- The method remains inference‑only, uses only released checkpoints, and introduces a clear, measurable way to answer whether the “accuracy headroom’’ is best closed by investing more denoising steps per token, by generating more candidates, or by exploiting token‑level AR guidance.  

By directly measuring the FLOPs saved through token freezing and comparing the resulting Pareto frontier to uniform‑S and candidate‑expansion baselines, TADAG provides a principled, quantitative guide for allocating inference budget in DLMs—addressing the accuracy headroom (L6) and the compute‑normalization gap (L9) identified in the target paper.