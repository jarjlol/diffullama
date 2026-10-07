**Method: Token‑Level Adaptive Refinement via AR Confidence (TARAC)**  

**Rationale**  
The target paper identifies two open issues: (L6) a sizable “accuracy headroom’’ that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) a lack of compute‑normalized efficiency claims that prevents a fair comparison between diffusion language models (DLMs) and their autoregressive (AR) counterparts. Existing inference‑time work (Jacobi Forcing, TESS 2 reward guidance) shows that compute can be shifted between denoising and selection, but it remains unclear whether the headroom is best addressed by (i) uniformly increasing the denoising‑step budget *S* for all tokens, (ii) generating more candidates *N* and reranking them with a tiny AR scorer, or (iii) allocating extra denoising steps **only to those tokens that the AR scorer judges to be uncertain**.  

TARAC proposes a third, fine‑grained strategy: after a cheap, shared denoising base (e.g., *S₀* = 8 steps) we inspect the AR scorer’s per‑token log‑likelihood (or probability) on the partially denoised sequence. Tokens whose AR confidence falls below a data‑driven threshold receive additional denoising refinements (extra steps) while high‑confidence tokens are left untouched. This yields a **prompt‑ and token‑adaptive effective denoising budget** that directly targets the accuracy headroom where the model is most uncertain, without the overhead of scoring every diffusion step or generating many full candidates.  

Because the AR scorer is used **only as a confidence monitor** (no gradient guidance, no modification of the diffusion dynamics), the method stays strictly inference‑only, requires no training or adaptation, and can be implemented with the released checkpoints on a single RTX 6000 Pro Blackwell within a ten‑week schedule. By measuring quality versus the exact FLOP cost of the base steps plus the token‑wise refinements, we can quantify whether this selective‑refinement approach improves the Pareto frontier more effectively than uniformly increasing *S* or *N*, thereby answering the research problem in a novel, compute‑aware way.  

---

### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
- Verify tokenizer compatibility; if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load models in FP16; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → dimensionless quality in \([0,1]\) with equal benchmark contribution.  

### 3. Compute‑Normalized Efficiency Measurement  
- **Diffusion FLOPs per denoising step (full model):**  
  \[
  F_{\text{diff}} = \alpha \times |\theta_{\text{diff}}| \times L
  \]  
  where \(\alpha\approx2\) (multiply‑add), \(|\theta_{\text{diff}}|\) is the diffusion parameter count, and \(L=128\) is the fixed sequence length.  
- **AR scorer FLOPs per token‑level evaluation:** a single forward pass of the frozen AR base model over the same sequence length; empirically ≤ 1 % of \(F_{\text{diff}}\) and added **once per token‑confidence query**.  
- **Total FLOPs for a prompt:**  
  \[
  F_{\text{total}} = S_{0}\times F_{\text{diff}} \times L_{\text{tokens}} \;+\; \sum_{t=1}^{L}\big( n_{t}\times F_{\text{diff}} \big) \;+\; L \times F_{\text{AR}}
  \]  
  where \(S_{0}\) is the shared base denoising budget, \(n_{t}\in\{0,1,2,\dots\}\) is the number of **extra** denoising steps allocated to token *t* after the confidence check, and \(F_{\text{AR}}\) is the AR forward‑pass cost (same for every token).  
- **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including base diffusion, confidence queries, and extra refinement steps) using CUDA events; verify linearity with the FLOP estimate (\(R^{2}>0.95\)).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

### 4. Token‑Level Adaptive Refinement via AR Confidence (TARAC)  
For each prompt *p* and each **base denoising budget** \(S_{0}\in\{8,16,32,64\}\):  

1. **Base diffusion:**  
   - Sample random noise \(\mathbf{z}_{T}\).  
   - Perform **exactly** \(S_{0}\) denoising steps using the diffusion model, obtaining a semi‑denoised latent \(\mathbf{z}_{S_{0}}\).  

2. **Token‑level confidence scoring:**  
   - Decode \(\mathbf{z}_{S_{0}}\) to a token sequence \(\hat{\mathbf{x}}\) (argmax over the vocabulary; this yields a deterministic provisional output that is sufficient for confidence estimation).  
   - Run the frozen AR base model once over \(\hat{\mathbf{x}}\) to obtain per‑token log‑likelihoods \(\ell_{t}= \log p_{\text{AR}}(x_{t}\mid x_{<t})\).  
   - Convert to confidences \(c_{t}= \exp(\ell_{t})\) (higher = more confident).  

3. **Adaptive refinement allocation:**  
   - Compute a token‑wise confidence threshold \(\tau_{p}\) as the **α‑quantile** (e.g., 30th percentile) of \(\{c_{t}\}\) for the current prompt. This threshold is **prompt‑specific** and computed **only from the base diffusion output**, avoiding any validation‑set leakage.  
   - For each token *t*: if \(c_{t}<\tau_{p}\) allocate **one extra denoising step** (\(n_{t}=1\)); otherwise \(n_{t}=0\).  
   - Optionally, allow a second refinement round: repeat steps 2‑3 on the latent after the first extra step, allocating a second extra step to tokens that remain below a stricter threshold (e.g., 10th percentile). This yields at most two extra steps per token, keeping the total compute bounded.  

4. **Final decoding:**  
   - After allocating \(\{n_{t}\}\), run the diffusion model for the required extra steps **only on the selected token positions**. This is implemented by masking the diffusion update: tokens with \(n_{t}=0\) keep their current latent value, while tokens with \(n_{t}>0\) undergo the additional step(s).  
   - Decode the final latent to obtain the output sequence \(\mathbf{x}^{(p)}\).  

5. **Candidate generation (optional):**  
   - To study the interaction with candidate diversity, repeat the entire TARAC process **N** times with independent noise seeds (N ∈ {1,2,4,8}) and retain the candidate with the highest AR sequence likelihood (sum of \(\ell_{t}\)).  
   - When \(N=1\) the method reduces to pure token‑adaptive refinement; when \(N>1\) we jointly exploit candidate diversity and token‑wise refinement.  

**Key properties**  
- The AR scorer is used **solely as a confidence monitor**; it never modifies the diffusion dynamics (no gradient guidance, no loss back‑propagation).  
- Compute is allocated **where the AR model is uncertain**, directly targeting the accuracy headroom identified in (L6).  
- The method respects the inference‑only constraint, uses only released checkpoints, and adds at most a small, bounded overhead (one AR forward pass per token per refinement round).  

### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based) and each combination of \((S_{0}, N)\), compute the pair (**total FLOPs per token**, **UQM**) averaged over all prompts.  
- Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**: (i) varying \(S_{0}\) at fixed \(N\) to see the effect of base denoising budget; (ii) varying \(N\) at fixed \(S_{0}\) to isolate candidate diversity; (iii) varying the confidence‑threshold quantile (e.g., 10th, 30th, 50th percentile) to assess the sensitivity of token‑wise refinement.  

### 6. Trade‑off Analysis  
- **Marginal quality gain per base denoising step:** For each fixed \(N\) and confidence‑threshold quantile, regress UQM against \(S_{0}\) and report \(\Delta\text{UQM}/\Delta S_{0}\).  
- **Marginal quality gain per additional candidate:** For each fixed \(S_{0}\) and threshold, regress UQM against \(\log_{2}(N)\) and report \(\Delta\text{UQM}/\Delta\log_{2}N\).  
- **Marginal quality gain per unit of token‑wise refinement:** For each fixed \(S_{0}\) and \(N\), regress UQM against the **average number of extra steps per token** (\(\bar{n} = \frac{1}{L}\sum_{t} n_{t}\)) and report \(\Delta\text{UQM}/\Delta\bar{n}\).  
- **Effect of token‑wise refinement vs. uniform step increase:**  
  - Define three compute‑matched strategies at a given budget B (e.g., 1×, 2×, 4× the FLOPs of the base AR model):  
    1. **Uniform‑S:** increase \(S\) while keeping \(N=1\) and no token‑wise refinement.  
    2. **Uniform‑N:** increase \(N\) while keeping \(S=S_{0}\) (lowest base) and no token‑wise refinement.  
    3. **TARAC:** use the base \(S_{0}\) (lowest) and allocate extra steps via the confidence monitor (optionally with \(N>1\)).  
  - Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM difference between TARAC and each uniform strategy at the same compute budget.  
  - Declare a strategy superior if the 95 % bootstrap CI of the UQM difference does **not** contain zero.  
- **Baseline comparison:** Repeat the entire pipeline with the standard reranking approach (generate \(N\) candidates with a uniform \(S\), score with AR model, pick best) to quantify how much TARAC shifts the Pareto frontier relative to candidate‑only reranking.  

### 7. Generalizability Checks  
- **Hold‑out evaluation:** Repeat the full pipeline on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
- **Unseen DLM:** Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
- **Alternative lightweight scorers:** Replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the TARAC experiment to assess scorer‑agnosticism.  
- **Sequence‑length ablation:** Briefly evaluate TARAC at \(L\in\{64,256,512\}\) to confirm that the adaptive refinement heuristic scales with longer generations (re‑compute FLOPs accordingly).  
- **Threshold sensitivity:** Vary the confidence‑threshold quantile (e.g., 5th, 10th, 20th, 30th, 40th percentile) and the number of refinement rounds (1 vs. 2) to ensure robustness of the adaptive behavior.  
- **Cross‑family scoring:** Use a GPT‑2 scorer on LLaMA‑based DLM outputs (and vice‑versa) to test whether the confidence signal transfers across model families.  

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  
- **Weeks 1‑2:** environment setup, checkpoint download, tokenizer alignment, implement diffusion step‑wise loop, AR forward‑pass for token‑level log‑likelihoods, FLOP‑profiling harness.  
- **Weeks 3‑4:** implement base diffusion (\(S_{0}\)), confidence scoring, threshold computation, and token‑wise masking for extra steps; collect UQM and FLOP data for all \((S_{0}, N)\) combos and both diffusion families.  
- **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions.  
- **Week 6:** bootstrap significance testing and baseline (uniform‑S, uniform‑N, standard reranking) comparisons.  
- **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, sequence‑length ablation, threshold sensitivity, cross‑family scoring).  
- **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, confidence‑threshold sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Why this method is substantively different**  
Unlike the prior static/dynamic *(N, S)* grid (which pre‑fixes the number of candidates or uses a two‑stage refinement step) and unlike gradient‑guidance approaches (which continuously perturb the denoising trajectory with AR scores), **TARAC treats the AR scorer as a per‑token confidence monitor that decides **where** to spend extra denoising compute**. This yields a **token‑adaptive effective denoising budget** without altering the diffusion dynamics, without requiring a predetermined number of candidates, and without relying on gradient‑based guidance. It directly investigates whether exploiting the AR model’s uncertainty estimates to allocate refinement steps closes the accuracy headroom more efficiently than uniformly adding steps or candidates, thereby addressing L6 and L9 in a novel, inference‑only fashion.