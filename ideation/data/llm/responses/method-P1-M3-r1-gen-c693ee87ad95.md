**Method: Adaptive Candidate Generation via AR‑Guided Early Acceptance (ACE‑v2)**  
**Rationale:** The target paper highlights two open issues: (L6) a sizable “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, leaving it unclear whether DLMs truly beat AR models when cost is accounted for. Existing inference‑time work (Jacobi Forcing, TESS 2 reward guidance) shows that compute can be reallocated between denoising and selection, but it remains unknown whether the headroom is best addressed by (i) raising the denoising‑step budget *S* or (ii) generating and reranking more samples *N* using a tiny AR scorer.  

We propose a fundamentally different inference‑time strategy: **let the AR scorer decide, on‑the‑fly, whether a newly generated candidate is good enough to stop further sampling**. Instead of fixing *N* ahead of time or continuously perturbing the denoising trajectory with AR scores, ACE‑v2 treats the AR scorer as a *gate* that monitors the quality of generated candidates and halts generation as soon as a candidate surpasses a dynamically‑adjusted quality threshold. This yields an **effective number of candidates** that adapts to prompt difficulty and to the intrinsic quality of the early samples, allowing us to directly measure whether allocating compute to more AR‑guided candidates (via early acceptance) improves the Pareto frontier more effectively than simply increasing the denoising‑step budget *S*.  

ACE‑v2 respects all constraints: it is inference‑only, uses only publicly released checkpoints, requires no training or adaptation, and can be implemented within the ten‑week timeline on a single RTX 6000 Pro Blackwell by fixing sequence length to 128 tokens and parallelizing the lightweight AR scoring across team members.

---

### 1. Model and Checkpoint Preparation  
* Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
* **Tokenizer handling:** Use the tokenizer that ships with each diffusion checkpoint. If a diffusion model’s tokenizer differs from its AR base, we **do not replace it** (to avoid shifting the model out of its trained token space). Instead, we verify that the AR scorer can consume the diffusion model’s token IDs by loading the AR model with the same tokenizer (i.e., we load the AR base *with* the diffusion model’s tokenizer). This stays within the inference‑only rule because no model weights are modified.  
* Load models in FP16; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.

### 2. Unified Quality Metric (UQM)  
* For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
* Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
* UQM = average of the five normalized scores → dimensionless quality in \([0,1]\) with equal benchmark contribution.  
* **Sanity check:** On a held‑out validation set (10 % of prompts) compute the Pearson correlation between AR NLL (averaged over tokens) and UQM; if \(|r|<0.3\) we will flag the scorer as weakly predictive and consider a lightweight hybrid score (AR NLL + length penalty) in an ablation (see §7).

### 3. Compute‑Normalized Efficiency Measurement  
* **Diffusion FLOPs per denoising step:**  
  \[
  F_{\text{diff}} = \sum_{l=1}^{L_{\text{layers}}} \bigl(2 \times C_{\text{in}}^{(l)} \times C_{\text{out}}^{(l)} \times K^{(l)} \times L_{\text{seq}}\bigr)
  \]  
  where the sum runs over all transformer layers, \(C_{\text{in/out}}\) are input/output channel dimensions, \(K^{(l)}\) is the effective kernel size (for attention: \(K = L_{\text{seq}}\); for feed‑forward: \(K = 1\)), and \(L_{\text{seq}}=128\). This formula captures the dominant matrix‑multiply cost and can be computed automatically from each model’s configuration (e.g., using `fvcore.nn.FlopCountAnalysis`).  
* **AR scorer overhead:** a single forward pass of the frozen AR base model over the same sequence length; empirically ≤ 1 % of diffusion FLOPs and added only when scoring a candidate.  
* **Total FLOPs for a candidate generated with S steps:**  
  \[
  F_{\text{cand}}(S) = S \times F_{\text{diff}} + F_{\text{AR}}.
  \]  
* **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including candidate generation and scoring) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
* Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.

### 4. Adaptive Candidate Generation via AR‑Guided Early Acceptance (ACE‑v2)  
For each prompt *p* and each base denoising‑step budget \(S_0 \in \{8,16,32,64,128\}\):

1. **Initialize** an empty list of accepted candidates, a running quality threshold \(\tau_p\), and a maximum candidate budget \(N_{\max}=8\).  
2. **Generate the first candidate:**  
   a. Sample random noise \(\mathbf{z}_T\) and run the diffusion model for exactly \(S_0\) denoising steps to obtain \(\mathbf{x}^{(1)}\).  
   b. Score \(\mathbf{x}^{(1)}\) with the AR base model: compute average negative log‑likelihood (NLL); define the AR score as \(-\text{NLL}\) (higher = better).  
   c. Set \(\tau_p \leftarrow \text{AR score}^{(1)}\) (the threshold starts at the score of the first sample).  
   d. **Accept** \(\mathbf{x}^{(1)}\) as the provisional output and **continue** to step 3 (we allow the possibility of finding a better candidate).  
3. **Iterate** for \(i = 2\) to \(N_{\max}\):  
   a. Sample a new noise seed and generate \(\mathbf{x}^{(i)}\) with the same \(S_0\) steps.  
   b. Compute its AR score \(s^{(i)} = -\text{NLL}(\mathbf{x}^{(i)})\).  
   c. **Improvement test:** if \(s^{(i)} \ge \tau_p + \delta\) (where \(\delta\) is a minimal improvement margin), then **accept** \(\mathbf{x}^{(i)}\) as the final output, **terminate** the loop, and set the effective number of candidates \(N_{\text{eff}} = i\).  
   d. Otherwise, **update** the running threshold to the best score seen so far: \(\tau_p \leftarrow \max(\tau_p, s^{(i)})\) and continue to the next iteration.  
   e. **Early‑stop patience:** if we have observed \(P\) consecutive candidates without improvement (i.e., \(s^{(i)} < \tau_p + \delta\) for \(P\) steps), we stop early and select the candidate with the highest AR score seen so far. We set \(P=2\) as a default (configurable).  
4. If the loop reaches \(N_{\max}\) without meeting the improvement criterion, we select the candidate with the highest AR score as the output; \(N_{\text{eff}} = N_{\max}\).  

*The margin \(\delta\) and patience \(P\) are **calibrated** on a small development split (5 % of the evaluation prompts) to achieve a target acceptance rate of roughly 30 % after the first candidate (i.e., we expect the algorithm to stop early on easy prompts and to use more candidates on hard prompts). The same \(\delta, P\) are then fixed for all test prompts.*  

**Why this fixes the earlier logical flaw:**  
- The threshold \(\tau_p\) is initialized to the score of the **first** candidate, not \(-\infty\).  
- Acceptance requires a **strict improvement** over the current best by at least \(\delta\). Thus the first candidate is **not** automatically accepted unless it already meets the improvement criterion (which it cannot, because there is no prior best). Instead, we treat the first candidate as establishing a baseline; we only stop when we find a *better* sample.  
- The threshold is updated to the best‑so‑far score on each rejection, guaranteeing a monotonic non‑decreasing barrier that reflects the highest quality observed.  
- The patience mechanism prevents endless looping when improvements become marginal, yielding a well‑defined effective candidate count \(N_{\text{eff}}\) that varies across prompts.

### 5. Pareto Front Construction  
* For each model family (GPT‑2‑based vs. LLaMA‑based) and each \(S_0\) value, collect the pair \((\text{total FLOPs per token}, \text{UQM})\) across all prompts (total FLOPs = \(\sum_{i=1}^{N_{\text{eff}}} F_{\text{cand}}(S_0)\) + scoring overhead for each generated candidate).  
* Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
* Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
* Additionally, generate **slice‑wise frontiers**: (i) varying \(\delta\) at fixed \(S_0,N_{\max}\) to see the effect of the improvement margin; (ii) varying \(S_0\) at fixed \(\delta,N_{\max}\) to isolate the denoising‑step contribution; (iii) varying the effective candidate count (derived from \(N_{\text{eff}}\)) at fixed \(\delta,S_0\) to isolate the candidate‑selection contribution.

### 6. Trade‑off Analysis  
* **Marginal quality gain per denoising step:** For each fixed observed \(N_{\text{eff}}\) (averaged over prompts), regress UQM against \(S_0\) and report the slope \(\Delta\text{UQM}/\Delta S\).  
* **Marginal quality gain per additional candidate:** For each fixed \(S_0\), regress UQM against \(\log_2(N_{\text{eff}})\) and report \(\Delta\text{UQM}/\Delta\log_2 N\).  
* **Marginal quality gain per unit improvement margin:** Regress UQM against \(\delta\) (holding \(S_0,N_{\max}\) constant) to obtain \(\Delta\text{UQM}/\Delta\delta\).  
* **Effect of ACE‑v2 vs. static reranking:**  
  - Run a baseline where we generate a fixed number of candidates \(N \in \{1,2,4,8\}\) with step budget \(S_0\), score them with the AR model, and pick the highest‑scoring candidate (static reranking).  
  - Compare the ACE‑v2 frontier to the static‑reranking frontier at matched compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts).  
  - Declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  
* **Ablation of effective candidate distribution:** Report the histogram of \(N_{\text{eff}}\) across prompts for each \(S_0\) and each model family to demonstrate that ACE‑v2 actually varies the number of candidates used.

### 7. Generalizability Checks  
* **Hold‑out evaluation:** Repeat the entire pipeline on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
* **Unseen diffusion model:** Test an additional, unseen DLM (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
* **Alternative lightweight scorers:** Replace the AR base model with (a) a distilled 60M‑parameter Transformer trained on the same corpus, or (b) a frozen BERT‑style MLM, and repeat the ACE‑v2 experiment to assess scorer‑agnosticism.  
* **Cross‑task validation:** Apply ACE‑v2 to a non‑code task (e.g., summarization on CNN/DailyMail using the same DLMs) to see whether the early‑acceptance gate transfers when the AR scorer is replaced by a task‑specific lightweight reward model (trained on a small validation set).  
* **Threshold calibration sensitivity:** Vary the development‑set size used to set \(\delta\) and \(P\) (e.g., 2 %, 5 %, 10 %) and report the resulting AUPC to show robustness.  
* **FLOP model validation:** On a subset of models, compare the analytic FLOP estimate to measured GPU cycles via Nsight Systems; report the ratio and use it to correct any systematic bias in the efficiency metric.

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  
* **Weeks 1‑2:** environment setup, checkpoint download, tokenizer compatibility verification, implement FLOP‑analysis harness (using `fvcore`), basic diffusion sampling loop.  
* **Weeks 3‑4:** implement ACE‑v2 acceptance logic (margin \(\delta\), patience \(P\)), collect UQM and FLOP data for all \(S_0\) values and both diffusion families on the main evaluation suites.  
* **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions, bootstrap significance testing vs. static reranking.  
* **Week 6:** ablation of \(N_{\text{eff}}\) distribution, sensitivity to \(\delta\) and \(P\).  
* **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, cross‑task test).  
* **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, ACE‑v2 threshold‑sensitivity curves, \(N_{\text{eff}}\) histograms), prepare reproducibility package (Dockerfile, scripts, seeded RNG, calibration details).  

---  

**Substantive Difference from Prior Proposals:**  
The earlier works either (1) fixed a grid of \((N,S)\) pairs and optionally performed a two‑stage refinement, or (2) injected AR scores into the denoising dynamics via gradient‑based guidance. ACE‑v2 instead **treats the AR scorer as a stopping gate** that decides, after each fully‑generated candidate, whether to continue sampling based on a *strict improvement* criterion. This yields a **prompt‑adaptive effective candidate count** without altering the diffusion trajectory or requiring a predetermined number of samples. By directly measuring the resulting quality–compute curve, ACE‑v2 provides a clean, inference‑only test of the hypothesis that “better exploitation of candidate diversity” (via early acceptance) can close the accuracy headroom more efficiently than simply adding denoising steps, thereby addressing L6 and L9 in a novel way.