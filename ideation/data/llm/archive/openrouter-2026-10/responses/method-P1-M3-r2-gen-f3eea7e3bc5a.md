**Method: Progressive Candidate Pruning with an Autoregressive Scorer (PCP‑AR)**  
**Rationale:**  
The target paper identifies two open issues: (L6) a sizable “accuracy headroom’’ that could be closed by either more denoising steps or better exploitation of the model’s candidate diversity, and (L9) efficiency claims that are not compute‑normalized, making it unclear whether DLMs truly beat AR models when cost is accounted for. Existing inference‑time work (Jacobi Forcing, TESS 2 reward guidance) shows that compute can be reallocated between denoising and selection, but it remains unknown whether the headroom is best addressed by (i) raising the denoising‑step budget *S* or (ii) generating and reranking more samples *N* using a tiny AR scorer.  

PCP‑AR offers a **different inference‑time mechanism**: instead of fixing *N* ahead of time or injecting AR scores into the denoising dynamics, we start with a **large pool of parallel diffusion trajectories**, score them **after each denoising step** with the frozen AR base model, and **discard the lowest‑scoring candidates**. The surviving candidates continue to be denoised for the remaining steps. This **progressive pruning** reallocates compute from poorly‑scoring trajectories to the denoising of promising ones, thereby exploiting candidate diversity *without* generating a fixed number of full‑length samples. By varying the initial pool size and the pruning aggressiveness we can trace a quality‑compute curve that directly compares the returns of investing compute in more denoising steps versus in more (but pruned) candidates. The AR scorer acts as a lightweight, inference‑only reranker that guides the allocation of the denoising budget, allowing us to test whether such guidance shifts the Pareto frontier upward more effectively than simply increasing *S*.

Because PCP‑AR only requires frozen checkpoints, no training or adaptation, and can be implemented with a single RTX 6000 Pro Blackwell (FP16, sequence length = 128), it satisfies all resource constraints and fits comfortably within a ten‑week schedule.

---

### 1. Model and Checkpoint Preparation  
* Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
* Verify tokenizer compatibility: if a diffusion checkpoint’s tokenizer differs from its AR base, load the AR base **with the diffusion model’s tokenizer** (no weight change, inference‑only).  
* Load models in FP16; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.

### 2. Unified Quality Metric (UQM)  
* For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
* Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
* UQM = average of the five normalized scores → dimensionless quality in \([0,1]\) with equal benchmark contribution.  
* **Sanity check:** On a held‑out validation set (5 % of prompts) compute the Pearson correlation between AR NLL (averaged over tokens) and UQM. If \(|r|<0.3\) we will flag the scorer as weakly predictive and fall back to a hybrid score (AR NLL + length penalty, see §7).

### 3. Compute‑Normalized Efficiency Measurement  
* **Diffusion FLOPs per denoising step (per candidate):**  
  \[
  F_{\text{diff}} = \sum_{l=1}^{L_{\text{layers}}} \bigl(2 \times C_{\text{in}}^{(l)} \times C_{\text{out}}^{(l)} \times K^{(l)} \times L_{\text{seq}}\bigr)
  \]  
  where \(C_{\text{in/out}}\) are input/output channel dimensions, \(K^{(l)}\) is the effective kernel size (attention: \(K = L_{\text{seq}}\); feed‑forward: \(K = 1\)), and \(L_{\text{seq}}=128\). This is obtained automatically via `fvcore.nn.FlopCountAnalysis`.  
* **AR scorer overhead:** a single forward pass of the frozen AR base model over the same sequence length; empirically ≤ 1 % of diffusion FLOPs and added **only when scoring a candidate** (see §4).  
* **Total FLOPs for a candidate that survives *s* denoising steps:**  
  \[
  F_{\text{cand}}(s) = s \times F_{\text{diff}} + F_{\text{AR}}.
  \]  
* **Wall‑clock validation:** warm‑up 100 tokens, then measure average latency per generated token (including candidate generation, scoring, and pruning) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
* Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.

### 4. Progressive Candidate Pruning with AR Scorer (PCP‑AR)  
For each prompt *p* we define three hyper‑parameters:  

* **Initial pool size** \(N_0 \in \{8,16,32,64\}\) – number of independent diffusion trajectories started from random noise.  
* **Pruning ratio** \(\rho \in \{0.25,0.5,0.75\}\) – after each denoising step we keep the top \((1-\rho)\) fraction of candidates (rounded up to at least 1).  
* **Total denoising‑ steps** \(S \in \{8,16,32,64,128\}\) – the full schedule length; after the final step we retain the single highest‑scoring candidate as the model output.  

Algorithm (per prompt):  

1. **Initialize** \(N_0\) candidates \(\{\mathbf{z}_T^{(i)}\}_{i=1}^{N_0}\) with independent Gaussian noise.  
2. **For** denoising step \(t = T, T-1, \dots, 1\):  
   a. Run the diffusion denoising network on **all surviving candidates** to obtain predicted clean token logits \(\mathbf{p}_\theta(\mathbf{z}_t^{(i)})\).  
   b. Convert logits to a probability distribution and obtain the **expected embedding** \(\mathbf{e}_t^{(i)}\) (or sample a tentative token sequence).  
   c. **Score** each candidate with the AR base model: compute the average negative log‑likelihood (NLL) of the expected embedding under the AR model; define the AR score as \(-\text{NLL}\) (higher = better).  
   d. **Prune:** sort candidates by AR score, discard the lowest \(\rho\) fraction, keep the rest for the next step. If the number of survivors falls below 1, keep the single best candidate.  
3. **After** the final step (\(t=0\)), decode the remaining candidates to token sequences, compute their AR scores (full‑sequence NLL), and select the highest‑scoring candidate as the output for the prompt.  

*The total number of diffusion steps actually executed is*  
\[
\text{Steps}_{\text{executed}} = \sum_{t=1}^{S} N_{\text{surv}}(t)
\]  
*where \(N_{\text{surv}}(t)\) is the number of candidates alive at step \(t\). The overall FLOP cost is*  
\[
\text{FLOPs}_{\text{total}} = \sum_{t=1}^{S} N_{\text{surv}}(t) \times F_{\text{diff}} \;+\; \bigl(\sum_{t=1}^{S} N_{\text{surv}}(t)\bigr) \times F_{\text{AR}} \;+\; F_{\text{AR}}^{\text{final}}
\]  
*where the last term accounts for the final full‑sequence AR scoring of the surviving candidates (negligible compared to the diffusion term).*

### 5. Pareto Front Construction  
* For each model family (GPT‑2‑based vs. LLaMA‑based) and each combination \((N_0,\rho,S)\) we compute the pair **(total FLOPs per token, UQM)** averaged over all prompts.  
* Plot quality (UQM) versus compute (FLOPs) and derive the **empirical Pareto frontier** (points where no other point has both higher/equal quality and lower/equal compute).  
* Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
* Additionally, generate **slice‑wise frontiers**: (i) varying \(\rho\) at fixed \(N_0,S\) to see the effect of pruning aggressiveness; (ii) varying \(S\) at fixed \(N_0,\rho\) to isolate the denoising‑step contribution; (iii) varying the effective candidate‑budget \(\displaystyle B = \sum_{t} N_{\text{surv}}(t)\) at fixed \(\rho,S\) to isolate the candidate‑selection contribution.

### 6. Trade‑off Analysis  
* **Marginal quality gain per denoising step:** For each fixed observed effective candidate‑budget \(B\) (averaged over prompts), regress UQM against \(S\) and report the slope \(\Delta\text{UQM}/\Delta S\).  
* **Marginal quality gain per additional candidate‑budget:** For each fixed \(S\), regress UQM against \(\log_2(B)\) and report \(\Delta\text{UQM}/\Delta\log_2 B\).  
* **Marginal quality gain per unit pruning ratio:** Regress UQM against \(\rho\) (holding \(N_0,S\) constant) to obtain \(\Delta\text{UQM}/\Delta\rho\).  
* **Effect of PCP‑AR vs. static reranking:**  
  - Run a baseline where we generate a fixed number of full‑length candidates \(N \in \{1,2,4,8\}\) with step budget \(S\), score them with the AR model, and pick the highest‑scoring candidate (static reranking).  
  - Compare the PCP‑AR frontier to the static‑reranking frontier at matched compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts).  
  - Declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  
* **Ablation of effective candidate distribution:** Report the histogram of the effective candidate‑budget \(B\) across prompts for each \((N_0,\rho,S)\) to demonstrate that PCP‑AR actually varies the amount of computation devoted to candidate exploration.

### 7. Generalizability Checks  
* **Hold‑out evaluation:** Repeat the entire pipeline on HumanEval‑plus and MBPP to ensure findings are not benchmark‑specific.  
* **Unseen diffusion model:** Test an additional, unseen DLM (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
* **Alternative lightweight scorers:** Replace the AR base model with (a) a distilled 60M‑parameter Transformer trained on the same corpus, or (b) a frozen BERT‑style MLM, and repeat the PCP‑AR experiment to assess scorer‑agnosticism.  
* **Threshold‑free variant:** As a control, run PCP‑AR with the AR scorer disabled (i.e., random pruning) to quantify the contribution of the AR‑guided pruning versus undirected candidate reduction.  
* **FLOP model validation:** On a subset of models, compare the analytic FLOP estimate to measured GPU cycles via Nsight Systems; report the ratio and use it to correct any systematic bias in the efficiency metric.  
* **Calibration sensitivity:** Vary the development‑set used to set any heuristic (e.g., if we decide to adapt \(\rho\) per model family on a 5 % dev split) and report the resulting AUPC to show robustness.

### 8. Resource‑Aware Implementation Plan (10‑Week Timeline)  
* **Weeks 1‑2:** environment setup, checkpoint download, tokenizer verification, implement FLOP‑analysis harness (`fvcore`), basic diffusion sampling loop.  
* **Weeks 3‑4:** implement PCP‑AR pruning logic (dynamic candidate survival, AR scoring per step), collect UQM and FLOP data for all \((N_0,\rho,S)\) combinations on the main evaluation suites for both diffusion families.  
* **Week 5:** Pareto front extraction, AUPC calculation, marginal‑gain regressions, bootstrap significance testing vs. static reranking.  
* **Week 6:** ablation of effective candidate distribution, sensitivity to \(\rho\) and \(N_0\).  
* **Weeks 7‑8:** generalization experiments (held‑out prompts, extra DLM, alternative scorers, random‑pruning control).  
* **Weeks 9‑10:** write‑up, visualizations (Pareto plots, marginal‑gain bar charts, PCP‑AR pruning‑ratio sensitivity curves, candidate‑budget histograms), prepare reproducibility package (Dockerfile, scripts, seeded RNG, calibration details).

---

#### Substantive Difference from Prior Proposals  
* **ACE‑v2** decides after a *complete* candidate whether to stop generating more full‑length samples.  
* **Guidance‑Driven Diffusion** injects AR scores into the denoising dynamics at every step via gradient updates.  
* **PCP‑AR** instead **starts with many parallel trajectories**, scores them **after each denoising step**, and **prunes the weakest survivors**, thereby continuously reallocating the denoising budget from low‑quality to high‑quality candidates. This mechanism explores the trade‑off between denoising steps and candidate diversity in a way that neither prior proposal does, while remaining strictly inference‑only and using only released checkpoints.