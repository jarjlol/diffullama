**Method: AR‑Guided Metropolis‑Hastings Diffusion Sampling (AR‑MHDS)**  

1. **Model and Checkpoint Preparation**  
   - Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
   - Verify that each diffusion checkpoint uses the same tokenizer as its AR base; if a mismatch exists, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change, still inference‑only).  
   - Load all models in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

2. **Unified Quality Metric (UQM)**  
   - For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
   - Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
   - UQM = (average of the five normalized scores). This yields a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

3. **Compute‑Normalized Efficiency Measurement**  
   - **Diffusion FLOPs per denoising step**: \(F_{\text{diff}} = \alpha \times |\theta_{\text{diff}}| \times L\) (α≈2 for multiply‑add, \(|\theta_{\text{diff}}|\) = diffusion parameter count, L=128).  
   - **AR‑scorer FLOPs per proposal**: a single forward pass of the frozen AR base model over the same sequence length, \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\). Empirically this is ≤ 1 % of \(F_{\text{diff}}\) and is added for every proposal.  
   - **Total FLOPs per proposal**: \(F_{\text{prop}} = S \times F_{\text{diff}} + F_{\text{AR}}\).  
   - **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including proposal generation, diffusion denoising, and AR scoring) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
   - Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

4. **AR‑Guided Metropolis‑Hastings Sampling Procedure**  
   - For each prompt, initialize a Markov chain with a random noise tensor \(\mathbf{z}_T^{(0)} \sim \mathcal{N}(0,\mathbf{I})\).  
   - For iteration \(i = 1,\dots,N_{\text{iter}}\):  
        a. **Proposal**: generate a candidate noise \(\mathbf{z}_T^{\text{prop}} = \mathbf{z}_T^{(i-1)} + \sigma \boldsymbol{\epsilon}\), where \(\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})\) and \(\sigma\) controls the proposal scale (fixed across experiments, e.g., \(\sigma=0.1\)).  
        b. **Denoising**: run the diffusion model’s denoising network for **S** steps on \(\mathbf{z}_T^{\text{prop}}\) to obtain a candidate clean sequence \(\mathbf{x}_0^{\text{prop}}\).  
        c. **Scoring**: compute the AR‑model log‑likelihood of the candidate, \(s^{\text{prop}} = \log p_{\text{AR}}(\mathbf{x}_0^{\text{prop}})\) (average token log‑likelihood; higher = better).  
        d. **Acceptance probability**:  
           \[
           a = \min\bigl(1,\; \exp(s^{\text{prop}} - s^{(i-1)})\bigr),
           \]  
           where \(s^{(i-1)}\) is the AR score of the current chain state.  
        e. **Update**: accept the proposal with probability \(a\); if accepted, set \(\mathbf{z}_T^{(i)} = \mathbf{z}_T^{\text{prop}}\) and \(s^{(i)} = s^{\text{prop}}\); otherwise keep the previous state (\(\mathbf{z}_T^{(i)} = \mathbf{z}_T^{(i-1)}\), \(s^{(i)} = s^{(i-1)}\)).  
        f. **Record**: store the current state’s sequence \(\mathbf{x}_0^{(i)}\) (the decoded version of \(\mathbf{z}_T^{(i)}\)) as a sample for later selection.  
   - After completing \(N_{\text{iter}}\) iterations, select the sample with the highest AR score as the model’s final output for the prompt.  
   - **Conditions explored**:  
        * Denoising‑step budget \(S \in \{8,16,32,64,128\}\) (controls diffusion compute per proposal).  
        * Number of Metropolis‑Hastings iterations \(N_{\text{iter}} \in \{1,2,4,8,16\}\) (controls how many proposals are generated; note that \(N_{\text{iter}}=1\) reduces to a single diffusion sample with AR scoring only for selection).  
        * Proposal scale \(\sigma\) fixed to a small value (e.g., 0.1) to ensure local exploration; a separate ablation can test \(\sigma\in\{0.05,0.1,0.2\}\).  

5. **Pareto Front Construction**  
   - For each model family (GPT‑2‑based vs. LLaMA‑based) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((S, N_{\text{iter}})\) condition.  
   - Derive the **empirical Pareto frontier** by retaining points where no other point has both higher or equal quality and lower or equal compute.  
   - Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
   - Additionally, generate **slice‑wise frontiers**: (i) varying \(S\) at fixed \(N_{\text{iter}}\) to isolate the denoising‑step contribution; (ii) varying \(N_{\text{iter}}\) at fixed \(S\) to isolate the proposal‑count contribution.  

6. **Trade‑off Analysis**  
   - **Marginal quality gain per additional denoising step**: fit a piecewise‑linear regression of UQM versus \(S\) (holding \(N_{\text{iter}}\) constant) and report the slope \(\Delta\text{UQM}/\Delta S\).  
   - **Marginal quality gain per additional proposal (iteration)**: fit a similar regression of UQM versus \(\log_2(N_{\text{iter}})\) (holding \(S\) constant).  
   - **Effect of AR‑guided MH vs. independent sampling + reranking**: repeat the entire pipeline with a baseline that draws \(N_{\text{ind}}\) independent diffusion samples (each with \(S\) denoising steps), scores them with the AR model, and keeps the highest‑scoring sample. Compare the two frontiers at fixed compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model) using paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the difference in UQM.  
   - **Statistical significance**: declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  

7. **Generalizability Checks**  
   - Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
   - Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
   - Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM (using its masked‑language‑model likelihood as a proxy score) and repeat the AR‑MHDS experiment to assess scorer‑agnosticism.  
   - Multi‑chain extension: run \(C\) independent MH chains in parallel (e.g., \(C=4\)) and merge their samples before selection; evaluate whether parallel chains improve the Pareto frontier without increasing per‑chain compute.  

8. **Resource‑aware Implementation Plan (10‑week timeline)**  
   - **Weeks 1‑2**: environment setup, checkpoint download, tokenizer alignment, implement diffusion denoising loop, AR‑scorer forward pass, and MH proposal/acceptance logic.  
   - **Weeks 3‑4**: build the sampling pipeline that supports variable \(S\) and \(N_{\text{iter}}\); collect baseline UQM and FLOP data for all conditions.  
   - **Week 5**: implement Pareto front extraction, AUPC calculation, and marginal‑gain regressions.  
   - **Week 6**: bootstrap significance testing and baseline (independent sampling + reranking) comparison.  
   - **Weeks 7‑8**: generalization experiments (held‑out prompts, extra DLM, alternative scorers, multi‑chain ablation).  
   - **Weeks 9‑10**: write‑up, visualizations (Pareto plots, marginal‑gain bar charts, MH‑trace diagnostics), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Rationale**  

The method directly tackles the two open issues highlighted in the target paper.  

*Accuracy headroom (L6):* By using the AR model’s sequence‑level likelihood as a Metropolis‑Hastings acceptance criterion, the sampler preferentially explores noise trajectories that lead to higher‑AR‑likelihood outputs. This exploits the model’s internal candidate diversity **without** requiring many independent full‑sequence samples; the Markov chain can climb toward high‑likelihood regions even when each individual proposal is noisy, effectively closing the accuracy headroom.  

*Compute‑normalization gap (L9):* Every proposal’s compute is explicitly accounted for: \(S\) denoising steps of the diffusion model plus one forward pass of the lightweight AR scorer. Consequently, each \((S, N_{\text{iter}})\) condition is evaluated on a strict compute‑normalized basis, enabling a fair comparison between investing compute in more denoising steps versus more proposals (MH iterations). The wall‑clock validation ensures that FLOP estimates reflect actual latency on the RTX 6000 Pro Blackwell.  

Unlike the previously suggested approach—which relies on static candidate generation followed by a two‑stage adaptive re‑allocation—AR‑MHDS treats denoising and selection as a **single stochastic process** where the AR scorer guides a random walk in noise space. This offers a distinct angle: rather than generating many independent candidates and then reranking, we reuse compute to iteratively improve a single trajectory guided by the AR scorer, potentially shifting the Pareto frontier upward more efficiently than merely increasing \(S\) or \(N\). The method remains fully inference‑only, uses only publicly released checkpoints, fits the ten‑week schedule on a single GPU, and yields results directly comparable to the baseline reranking strategy via shared quality and efficiency metrics.