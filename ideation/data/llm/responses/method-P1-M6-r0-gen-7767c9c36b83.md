**Method:**  
**Successive‑Halving Allocation for Diffusion Language Models (SHADLM)** – a multi‑round, tournament‑style inference strategy that dynamically shifts compute from generating many low‑quality candidates to refining a small set of promising ones, guided solely by a frozen autoregressive (AR) scorer.  

**Rationale:**  
The quality‑compute trade‑off of released DLMs can be viewed as a resource‑allocation problem: we have a fixed inference‑time FLOP budget and must decide how to split it between (i) exploring the solution space (more candidates, few denoising steps) and (ii) exploiting promising solutions (few candidates, many denoising steps). Existing work either (a) fixes a static \((N,S)\) pair, (b) uses a single‑stage adaptive re‑allocation (e.g., generate cheap candidates then boost the top‑k), or (c) steers the diffusion process with classifier‑guidance. None of these explicitly model the allocation as a sequential elimination tournament, which is known to be near‑optimal for hyper‑parameter optimization under a fixed budget (e.g., Hyperband).  

SHADLM treats each candidate as an “arm” whose unknown quality is estimated by the AR scorer’s negative log‑likelihood (NLL). Starting with a large pool of very cheap candidates, we iteratively discard the lowest‑scoring half and reinvest the saved compute into additional denoising steps for the survivors. This mirrors the intuition behind the accuracy headroom (L6): if the model’s candidate diversity contains high‑quality samples, we can uncover them by aggressive exploration; if the headroom is better closed by refining a few good samples, the halving process automatically shifts compute toward refinement as the pool shrinks. By keeping the AR scorer frozen and using it only for ranking, SHADLM respects the inference‑only constraint and avoids any training or adaptation.  

Because the halving schedule is deterministic given a total FLOP budget, we can sweep a range of budgets (e.g., 0.5×, 1×, 2×, 4× the FLOPs of the base AR model) and record the resulting quality (UQM) and compute for each model family. Plotting quality versus compute yields an empirical Pareto front that directly answers whether lightweight reranking—here instantiated as successive‑halving‑guided refinement—improves the trade‑off over simply increasing denoising steps or candidate count. Statistical significance is assessed via bootstrap resampling of prompts, and generalizability is checked on held‑out benchmarks and alternative DLMs.  

Overall, SHADLM provides a principled, compute‑normalized, and easily implementable alternative to static \((N,S)\) grids and single‑stage adaptive schemes, offering a clear answer to the research question while fitting the ten‑week timeline, single‑GPU limit, and inference‑only requirement.  

---  

### Detailed Procedure  

1. **Model and Checkpoint Preparation**  
   - Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
   - Ensure tokenizer compatibility; if a mismatch exists, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change).  
   - Load one diffusion model and its AR scorer in FP16 on the RTX 6000 Pro Blackwell; keep only this pair resident at a time.  

2. **Unified Quality Metric (UQM)**  
   - For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
   - Apply min‑max normalization **within each benchmark** across all conditions so the worst score → 0, best → 1.  
   - UQM = average of the five normalized scores (range [0, 1]).  

3. **Compute‑Normalized Efficiency Measurement**  
   - **Diffusion FLOPs per denoising step**: \(F_{\text{diff}} = \alpha \times |\theta| \times L\) with \(\alpha≈2\) (multiply‑add), \(|\theta|\) = diffusion parameter count, \(L=128\).  
   - **AR scorer FLOPs per forward pass**: \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\) (≤ 1 % of \(F_{\text{diff}}\)).  
   - **Total FLOPs for a condition**:  
     \[
     \text{FLOPs}_{\text{total}} = \sum_{r=1}^{R} N_r \times S_r \times F_{\text{diff}} \;+\; \bigl(\sum_{r=1}^{R} N_r\bigr) \times F_{\text{AR}},
     \]
     where round \(r\) uses \(N_r\) candidates each run for \(S_r\) denoising steps; the final term accounts for scoring every candidate once per round (the AR forward pass is negligible but added for completeness).  
   - **Wall‑clock validation**: warm‑up 100 tokens, then measure average latency per generated token (including all rounds) via CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
   - Report efficiency as **quality per FLOP** (or quality per millisecond).  

4. **Successive‑Halving Allocation (SHADLM) Algorithm**  
   - **Input**: total FLOP budget per token \(B\), minimum steps \(S_{\min}\) (e.g., 8), maximum steps \(S_{\max}\) (e.g., 128), halving factor \(\eta = 2\) (default).  
   - **Step 0 – Initial pool**:  
     - Choose an initial number of candidates \(N_0 = \left\lfloor \frac{B}{S_{\min} \times F_{\text{diff}} + F_{\text{AR}}} \right\rfloor\).  
     - Generate \(N_0\) independent diffusion trajectories, each using \(S_{\min}\) denoising steps.  
     - Score each trajectory with the AR scorer (average NLL over the sequence).  
   - **Iterative halving** (repeat until budget exhausted):  
     1. **Rank** candidates by AR score (lower NLL = higher quality).  
     2. **Retain** the top \(\frac{1}{\eta}\) fraction; denote the number of survivors \(N_{r+1} = \left\lfloor \frac{N_r}{\eta} \right\rfloor\).  
     3. **Allocate** the compute saved from discarding the lower fraction to increase the denoising steps of survivors:  
        \[
        S_{r+1} = S_r + \left\lfloor \frac{(N_r - N_{r+1}) \times S_r}{N_{r+1}} \right\rfloor,
        \]
        (i.e., redistribute the steps of the discarded candidates evenly among the survivors).  
     4. **Generate** new trajectories for the survivors by continuing denoising from the current latent state for the additional \(S_{r+1} - S_r\) steps (no need to restart from noise).  
     5. **Score** the updated candidates again with the AR scorer.  
     6. **Update** the consumed FLOPs using the formula in §3; stop when the cumulative FLOPs ≥ \(B\) (or when \(S_{r+1}\) reaches \(S_{\max}\)).  
   - **Output**: the candidate with the best AR score after the final round is taken as the model’s output for the prompt.  

   - **Budget sweep**: To construct Pareto fronts, repeat the above for a set of total budgets \(B \in \{0.5\times, 1\times, 2\times, 4\times\}\) the FLOPs of a single AR forward pass over 128 tokens (i.e., the cost of the base AR model). For each budget we record the resulting UQM and actual FLOPs consumed.  

5. **Pareto Front Construction**  
   - For each model family (GPT‑2‑based vs. LLaMA‑based) plot UQM (y‑axis) against measured FLOPs per token (x‑axis) for every budget point obtained via SHADLM.  
   - Derive the empirical Pareto frontier by retaining points where no other point has both higher or equal quality and lower or equal compute.  
   - Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  

6. **Trade‑off Analysis**  
   - **Marginal quality gain per denoising step**: fit a piecewise‑linear regression of UQM versus cumulative denoising steps (averaged across survivors) while holding the halving schedule fixed; report \(\Delta\text{UQM}/\Delta S\).  
   - **Marginal quality gain per surviving candidate**: regress UQM against \(\log_2(N_r)\) (number of candidates retained after each round) while holding step allocation fixed.  
   - **Effect of halving vs. uniform increase**: for a fixed compute budget (e.g., 1×, 2×, 4× the base AR FLOPs), compare the UQM achieved by:  
     - (a) SHADLM (successive halving).  
     - (b) Uniform increase of denoising steps with \(N=1\) (baseline diffusion).  
     - (c) Uniform increase of candidate count with \(S=S_{\min}\) (plain reranking).  
     Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences; declare a strategy superior if the 95 % CI does not contain zero.  
   - **Baseline comparison**: repeat the entire pipeline with the static \((N,S)\) grid (the approach described in the research problem) to quantify how much SHADLM shifts the Pareto frontier relative to naïve candidate‑only reranking or step‑only increase.  

7. **Generalizability Checks**  
   - Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
   - Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
   - Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat SHADLM to assess scorer‑agnosticism.  

8. **Resource‑aware Implementation Plan (10‑week timeline)**  
   - **Weeks 1‑2**: environment setup, checkpoint download, tokenizer alignment, implement FLOP‑profiling harness and wall‑clock timing utilities.  
   - **Weeks 3‑4**: implement the successive‑halving loop (including latent‑state reuse for incremental denoising), collect UQM and FLOP data for all budget points.  
   - **Week 5**: Pareto front extraction, AUPC computation, marginal‑gain regressions.  
   - **Week 6**: bootstrap significance testing and baseline (static grid) comparison.  
   - **Weeks 7‑8**: generalization experiments (held‑out prompts, extra DLM, alternative scorers).  
   - **Weeks 9‑10**: write‑up, visualizations (Pareto plots, marginal‑gain bar charts, halving‑schedule sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---  

**Why SHADLM is substantively different**  
Unlike the previously proposed methods—which either (i) use a single adaptive re‑allocation of compute after an initial cheap generation, (ii) steer the diffusion process with classifier‑guidance, (iii) accept candidates early based on AR confidence, or (iv) selectively denoise uncertain tokens—SHADLM treats inference as a **sequential elimination tournament**. It starts with broad exploration (many cheap candidates) and, through statistically principled halving, automatically shifts compute toward exploitation (few candidates, many steps) only when the AR scorer indicates that the surviving candidates are worth refining. This mechanism directly investigates whether the accuracy headroom is better closed by **increasing denoising steps** or by **exploiting candidate diversity**, and it does so under a strict compute‑normalized, inference‑only regime, offering a fresh angle on the quality‑compute trade‑off of released diffusion language models.