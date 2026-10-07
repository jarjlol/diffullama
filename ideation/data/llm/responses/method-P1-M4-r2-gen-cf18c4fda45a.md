**Method: AR‑Guided Adaptive Denoising with Early Stopping and Candidate Rescheduling (AGADES)**  

**Rationale**  
The target paper identifies two open issues: (L6) a sizable “accuracy headroom’’ that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, making it unclear whether DLMs truly beat AR models when FLOPs are accounted for. Existing inference‑time tricks (Jacobi Forcing, TESS 2 reward guidance, static candidate reranking) either (i) add a uniform compute overhead (guidance) or (ii) treat all candidates identically (same denoising‑step budget per sample).  

AGADES attacks the headroom from a **different angle**: it uses the lightweight AR scorer not only for final ranking but also as an **online signal of per‑candidate convergence**. During generation we monitor how much each candidate’s AR‑score improves after each denoising step. When the improvement falls below a small threshold we deem that candidate “converged’’ and stop spending further steps on it. The compute that would have been wasted on already‑saturated candidates is **re‑allocated** either (a) to increase the number of candidates N (exploring more diverse modes) or (b) to give extra denoising steps to the most promising candidates.  

Because the decision to stop or continue is made **per candidate and per step**, the total denoising‑step budget is spent where it yields the highest marginal quality gain. This yields a Pareto frontier that can dominate both (i) uniform step allocation with static reranking and (ii) pure candidate‑exploration with fixed steps, while still respecting the strict inference‑only constraint (no weight updates, only forward passes of the frozen AR model).  

The method is fully compatible with the released DLM checkpoints, requires only the AR base model for scoring, and can be implemented within the ten‑week timeline on a single RTX 6000 Pro Blackwell by parallelizing candidate processing across the three GPU‑enabled team members.  

---  

### 1. Model and Checkpoint Preparation  
- Assemble the five released DLMs: **DiffuGPT‑S/M**, **DiffuLLaMA‑6.74B**, **Dream‑7B**, **LLaDA‑8B**, **DiffuCoder‑7B**.  
- For each DLM obtain its corresponding AR base model (GPT‑2‑small/medium for the GPT‑2‑based DLMs; LLaMA‑7B for the LLaMA‑based DLMs).  
- Verify tokenizer compatibility; if the diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
- Load **one diffusion model and its AR scorer** in FP16 on the RTX 6000 Pro Blackwell at a time; this fits comfortably within the 96 GB memory budget.  

### 2. Unified Quality Metric (UQM)  
- For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
- Apply **min‑max normalization within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
- UQM = average of the five normalized scores → a dimensionless quality score in [0, 1] where each benchmark contributes equally.  
- In addition to the scalar UQM, report per‑benchmark normalized scores to expose any domain‑specific effects.  

### 3. Compute‑Normalized Efficiency Measurement  
- **Diffusion FLOPs per denoising step (full sequence)**:  
  \[
  F_{\text{diff}} = \alpha \times |\theta| \times L
  \]  
  where \(\alpha\approx2\) (multiply‑add), \(|\theta|\) = diffusion parameter count, \(L=128\) (fixed sequence length).  
- **AR scorer FLOPs per evaluation**: a single forward pass of the frozen AR base model over the sequence (≈ 1 % of \(F_{\text{diff}}\)).  
- **Total FLOPs for a condition**:  
  \[
  \text{FLOPs}_{\text{total}} = 
  \sum_{i=1}^{N}\bigl(s_i \times F_{\text{diff}}\bigr) \;+\;
  \bigl(\text{number of AR evaluations}\bigr) \times F_{\text{AR}}
  \]  
  where \(s_i\) is the actual number of denoising steps executed for candidate \(i\).  
- **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including all denoising passes, AR scoring, and bookkeeping) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
- Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

### 4. AR‑Guided Adaptive Denoising Procedure (AGADES)  

| Symbol | Meaning |
|--------|---------|
| \(N\) | initial number of candidates (∈ {1, 2, 4, 8}) |
| \(S_{\max}\) | maximum denoising steps allowed per candidate (∈ {8, 16, 32, 64, 128}) |
| \(\epsilon\) | improvement threshold for early stopping (default = 1e‑4 in AR NLL) |
| \(B\) | total compute budget in FLOPs (varied to trace the Pareto front) |
| \(s_i\) | steps already executed for candidate \(i\) |
| \(\Delta_i\) | AR‑NLL improvement of candidate \(i\) after its most recent step |

**Algorithm (per prompt):**  

1. **Initialization**  
   - Sample \(N\) independent noise tensors \(\{\mathbf{z}^{(i)}_{T}\}_{i=1}^{N}\).  
   - Set \(s_i \gets 0\) for all candidates.  
   - Compute the initial AR‑NLL of each candidate’s decoded sequence (obtained by taking the argmax of the uniform distribution, i.e., a random token sequence) and store as \(\text{NLL}_i^{(0)}\).  

2. **Iterative step allocation**  
   While the cumulative FLOPs used \(\displaystyle\sum_i s_i F_{\text{diff}} + (\text{AR evals})F_{\text{AR}} < B\):  
   a. For each candidate that has not yet reached \(S_{\max}\), estimate the **marginal gain** if we were to spend one more step:  
      \[
      g_i = \frac{\text{NLL}_i^{(s_i)} - \text{NLL}_i^{(s_i-1)}}{1}\quad\text{(if }s_i>0\text{)};\quad
      g_i = 0\;\text{if }s_i=0
      \]  
      (i.e., the observed NLL improvement from the last step; for the first step we use a small optimistic prior).  
   b. Select the candidate \(i^\*\) with the largest \(g_i\) (break ties randomly).  
   c. Execute **one denoising step** on candidate \(i^\*\) (standard diffusion reverse‑process update).  
   d. Increment \(s_{i^\*} \gets s_{i^\*}+1\).  
   e. Decode the current latent of candidate \(i^\*\) (e.g., by taking the expected token distribution) and compute its AR‑NLL; store as \(\text{NLL}_{i^\*}^{(s_{i^\*})}\).  
   f. If \(s_{i^\*}=S_{\max}\) **or** \(\bigl|\text{NLL}_{i^\*}^{(s_{i^\*})}-\text{NLL}_{i^\*}^{(s_{i^\*}-1)}\bigr| < \epsilon\), mark candidate \(i^\*\) as **converged** and remove it from further consideration.  

3. **Final selection**  
   - After the compute budget is exhausted, each candidate \(i\) has a final sequence \(\mathbf{x}^{(i)}\) and an AR‑NLL \(\text{NLL}_i^{(s_i)}\).  
   - Choose the candidate with the **lowest AR‑NLL** as the model’s output for that prompt.  

**Explored conditions**  
- \(N \in \{1,2,4,8\}\) (initial pool size)  
- \(S_{\max} \in \{8,16,32,64,128\}\) (per‑candidate ceiling)  
- Total compute budget \(B\) varied to produce points spanning from ~0.5× to 4× the FLOPs of a uniform baseline (e.g., \(N=1, S=S_{\max}\)).  
- The threshold \(\epsilon\) is fixed; a sensitivity check with \(\epsilon\in\{10^{-5},10^{-4},10^{-3}\}\) is performed in the generalization phase.  

### 5. Pareto Front Construction  
- For each model family (GPT‑2‑based vs. LLaMA‑based) plot **UQM (y‑axis)** against **total FLOPs per token (x‑axis)** for every \((N, S_{\max}, B)\) condition.  
- Derive the **empirical Pareto frontier** by retaining points where no other point has both **higher or equal quality** and **lower or equal compute**.  
- Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
- Additionally, generate **slice‑wise frontiers**:  
  * Vary \(B\) at fixed \(N, S_{\max}\) to see the effect of total compute.  
  * Vary \(N\) at fixed \(B, S_{\max}\) to isolate the candidate‑exploration contribution.  
  * Vary \(S_{\max}\) at fixed \(B, N\) to isolate the per‑candidate step ceiling.  

### 6. Trade‑off Analysis  
- **Marginal quality gain per denoising step (adaptive)**: fit a piecewise‑linear regression of UQM versus cumulative steps \(\sum_i s_i\) (holding \(N\) and \(S_{\max}\) constant) and report \(\Delta\text{UQM}/\Delta\bigl(\sum_i s_i\bigr)\).  
- **Marginal quality gain per additional candidate**: regress UQM against \(\log_2(N)\) (holding total compute budget \(B\) and \(S_{\max}\) constant).  
- **Effect of adaptive allocation vs. uniform baselines**: for a set of fixed compute budgets (e.g., 0.5×, 1×, 2× the FLOPs of the base AR model), compare the UQM achieved by:  
  1. **Uniform denoising + static reranking** (each of \(N\) candidates receives exactly \(S = B/(N\cdot F_{\text{diff}})\) steps, then AR‑rerank).  
  2. **Pure candidate exploration** (fix \(S=S_{\max}\), vary \(N\) such that total FLOPs ≈ \(B\)).  
  3. **AGADES** (adaptive step allocation as described).  
  Use **paired bootstrap** (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences.  
- **Statistical significance**: declare a strategy superior if the 95 % bootstrap CI of the UQM difference does **not** contain zero.  
- **Ablation of the early‑stopping signal**: repeat the pipeline with a random stopping signal (i.e., stop candidates after a geometrically distributed number of steps independent of AR NLL) to confirm that the observed gains stem from the AR‑guided criterion rather than merely from step‑count variability.  

### 7. Generalizability Checks  
- **Hold‑out evaluation** on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
- **Unseen DLM**: test a 3B‑parameter diffusion model from the dLLM zoo (not used in the main analysis) to verify that trends extrapolate across architectures.  
- **Alternative lightweight scorers**: replace the AR base model with (i) a distilled 60M‑parameter Transformer, (ii) a frozen BERT‑style MLM, and (iii) a small reward model fine‑tuned on human preferences; run AGADES with each scorer to assess scorer‑agnosticism.  
- **Varying sequence length**: repeat a subset of experiments with \(L=256\) tokens to test scalability of the adaptive step allocation.  
- **Per‑benchmark reporting**: alongside UQM, present per‑benchmark normalized scores to confirm that gains are not confined to a single task type.  

### 8. Resource‑aware Implementation Plan (10‑week timeline)  

| Week | Activities |
|------|------------|
| 1‑2 | Environment setup, checkpoint download, tokenizer alignment, implement the diffusion denoising step with step‑counter and AR‑scoring hook; verify that AR NLL can be obtained after each step without breaking the diffusion process. |
| 3‑4 | Build the adaptive step‑allocation loop (priority queue based on observed NLL improvement); collect baseline UQM and FLOP data for all \((N, S_{\max}, B)\) combos. |
| 5 | Implement Pareto front extraction, AUPC calculation, and marginal‑gain regressions. |
| 6 | Perform bootstrap significance testing and baseline comparisons (uniform + static reranking, pure candidate exploration, random‑stopping ablation). |
| 7‑8 | Generalization experiments (held‑out prompts, extra DLM, alternative scorers, longer sequences). |
| 9‑10 | Write‑up, visualizations (Pareto plots, marginal‑gain bar charts, early‑stopping sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG). |

---  

**Why AGADES is substantively different**  
- Unlike the static candidate‑generation‑then‑rerank approach (Method 1), AGADES **varies the denoising‑step budget per candidate online**, guided by the AR scorer’s intermediate feedback.  
- Unlike gradient‑based guidance (Method 2), AGADES **does not modify the diffusion latent with a gradient step**; it merely decides *when* to stop spending steps on a given candidate based on a cheap scalar signal (AR NLL improvement).  
- Unlike the token‑wise masking method (Method 3/ATAR), AGADES operates at the **candidate level**, requiring only a forward pass of the AR model after each denoising step, and it does **not rely on attention masks or partial denoising**; it works with any released diffusion checkpoint that supports the standard reverse process.  

By explicitly measuring the compute spent on each candidate and reallocating unused steps where the AR scorer predicts the highest marginal return, AGADES provides a **principled, inference‑only mechanism** for improving the Pareto frontier of quality versus compute in released diffusion language models.