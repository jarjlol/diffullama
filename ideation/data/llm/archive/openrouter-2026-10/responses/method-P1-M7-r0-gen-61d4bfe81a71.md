**Method: Tokenwise Adaptive Denoising with AR‑Guided Early Exit (TADA)**  

**Overview**  
TADA treats the denoising process of a diffusion language model (DLM) as a *per‑token* iterative refinement problem. Instead of applying a uniform denoising‑step budget `S` to every token position, we run the diffusion denoiser step‑by‑step and, after each step, query the frozen autoregressive (AR) base model for the *conditional likelihood* of the current token given the partially denoised context. Denoising for a token stops as soon as the AR‑model confidence exceeds a user‑defined threshold `τ` (or the improvement in likelihood falls below a small ε). Tokens that are easy to predict (high early likelihood) receive few denoising steps, whereas ambiguous or structurally complex tokens (e.g., rare identifiers, long‑range dependencies) receive more steps. After denoising all tokens, we obtain one completed sequence. To exploit candidate diversity we repeat this adaptive denoising `N` times with different random noise seeds, score each full sequence with the AR model (average token log‑likelihood), and keep the highest‑scoring candidate. The pair `(τ, N)` thus controls the trade‑off between denoising‑step investment and candidate‑selection investment, enabling a fine‑grained Pareto analysis of quality versus compute.

---

### Step‑by‑Step Procedure  

1. **Model and Checkpoint Preparation**  
   - Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their matching AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
   - Ensure tokenizer compatibility; if a diffusion checkpoint uses a different tokenizer, replace it with the AR base’s tokenizer (no weight change, still inference‑only).  
   - Load each diffusion model and its AR scorer in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion‑AR pair resident at a time to stay within the 96 GB memory budget.  

2. **Unified Quality Metric (UQM)** – identical to the protocol described in the research problem:  
   - For each benchmark (HumanEval pass@1, pass@10, GSM8K, SIQA, WinoGrande) compute the raw score per model‑condition.  
   - Apply min‑max normalization **within each benchmark** across all conditions so the worst score maps to 0 and the best to 1.  
   - UQM = average of the five normalized scores (range [0, 1]).  

3. **Compute‑Normalized Efficiency Measurement**  
   - **Per‑token diffusion FLOPs**:  
     \[
     F_{\text{diff}}^{(t)} = \alpha \times |\theta| \times s_t
     \]  
     where \(|\theta|\) is the diffusion parameter count, \(s_t\) is the *actual* number of denoising steps executed for token \(t\), and \(\alpha\approx2\) accounts for multiply‑add operations.  
   - **AR scorer FLOPs per token**: a single forward pass of the frozen AR model over the same token (≈ 1 % of diffusion FLOPs per step).  
   - **Total FLOPs for a generated sequence**:  
     \[
     \text{FLOPs}_{\text{seq}} = \sum_{t=1}^{L} \bigl(F_{\text{diff}}^{(t)} + F_{\text{AR}}\bigr)
     \]  
     with fixed sequence length \(L=128\).  
   - **Wall‑clock validation**: warm‑up 100 tokens, then measure average latency per generated token (including AR scoring after each denoising step) using CUDA events; verify linearity between measured latency and the FLOP estimate (R² > 0.95).  
   - Report efficiency as **UQM per FLOP** (or UQM per millisecond).  

4. **Tokenwise Adaptive Denoising with AR‑Guided Early Exit**  
   - **Initialize**: sample a standard Gaussian noise tensor \(\mathbf{z}_T \in \mathbb{R}^{L \times d}\) (same shape as the token embeddings).  
   - **Iterate denoising steps** \(t = T, T-1, \dots, 1\):  
     a. Run the diffusion denoising network to obtain predicted clean token logits \(\mathbf{p}_\theta(\mathbf{z}_t)\).  
     b. Convert logits to a probability distribution and compute the *expected* token embedding \(\hat{\mathbf{e}}_t = \sum_v p_\theta(v| \mathbf{z}_t) \mathbf{e}_v\) (or simply sample a token \(\hat{x}_t\) if preferred).  
     c. **AR guidance query**: feed the partially denoised sequence up to position \(t\) (i.e., \(\hat{x}_{<t}\) plus a padding token for the current position) to the frozen AR base model and obtain the conditional log‑likelihood \(\log p_{\text{AR}}(\hat{x}_t \mid \hat{x}_{<t})\).  
     d. **Early‑exit decision**:  
        - If \(\log p_{\text{AR}}(\hat{x}_t \mid \hat{x}_{<t}) \ge \tau\) **or** the absolute improvement in this log‑likelihood relative to the previous step is < ε, *freeze* token \(t\): set its noise to zero (or to a low‑variance Gaussian) and skip further denoising updates for this token in all subsequent steps.  
        - Otherwise, continue denoising token \(t\) in the next step.  
   - **Termination**: after completing step \(t=1\) (or when all tokens have been frozen), decode the final token sequence \(\mathbf{x}\) from \(\mathbf{z}_0\) (argmax or sampling).  
   - **Candidate generation**: repeat the entire adaptive denoising process \(N\) times with independent noise seeds to obtain \(N\) candidate sequences. Score each candidate with the AR model using the *average* token log‑likelihood (sum of the conditional log‑likelihoods computed during denoising, divided by L). Retain the candidate with the highest AR score.  

   - **Hyper‑parameters to sweep**:  
     - Confidence threshold \(\tau \in \{-\infty, -2.0, -1.5, -1.0, -0.5, 0.0\}\) (where \(\tau=-\infty\) disables early exit, i.e., uniform‑step diffusion).  
     - Minimum improvement ε = 0.01 (fixed, small enough to avoid premature stopping).  
     - Number of candidates \(N \in \{1,2,4,8\}\).  
     - Implicit denoising‑step budget is *emergent*: the average steps per token \(\bar{s} = \frac{1}{L}\sum_t s_t\) will be recorded for each condition.  

5. **Pareto Front Construction**  
   - For each model family (GPT‑2‑based vs. LLaMA‑based DLMs) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((\tau, N)\) condition.  
   - Derive the **empirical Pareto frontier** by keeping points where no other point has both ≥ quality and ≤ compute.  
   - Compute **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
   - Additionally, generate **slice‑wise frontiers**:  
     - Vary \(\tau\) at fixed \(N\) to see the effect of early‑exit aggressiveness.  
     - Vary \(N\) at fixed \(\tau\) to isolate the contribution of candidate selection.  

6. **Trade‑off Analysis**  
   - **Marginal quality gain per denoising step (average)**: fit a piecewise‑linear regression of UQM versus \(\bar{s}\) (holding \(N\) constant) and report the slope \(\Delta\text{UQM}/\Delta\bar{s}\).  
   - **Marginal quality gain per additional candidate**: regress UQM versus \(\log_2(N)\) (holding \(\tau\) constant).  
   - **Marginal quality gain per unit threshold shift**: regress UQM versus \(\tau\) (holding \(N,\bar{s}\) constant) to obtain \(\Delta\text{UQM}/\Delta\tau\).  
   - **Effect of adaptive vs. uniform steps**: for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare UQM achieved by:  
        (a) uniform‑step diffusion with a fixed \(S\) (no early exit, \(\tau=-\infty\)),  
        (b) TADA with the same average \(\bar{s}\) but early exit (\(\tau\) varied),  
        (c) increasing \(N\) with uniform steps.  
     Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences.  
   - **Statistical significance**: declare a strategy superior if the 95 % bootstrap CI of the UQM difference does not contain zero.  
   - **Baseline comparison**: repeat the entire pipeline with the standard reranking approach (generate \(N\) candidates using uniform‑step diffusion, score with AR model, pick best) to quantify how much TADA shifts the Pareto frontier relative to naïve candidate‑only reranking.  

7. **Generalizability Checks**  
   - Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
   - Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
   - Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the TADA experiment to assess scorer‑agnosticism.  

8. **Resource‑aware Implementation Plan (10‑week timeline)**  
   - **Weeks 1‑2**: environment setup, checkpoint download, tokenizer alignment, implement the AR‑guided early‑exit logic (forward pass + conditional log‑likelihood extraction).  
   - **Weeks 3‑4**: build the sampling loop that supports variable \(\tau\) and \(N\); collect baseline UQM and FLOP data for all conditions.  
   - **Week 5**: implement Pareto front extraction, AUPC calculation, and marginal‑gain regressions.  
   - **Week 6**: bootstrap significance testing and baseline (uniform‑step reranking) comparison.  
   - **Weeks 7‑8**: generalization experiments (held‑out prompts, extra DLM, alternative scorers).  
   - **Weeks 9‑10**: write‑up, visualizations (Pareto plots, marginal‑gain bar charts, early‑exit sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

### Rationale  

The target paper identified two concrete gaps: (L6) a sizable “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that lack compute normalization, obscuring any true quality‑efficiency advantage of DLMs over AR baselines. Existing inference‑time strategies (Jacobi Forcing, TESS 2 reward guidance, guided diffusion with gradient‑based guidance, selective denoising after a cheap pass, or tournament‑style successive halving) all treat the denoising budget as *uniform* across token positions or rely on a single global scalar (e.g., guidance weight, number of candidates, or refinement steps).  

TADA introduces a **token‑wise compute allocation mechanism** that directly addresses L6 by letting the model spend more denoising effort where the AR scorer indicates high uncertainty (low conditional likelihood) and less effort where the model is already confident. This yields a non‑uniform step distribution \(\{\,s_t\,\}\) that is *intrinsically tied to the difficulty of each token*, thereby extracting more quality from a fixed total FLOP budget than uniform step allocation can achieve.  

Because the AR scorer is used only to read conditional likelihoods (a single forward pass per token per denoising step), its overhead remains negligible (< 1 % of diffusion FLOPs), satisfying the inference‑only constraint and enabling accurate compute‑normalized efficiency measurement (addressing L9).  

By varying the confidence threshold \(\tau\) we obtain a smooth spectrum from uniform‑step diffusion (\(\tau=-\infty\)) to aggressive early exit (high \(\tau\)), which, together with the candidate count \(N\), sweeps a two‑dimensional space of denoising‑step investment versus candidate‑selection investment. Constructing Pareto fronts in this space lets us quantify whether the headroom is better closed by **spending steps adaptively** (TADA) or by **generating and reranking more samples** (the baseline). The marginal‑gain analyses and bootstrap significance tests provide a rigorous, statistically grounded answer to the research question.  

Finally, the method is fully generalizable: it requires only a diffusion model and any frozen autoregressive scorer (or even a lightweight MLM), needs no training or adaptation, and can be applied to any released DLM checkpoint, sequence length, or benchmark suite. Thus, TADA offers a novel, rigorous, and practical pathway to close the accuracy headroom while providing a compute‑normalized view of the quality‑efficiency trade‑off.