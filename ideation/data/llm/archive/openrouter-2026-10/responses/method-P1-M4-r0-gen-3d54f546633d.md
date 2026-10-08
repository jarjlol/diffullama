**Method: AR‑Guided Selective Denoising (AGSD)**  

1. **Model and Checkpoint Preparation**  
   - Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
   - Ensure each diffusion checkpoint uses the same tokenizer as its AR base; if a mismatch exists, replace the diffusion model’s tokenizer with the AR base’s tokenizer (no weight change, still inference‑only).  
   - Load models in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

2. **Unified Quality Metric (UQM)**  
   - For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
   - Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
   - UQM = (average of the five normalized scores). This yields a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

3. **Compute‑Normalized Efficiency Measurement**  
   - **Diffusion FLOPs per denoising step**: \(F_{\text{diff}} = \alpha \times |\theta| \times L\) (α≈2 for multiply‑add, \(|\theta|\) = diffusion parameter count, L=128).  
   - **AR scorer FLOPs per token**: a single forward pass of the frozen AR base model over the same sequence length; empirically ≤ 1 % of diffusion FLOPs.  
   - **Total FLOPs for a condition**:  
     \[
     \text{FLOPs}_{\text{total}} = N \times \bigl[ S_0 \times F_{\text{diff}} + S_1 \times F_{\text{diff}} \bigr] + N \times F_{\text{AR}}
     \]
     where \(S_0\) is the initial low‑step budget, \(S_1=S-S_0\) the refinement‑step budget, and the final \(F_{\text{AR}}\) term accounts for scoring the refined candidates (≤ 1 % of diffusion FLOPs).  
   - **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including both denoising passes and AR scoring) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
   - Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

4. **AR‑Guided Selective Denoising Procedure**  
   - **Step 0 – Initial low‑step generation**: For each prompt, sample **N ∈ {1, 2, 4, 8}** independent diffusion trajectories, each using a small denoising‑step budget **S₀ ∈ {4, 8}** (chosen so that \(S_0 \ll S_{\max}\)). Each trajectory yields a full sequence \(\mathbf{x}^{(i)}_0\).  
   - **Step 1 – Token‑wise uncertainty estimation**: Feed each \(\mathbf{x}^{(i)}_0\) to the frozen AR base model and compute the token‑level negative log‑likelihood (NLL) or, equivalently, the token‑wise entropy \(H_t^{(i)} = -\sum_v p_t(v)\log p_t(v)\). Define an uncertainty mask  
     \[
     M^{(i)}_t = \begin{cases}
     1 & \text{if } H_t^{(i)} > \tau \\
     0 & \text{otherwise}
     \end{cases}
     \]
     where the threshold \(\tau\) is set per‑model as the 75‑th percentile of entropy across all tokens in the batch (so roughly the top‑25 % most uncertain tokens are masked).  
   - **Step 2 – Selective refinement**: For each candidate, keep the high‑confidence tokens fixed and diffuse only the masked positions. Concretely, construct a partially noisy tensor \(\mathbf{z}^{(i)}_T\) where entries corresponding to unmasked tokens are set to the clean embedding (or a low‑variance Gaussian) and masked entries are sampled from the standard diffusion noise. Run the diffusion denoising network for the remaining **S₁ = S – S₀** steps **only on the masked entries** (the network still processes the full sequence, but the gradient w.r.t. unmasked positions is zero‑ed because their noise is fixed). This yields a refined candidate \(\mathbf{x}^{(i)}_1\).  
   - **Step 3 – Scoring and selection**: Compute the AR‑model NLL (or log‑likelihood) of each refined candidate \(\mathbf{x}^{(i)}_1\). Retain the candidate with the lowest NLL as the model’s output for that prompt.  
   - **Conditions explored**:  
     - Initial step budget \(S₀ \in \{4, 8\}\) (kept small to keep compute cheap).  
     - Total denoising budget \(S \in \{8, 16, 32, 64, 128\}\) (so \(S₁ = S - S₀\)).  
     - Number of candidates \(N \in \{1, 2, 4, 8\}\).  
     - (Optional) Uncertainty‑mask percentile \(\tau\) ∈ {60 %, 70 %, 80 %} to test sensitivity.  

5. **Pareto Front Construction**  
   - For each model family (GPT‑2‑based vs. LLaMA‑based) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every (\(S₀, S, N\)) condition.  
   - Derive the **empirical Pareto frontier** by retaining points where no other point has both higher or equal quality and lower or equal compute.  
   - Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
   - Additionally, generate **slice‑wise frontiers**: (i) varying \(S\) at fixed \(S₀,N\) to see the effect of refinement steps; (ii) varying \(N\) at fixed \(S₀,S\) to isolate the candidate‑selection contribution; (iii) varying \(S₀\) at fixed \(S,N\) to assess the impact of the initial cheap generation.  

6. **Trade‑off Analysis**  
   - **Marginal quality gain per denoising step (refinement phase)**: fit a piecewise‑linear regression of UQM versus \(S₁\) (holding \(S₀,N\) constant) and report the slope \(\Delta\text{UQM}/\Delta S₁\).  
   - **Marginal quality gain per additional candidate**: fit a similar regression of UQM versus \(\log_2(N)\) (holding \(S₀,S\) constant).  
   - **Marginal quality gain per initial cheap step**: regress UQM against \(S₀\) (holding \(S,N\) constant) to obtain \(\Delta\text{UQM}/\Delta S₀\).  
   - **Effect of selective refinement vs. uniform steps**: for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by (a) uniform denoising with \(S₀=0\) (i.e., standard \(S\) steps across all tokens), (b) the AGSD procedure with the same total \(S\), and (c) increasing \(N\) with uniform steps. Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the UQM differences.  
   - **Statistical significance**: declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  
   - **Baseline comparison**: repeat the entire pipeline with the standard reranking approach (generate \(N\) candidates with uniform \(S\) steps, score with AR model, pick best) to quantify how much AGSD shifts the Pareto frontier relative to naïve candidate‑only reranking.  

7. **Generalizability Checks**  
   - Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
   - Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
   - Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the AGSD experiment to assess scorer‑agnosticism.  

8. **Resource‑aware Implementation Plan (10‑week timeline)**  
   - **Weeks 1‑2**: environment setup, checkpoint download, tokenizer alignment, implement token‑wise entropy computation and masking logic.  
   - **Weeks 3‑4**: build the two‑stage denoising loop (initial cheap pass → masked refinement pass); collect baseline UQM and FLOP data for all (\(S₀,S,N\)) combos.  
   - **Week 5**: implement Pareto front extraction, AUPC calculation, and marginal‑gain regressions.  
   - **Week 6**: bootstrap significance testing and baseline (uniform‑step reranking) comparison.  
   - **Weeks 7‑8**: generalization experiments (held‑out prompts, extra DLM, alternative scorers).  
   - **Weeks 9‑10**: write‑up, visualizations (Pareto plots, marginal‑gain bar charts, uncertainty‑mask sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Rationale**  

The target paper’s unresolved issues are: (L6) a sizable “accuracy headroom” that could be closed by either more denoising steps or better exploitation of candidate diversity, and (L9) efficiency claims that are not compute‑normalized, obscuring whether DLMs truly beat AR models. Existing inference‑time tricks (Jacobi Forcing, TESS 2 reward guidance) show that compute can be shifted between denoising and selection, but they treat the whole sequence uniformly.  

AGSD attacks the headroom from a different angle: it uses the lightweight AR scorer not merely to rank full candidates, but to **detect where each candidate is uncertain** and then **focus additional denoising compute on those uncertain tokens**. By allocating the denoising budget adaptively—cheap, diverse generation followed by targeted refinement—AGSD extracts more value from a fixed compute budget than either (i) uniformly increasing denoising steps for all tokens or (ii) simply generating more full candidates and reranking them.  

Because the AR model is only used for token‑level entropy computation and final scoring, the method remains strictly inference‑only, uses only publicly released checkpoints, and adds negligible overhead (the AR forward pass is ≤ 1 % of diffusion FLOPs). The unified quality metric and FLOP‑normalized efficiency measurement enable a fair, compute‑aware comparison across model families. The Pareto‑front analysis directly quantifies whether AGSD shifts the frontier upward (higher quality at equal compute) compared to uniform‑step baselines and candidate‑only reranking, thereby answering the core question of how the quality–compute trade‑off changes when we vary denoising steps versus number of generation candidates, and whether a lightweight AR reranker can improve that trade‑off.  

Finally, the plan is feasible within the ten‑week timeline on a single RTX 6000 Pro Blackwell: the two‑stage denoising loop adds only a modest constant factor to the existing candidate‑generation pipeline, and all analysis (bootstrapping, Pareto extraction, generalization) can be parallelized across the three GPU‑enabled team members while the remaining four handle CPU‑only tasks such as evaluation harness construction and result aggregation. This makes AGSD a clear, innovative, rigorous, valid, and generalizable approach to the stated research problem.