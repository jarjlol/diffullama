**Method:**  
**Guidance‑Driven Diffusion Sampling with a Lightweight Autoregressive Scorer**  

1. **Model and Checkpoint Preparation**  
   - Assemble the five released DLMs (DiffuGPT‑S/M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B) and their corresponding AR base models (GPT‑2‑small/medium, LLaMA‑7B).  
   - Verify that each diffusion checkpoint uses the same tokenizer as its AR base; if a mismatch exists, swap the diffusion model’s tokenizer for the AR base’s tokenizer (no weight change, still inference‑only).  
   - Load all models in FP16 on the RTX 6000 Pro Blackwell; keep only one diffusion model and its AR scorer resident at a time to stay within the 96 GB memory budget.  

2. **Unified Quality Metric (UQM)**  
   - For each benchmark (HumanEval pass@1, pass@10; GSM8K; SIQA; WinoGrande) compute the raw score per model‑condition.  
   - Apply min‑max normalization **within each benchmark** across all conditions so that the worst score maps to 0 and the best to 1.  
   - UQM = (average of the five normalized scores). This yields a dimensionless quality score in [0, 1] where each benchmark contributes equally.  

3. **Compute‑Normalized Efficiency Measurement (including guidance overhead)**  
   - **Diffusion FLOPs per denoising step**:  \(F_{\text{diff}} = \alpha \times |\theta| \times L\)  (α≈2 for multiply‑add, \(|\theta|\) = diffusion parameter count, L=128).  
   - **Guidance FLOPs per step**: a forward pass of the AR scorer to obtain token‑level log‑likelihoods, followed by a backward pass to compute the gradient of the summed log‑likelihood w.r.t. the input embeddings. Empirically a forward + backward pass costs ≈ 2 × \(F_{\text{AR}}\) where \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\).  
   - **Total FLOPs per denoising step**:  \(F_{\text{step}} = F_{\text{diff}} + 2 \times F_{\text{AR}}\).  
   - **AR scorer overhead for reranking (baseline)**: a single forward pass of the AR model over the final decoded sequence (≤ 1 % of diffusion FLOPs) and is added only when evaluating the baseline reranking strategy.  
   - **Wall‑clock validation**: run a 100‑token warm‑up, then measure average latency per generated token (including guidance gradient computation) using CUDA events; verify linearity between measured latency and FLOP estimate (R² > 0.95).  
   - Report efficiency as **quality per FLOP** (or quality per millisecond) for each condition.  

4. **Guidance‑Driven Sampling Procedure**  
   - For each prompt, initialize a random noise tensor \(\mathbf{z}_T\) (standard diffusion schedule).  
   - At each denoising step \(t = T, T-1, \dots, 1\):  
     a. Run the diffusion model’s denoising network to obtain the predicted clean token logits \(\mathbf{p}_\theta(\mathbf{z}_t)\).  
     b. Convert logits to a probability distribution and sample a tentative token sequence \(\hat{\mathbf{x}}_t\) (or use the expected embedding).  
     c. **Guidance step**: feed \(\hat{\mathbf{x}}_t\) (or its embedding) to the frozen AR base model, compute the summed log‑likelihood \(\log p_{\text{AR}}(\hat{\mathbf{x}}_t)\), and back‑propagate to obtain the gradient \(\nabla_{\mathbf{z}_t} \log p_{\text{AR}}(\hat{\mathbf{x}}_t)\).  
     d. Update the noisy latent: \(\mathbf{z}_{t-1} = \mathbf{z}_t - \eta \bigl[ \nabla_{\mathbf{z}_t} \log p_{\theta}(\mathbf{z}_t) - \lambda \, \nabla_{\mathbf{z}_t} \log p_{\text{AR}}(\hat{\mathbf{x}}_t) \bigr]\), where \(\eta\) is the diffusion step size (fixed by the scheduler) and \(\lambda \ge 0\) is a **guidance weight** that controls the strength of the AR scorer.  
   - After the final step \(t=0\), decode \(\mathbf{z}_0\) to obtain the output sequence.  
   - **Candidate generation**: repeat the above guided sampling process \(N\) times (independent noise seeds) to obtain \(N\) guided candidates; retain the candidate with the highest AR scorer log‑likelihood (equivalent to picking the lowest NLL).  
   - **Conditions explored**:  
     - Guidance weight \(\lambda \in \{0.0, 0.2, 0.5, 1.0, 2.0\}\) (λ=0 corresponds to standard diffusion, i.e., no guidance).  
     - Denoising‑step budget \(S \in \{8, 16, 32, 64, 128\}\) (by truncating the schedule after S steps).  
     - Number of candidates \(N \in \{1, 2, 4, 8\}\).  
   - For each (\(\lambda, S, N\)) tuple we record UQM and total FLOPs (computed as \(S \times F_{\text{step}} + N \times\) overhead for final AR scoring of candidates).  

5. **Pareto Front Construction**  
   - For each model family (GPT‑2‑based vs. LLaMA‑based) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every (\(\lambda, S, N\)) condition.  
   - Derive the **empirical Pareto frontier** by keeping points where no other point has both higher or equal quality and lower or equal compute.  
   - Compute the **area under the Pareto curve (AUPC)** as a scalar summary of efficiency; higher AUPC indicates a more favorable quality‑compute trade‑off.  
   - Additionally, generate **slice‑wise frontiers**: (i) varying \(\lambda\) at fixed \(S,N\) to see the effect of guidance strength; (ii) varying \(S\) at fixed \(\lambda,N\) to isolate the denoising‑step contribution; (iii) varying \(N\) at fixed \(\lambda,S\) to isolate the candidate‑selection contribution.  

6. **Trade‑off Analysis**  
   - **Marginal quality gain per denoising step**: fit a piecewise‑linear regression of UQM versus \(S\) (holding \(\lambda,N\) constant) and report the slope \(\Delta\text{UQM}/\Delta S\).  
   - **Marginal quality gain per additional candidate**: fit a similar regression of UQM versus \(\log_2(N)\) (holding \(\lambda,S\) constant).  
   - **Marginal quality gain per unit guidance weight**: regress UQM against \(\lambda\) (holding \(S,N\) constant) to obtain \(\Delta\text{UQM}/\Delta\lambda\).  
   - **Effect of guidance vs. pure step increase**: for a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare the UQM achieved by (a) increasing \(S\) with \(\lambda=0\), (b) increasing \(N\) with \(\lambda=0\), and (c) increasing \(\lambda\) while keeping \(S,N\) low. Use paired bootstrap (10 000 resamples of prompts) to obtain confidence intervals for the differences in UQM.  
   - **Statistical significance**: declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not contain zero.  
   - **Baseline comparison**: repeat the entire pipeline with the standard reranking approach (generate \(N\) candidates with \(\lambda=0\), score with AR model, pick best) to quantify how much the guidance‑driven method shifts the Pareto frontier relative to candidate‑only reranking.  

7. **Generalizability Checks**  
   - Hold‑out evaluation on HumanEval‑plus and MBPP to ensure observations are not benchmark‑specific.  
   - Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
   - Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the guidance experiment to assess scorer‑agnosticism.  

8. **Resource‑aware Implementation Plan (10‑week timeline)**  
   - **Weeks 1‑2**: environment setup, checkpoint download, tokenizer alignment, implement guidance gradient computation (forward + backward pass through AR scorer).  
   - **Weeks 3‑4**: build the sampling loop that supports variable \(S\), \(\lambda\), and \(N\); collect baseline UQM and FLOP data for all conditions.  
   - **Week 5**: implement Pareto front extraction, AUPC calculation, and marginal‑gain regressions.  
   - **Week 6**: bootstrap significance testing and baseline (reranking) comparison.  
   - **Week 7‑8**: generalization experiments (held‑out prompts, extra DLM, alternative scorers).  
   - **Weeks 9‑10**: write‑up, visualizations (Pareto plots, marginal‑gain bar charts, guidance‑weight sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

---

**Rationale:**  
The proposed method directly tackles the two open issues highlighted in the target paper.  

*Accuracy headroom (L6):* By injecting the AR scorer’s gradient into the denoising process, we continuously steer the diffusion trajectory toward regions of high AR likelihood. This exploits the model’s internal candidate diversity **without** needing to generate and store many full sequences; the guidance acts as an infinite‑sample, low‑overhead reranker that refines a single trajectory in‑place.  

*Compute‑normalization gap (L9):* Guidance adds a predictable, quantifiable overhead (forward + backward pass of a lightweight AR model) that we include in the FLOP budget. Consequently, every (\(\lambda, S, N\)) condition is evaluated on a **strict compute‑normalized** basis, allowing a fair comparison between investing compute in more denoising steps, more candidates, or stronger guidance.  

Unlike the previously suggested approach—which relies on static candidate generation followed by a two‑stage adaptive re‑allocation—this method integrates selection and denoising into a single, gradient‑driven loop. It therefore offers a distinct angle: rather than treating denoising and reranking as separate stages, we treat the AR scorer as a **guidance signal** that shapes the diffusion process itself, enabling a finer‑grained trade‑off exploration and potentially shifting the Pareto frontier upward more efficiently than merely increasing \(S\) or \(N\). The method remains fully inference‑only, uses only publicly released checkpoints, fits the ten‑week schedule on a single RTX 6000 Pro Blackwell, and yields results that are directly comparable to the baseline reranking strategy via shared quality and efficiency metrics.