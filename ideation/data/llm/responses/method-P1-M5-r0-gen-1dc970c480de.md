**Method:**  
**AR‑Guided Importance Resampling for Diffusion Language Models (AGIR‑DLM)**  

**Rationale:**  
The target paper points out two gaps: (L6) a sizable “accuracy headroom” that could be closed either by more denoising steps or by better exploiting the model’s candidate diversity, and (L9) the lack of compute‑normalized efficiency claims that prevents a fair comparison with autoregressive (AR) baselines. Existing inference‑time tricks either (i) add a gradient‑based guidance signal (Method 2) or (ii) prune/refine candidates based on token‑wise uncertainty (Method 3). Both approaches still treat the denoising process as a uniform, per‑token budget or as a binary mask‑based refinement.  

AGIR‑DLM takes a different angle: it treats the set of diffusion trajectories as a **sampled proposal distribution** and uses the lightweight AR scorer to **re‑weight** those proposals in an importance‑sampling loop. By repeatedly resampling high‑scoring trajectories and allocating additional denoising steps only to the resampled set, the method **concentrates compute on the most promising regions of the search space** while preserving the diversity that comes from sampling multiple noise seeds. This directly attacks the accuracy headroom by making better use of candidate diversity, and it provides a clear, compute‑normalized way to trade off denoising steps against the number of candidates (via the number of importance‑resampling stages and the step budget per stage). Because the AR model is only used for scoring (forward pass) and never updated, the approach stays strictly within the inference‑only constraint and can be implemented with the same FLOP‑bookkeeping used in the baseline plans.  

The method proceeds as follows:  

1. **Model and Checkpoint Preparation** – identical to the baseline (load each released DLM and its AR base model in FP16, align tokenizers, keep one diffusion model + its AR scorer resident at a time).  

2. **Unified Quality Metric (UQM)** – mean of min‑max‑normalized scores across HumanEval pass@1/10, GSM8K, and SIQA/WinoGrande, as defined in the problem statement.  

3. **Compute‑Normalized Efficiency Measurement** –  
   - Diffusion FLOPs per denoising step: \(F_{\text{diff}} = \alpha \times |\theta| \times L\) (α≈2, \(|\theta|\) = diffusion parameter count, L=128).  
   - AR‑scorer FLOPs per forward pass: \(F_{\text{AR}} = \alpha \times |\theta_{\text{AR}}| \times L\) (≤ 1 % of \(F_{\text{diff}}\)).  
   - For an AGIR‑DLM condition with **R** resampling rounds, **N** candidates per round, and **S** denoising steps per candidate per round, total FLOPs per token are:  
     \[
     \text{FLOPs}_{\text{total}} = R \times N \times \bigl[ S \times F_{\text{diff}} + F_{\text{AR}} \bigr].
     \]  
   - Wall‑clock latency is measured on the RTX 6000 Pro Blackwell (100‑token warm‑up, then average latency per generated token) to validate the FLOP estimate (target \(R^2>0.95\)).  
   - Efficiency is reported as **quality per FLOP** (or quality per millisecond).  

4. **AGIR‑DLM Sampling Procedure** – for each evaluation prompt:  

   a. **Initial proposal generation** – draw \(N\) independent noise vectors \(\{\mathbf{z}^{(i)}_T\}_{i=1}^N\) and run the diffusion denoising network for a small base budget \(S_0\) steps (e.g., \(S_0=8\)), obtaining raw candidates \(\{\mathbf{x}^{(i)}_0\}\).  

   b. **Scoring** – compute the AR‑model log‑likelihood (or equivalently, the negative log‑likelihood) of each candidate: \(s^{(i)} = \log p_{\text{AR}}(\mathbf{x}^{(i)}_0)\).  

   c. **Importance resampling** – normalize scores to obtain weights \(w^{(i)} = \frac{\exp(s^{(i)})}{\sum_j \exp(s^{(j)})}\). Resample \(N\) indices \(\{i'\}\) according to the categorical distribution defined by \(\{w^{(i)}\}\) (with replacement). This yields a multiset of high‑scoring proposals that will receive additional denoising effort.  

   d. **Refinement** – for each resampled index \(i'\), continue denoising the corresponding latent \(\mathbf{z}^{(i')}_{T-S_0}\) for an additional \(S_{\text{ref}}\) steps (e.g., \(S_{\text{ref}} = S - S_0\)), producing refined candidates \(\{\mathbf{x}^{(i')}_1\}\).  

   e. **Iterate** – repeat steps (b)–(d) for a total of \(R\) importance‑resampling rounds. After the final round, select the candidate with the highest AR score as the model output for that prompt.  

   f. **Hyper‑parameter grid** – explore:  
      - Base budget \(S_0 \in \{4,8\}\) (kept small to keep early proposals cheap).  
      - Refinement budget per round \(S_{\text{ref}} \in \{8,16,32,64\}\) (so total steps per candidate = \(S_0 + R \times S_{\text{ref}}\)).  
      - Number of candidates per round \(N \in \{1,2,4,8\}\).  
      - Number of resampling rounds \(R \in \{1,2,3\}\).  
   This grid lets us independently vary **(i)** the amount of denoising work per candidate (via \(S_0, S_{\text{ref}}, R\)) and **(ii)** the degree of candidate exploitation (via \(N\) and the resampling mechanism).  

5. **Pareto Front Construction** – for each model family (GPT‑2‑based vs. LLaMA‑based DLMs) plot UQM (y‑axis) against total FLOPs per token (x‑axis) for every \((S_0, S_{\text{ref}}, R, N)\) condition. Derive the empirical Pareto frontier (no other point dominates in both higher quality and lower/equal compute). Compute the **area under the Pareto curve (AUPC)** as a scalar efficiency summary.  

6. **Trade‑off Analysis** –  
   - **Marginal quality gain per denoising step**: regress UQM against total steps per candidate (\(S_0 + R \times S_{\text{ref}}\)) while holding \(N\) and \(R\) constant; report \(\Delta\text{UQM}/\Delta S\).  
   - **Marginal quality gain per additional candidate**: regress UQM against \(\log_2(N)\) while holding step budget constant.  
   - **Marginal quality gain per resampling round**: regress UQM against \(R\) while holding \(S_0, S_{\text{ref}}, N\) constant.  
   - For a fixed compute budget (e.g., 1×, 2×, 4× the FLOPs of the base AR model), compare three strategies using paired bootstrap (10 000 prompt resamples):  
        1. **Increase denoising steps only** (\(N=1, R=1\), varying \(S\)).  
        2. **Increase candidates only** (\(S=S_0, R=1\), varying \(N\)).  
        3. **AGIR‑DLM** (jointly varying \(N\) and \(R\) under the same FLOP ceiling).  
   - Declare a strategy superior if the 95 % bootstrap confidence interval of the UQM difference does not exclude zero.  
   - As a baseline, repeat the entire pipeline with the simple reranking approach (generate \(N\) candidates with uniform \(S\) steps, score with AR model, pick best) to quantify how much AGIR‑DLM shifts the Pareto frontier relative to naïve candidate‑only reranking.  

7. **Generalizability Checks** –  
   - Hold‑out evaluation on HumanEval‑plus and MBPP to confirm findings are not benchmark‑specific.  
   - Test an additional, unseen diffusion model (e.g., a 3B‑parameter model from the dLLM zoo) to verify that trends extrapolate across architectures.  
   - Alternative lightweight scorers: replace the AR base model with a distilled 60M‑parameter Transformer or a frozen BERT‑style MLM and repeat the AGIR‑DLM experiment to assess scorer‑agnosticism.  

8. **Resource‑aware Implementation Plan (10‑week timeline)** –  
   - **Weeks 1‑2**: environment setup, checkpoint download, tokenizer alignment, implement FLOP‑profiling harness and AR‑scorer forward pass.  
   - **Weeks 3‑4**: build the importance‑resampling loop (supporting variable \(S_0, S_{\text{ref}}, R, N\)); collect baseline UQM and FLOP data for all conditions.  
   - **Week 5**: Pareto front extraction, AUPC computation, marginal‑gain regressions.  
   - **Week 6**: bootstrap significance testing and baseline (simple reranking) comparison.  
   - **Weeks 7‑8**: generalization experiments (held‑out prompts, extra DLM, alternative scorers).  
   - **Weeks 9‑10**: write‑up, visualizations (Pareto plots, marginal‑gain bar charts, resampling‑round sensitivity curves), prepare reproducibility package (Dockerfile, scripts, seeded RNG).  

**Why AGIR‑DLM is substantively different:**  
- It does **not** rely on gradient‑based guidance (unlike Method 2) nor on token‑wise uncertainty masking (unlike Method 3).  
- It treats the AR scorer as a **weighting function** for importance resampling, thereby explicitly leveraging the *distribution* of candidate scores rather than only the top‑1 score or a binary mask.  
- The compute allocation is **structured across rounds**: each round can trade a modest increase in denoising steps for a potentially large boost in candidate quality via resampling, offering a finer‑grained control over the denoising‑step vs. candidate‑number trade‑off than the static (N,S) grid or the two‑stage selective denoising of Method 3.  
- The method remains fully inference‑only, uses only publicly released checkpoints, and fits within the stipulated hardware and timeline constraints.  

By quantifying how importance‑resampling‑driven refinement moves the Pareto frontier, AGIR‑DLM directly answers the research question: it reveals whether the accuracy headroom is better closed by spending compute on more denoising steps, on more candidates, or on a principled combination of both via lightweight AR‑guided resampling.