**Method**

1. **Model and Checkpoint Preparation**  
   - Gather the five released DLMs: DiffuGPT‑S, DiffuGPT‑M, DiffuLLaMA‑6.74B, Dream‑7B, LLaDA‑8B, DiffuCoder‑7B.  
   - For each DLM identify its *canonical* AR base model (GPT‑2‑small/medium for the GPT‑2 family; LLaMA‑7B for the LLaMA family).  
   - **Tokenizer verification:** Load the diffusion checkpoint’s tokenizer and the AR base’s tokenizer. Compute the SHA‑256 of their vocabularies; if they match exactly, proceed. If they differ (which does not occur for the selected families), discard that pair from the study – this keeps the inference‑only constraint intact and guarantees a 1‑to‑1 token‑ID mapping without any weight modification.  
   - Load each model in FP16 on the RTX 6000 Pro Blackwell; keep only one model resident at a time to avoid GPU contention.  

2. **Unified Quality Metric (UQM)**  
   - For each benchmark compute the raw score per model‑condition: HumanEval pass@1, HumanEval pass@10, GSM8K accuracy, SIQA accuracy, WinoGrande accuracy.  
   - **Min‑max normalization per benchmark:**  
     \[
     \tilde{s}_{b,c}= \frac{s_{b,c}-\min\limits_{c'} s_{b,c'}}{\max\limits_{c'} s_{b,c'}-\min\limits_{c'} s_{b,c'}}\in[0,1].
     \]  
     This is performed **across all conditions (all models, all (N,S) pairs, and all adaptive strategies)** for each benchmark *b* separately, ensuring each benchmark contributes equally and the score remains interpretable.  
   - **UQM** = average of the five normalized scores, yielding a dimensionless quality in [0,1]. Raw benchmark scores are reported alongside UQM for transparency.  

3. **Compute‑Normalized Efficiency Measurement**  
   - **Per‑model FLOP multiplier (αᵢ):**  
     - Profile a single denoising step (batch = 1, sequence length = L = 128) using `torch.profiler` (or Nsight) to obtain the number of multiply‑add (MAC) operations.  
     - Convert MACs to FLOPs (1 MAC = 2 FLOPs) and solve for αᵢ in  
       \[
       \text{FLOPs}_{\text{step}} = \alpha_i \times |\theta_i| \times L .
       \]  
     - Store αᵢ for each DLM (e.g., GPT‑2‑based ≈ 2.0, LLaMA‑based ≈ 2.1, Dream‑7B ≈ 2.2, LLaDA‑8B ≈ 2.0).  
   - **Total FLOPs per generated token:**  
     \[
     \text{FLOPs}_{\text{token}} = \alpha_i \times |\theta_i| \times S \times L \times N \;+\; \text{FLOPs}_{\text{AR‑scorer}},
     \]  
     where *S* is the denoising‑step budget, *N* the number of candidates, and `FLOPs_AR‑scorer` is the forward‑pass cost of the frozen AR base model for one candidate (measured once per model; empirically ≤ 1 % of the diffusion term).  
   - **Wall‑clock validation:**  
     - Warm‑up with 100 generated tokens, then measure average latency per token (including candidate generation, scoring, and selection) using CUDA events over 500 tokens.  
     - Fit a linear regression of measured latency versus predicted FLOPs; retain the model only if \(R^2>0.95\).  
   - **Efficiency:** Report **quality per FLOP** (UQM ÷ FLOPs) and, for intuition, **quality per millisecond** (UQM ÷ latency).  

4. **Candidate Generation and Lightweight Reranking**  
   - **Sampling hyper‑parameters (fixed for all experiments):** temperature = 0.8, top‑p = 0.95, top‑k = 0 (i.e., nucleus sampling only), seed‑based reproducibility.  
   - **Search space:** \(N\in\{1,2,4,8\}\) candidates, denoising step budget \(S\in\{8,16,32,64,128\}\).  
   - **Scoring:** For each candidate compute the average negative log‑likelihood (NLL) under the frozen AR base model over the full generated sequence; lower NLL → higher AR score.  
   - **Static selection:** Choose the candidate with the lowest NLL as the model output for that prompt.  
   - **Adaptive compute allocation (novel contribution):**  
     - Define a total FLOP budget \(B\) corresponding to the maximum step budget \(S_{\max}=128\) for a *single* candidate:  
       \[
       B = \alpha_i \times |\theta_i| \times S_{\max} \times L .
       \]  
     - **Stage 1 (exploration):** Generate \(N_0\) candidates with a low step budget \(S_0\) (default \(S_0=8\)). FLOPs used:  
       \[
       B_1 = N_0 \times \alpha_i \times |\theta_i| \times S_0 \times L .
       \]  
     - **Stage 2 (refinement):**  
       - Rank the \(N_0\) candidates by AR NLL (ascending).  
       - Select the top \(k = \lfloor N_0/2 \rfloor\) candidates for refinement.  
       - Distribute the remaining FLOPs equally:  
         \[
         \Delta S = \left\lfloor \frac{B - B_1}{k \times \alpha_i \times |\theta_i| \times L} \right\rfloor .
         \]  
       - For each selected candidate, continue denoising from its current noisy state for an additional \(\Delta S\) steps (i.e., run the diffusion model for \(S_0+\Delta S\) steps total). Unselected candidates remain at their Stage 1 output.  
       - Pseudocode is provided in the Appendix.  
     - **Final selection:** Score all \(N_0\) candidates (now with heterogeneous step counts) using the AR NLL and pick the best.  
   - **Sensitivity analysis:** Repeat the adaptive procedure with alternative \((S_0,k)\) pairs \(\{(4,\lfloor N/2\rfloor),(8,\lfloor N/3\rfloor),(16,\lfloor N/2\rfloor)\}\).  
   - **Bottleneck check:** Measure wall‑clock time for the AR scorer per candidate; verify scorer overhead stays < 2 % of total latency even for the largest \(N\).  

5. **Pareto Front Construction**  
   - For each model family (GPT‑2‑based vs. LLaMA‑based) collect tuples \((\text{FLOPs}_{\text{token}},\text{UQM})\) for every static \((N,S)\) condition and every adaptive strategy.  
   - **Empirical Pareto frontier:** retain points where no other point has both ≥ UQM and ≤ FLOPs.  
   - **Area under the Pareto curve (AUPC):** sort frontier points by increasing FLOPs and compute the trapezoidal integral of UQM; higher AUPC indicates a more favorable trade‑off.  
   - **Hypervolume indicator (optional):** compute the dominated volume relative to a reference point \((\text{FLOPs}_{\text{ref}},\text{UQM}_{\text{ref}})\) where \(\text{FLOPs}_{\text{ref}}\) is the 95‑th percentile of observed FLOPs and \(\text{UQM}_{\text{ref}}=0\).  

6. **Trade‑off Analysis**  
   - **Marginal quality gain per denoising step:** Fit a generalized additive model (GAM) with a tensor‑product smooth term for \(S\) and \(\log_2(N)\) (to capture possible interactions). Extract the average partial derivative \(\partial \text{UQM}/\partial S\) (holding \(N\) constant) with 95 % confidence intervals via bootstrap (10 000 resamples of prompts).  
   - **Marginal quality gain per additional candidate:** From the same GAM, report \(\partial \text{UQM}/\partial \log_2(N)\) (holding \(S\) constant).  
   - **Effect of lightweight reranking:**  
     - Construct a baseline frontier where candidates are selected uniformly at random (AR scorer disabled).  
     - For each bootstrap resample compute the difference in AUPC (reranked − random) and the UQM shift at fixed FLOP levels (0.5×, 1×, 2× the FLOPs of the base AR model).  
     - Apply a Bonferroni correction for the three FLOP levels; declare a shift significant if the corrected 95 % confidence interval excludes zero.  
   - **Validation of AR NLL as a quality proxy:**  
     - On a held‑out subset of 200 prompts (randomly drawn from each benchmark) compute the Spearman correlation between candidate NLL and binary benchmark success (pass/fail).  
     - Additionally, fit an isotonic regression to map NLL to estimated success probability and report the Brier score.  
   - **Candidate independence check:**  
     - Encode each candidate with a frozen Sentence‑BERT model (all‑miniLM‑L6‑v2).  
     - Compute the average pairwise cosine distance; report mean and variance. Target mean distance > 0.4 (i.e., average cosine similarity < 0.6) to ensure sufficient diversity.  

7. **Generalizability Checks**  
   - **Held‑out prompts:** HumanEval‑plus, MBPP, and a subset of BIG‑Bench Hard (BBH) reasoning tasks limited to ≤ 128 tokens (e.g., word sorting, object counting).  
   - **Additional DLM:** Evaluate a 3B parameter diffusion model from the dLLM zoo (e.g., `stable-diffusion-lm-3B` or a 3B checkpoint released by the Dream team) using the same pipeline.  
   - **Alternative lightweight scorers:**  
     - Distilled 60M‑parameter AR model (TinyLLaMA‑60M).  
     - Frozen MLM (BERT‑base) scored via masked language‑model likelihood (averaged over positions).  
     - Compare reranking efficacy (AUPC gain) against the base AR scorer.  
   - **Continuous‑diffusion / flow‑matching baseline:** Include the YAN model (MoE‑FM) from the “Towards Faster Language Model Inference Using Mixture‑of‑Experts Flow Matching” paper to verify that the quality‑compute trade‑off patterns hold beyond discrete diffusion.  
   - **Instruction‑tuned DLMs:** Test Dream‑Instruct and DiffuCoder‑Instruct (if publicly released) to see whether the adaptive strategy generalizes to models already aligned for downstream tasks.  
   - **Second‑GPU validation:** Repeat a 20 % random subset of experiments on an RTX 4090 (24 GB) to confirm that the FLOP‑to‑latency mapping (\(R^2>0.95\)) holds across architectures.  
   - **Sequence‑length ablation:**  
     - Run the full static grid (N,S) with \(L\in\{128,256,512\}\) (generating the full length, then truncating to the first 128 tokens for evaluation).  
     - Measure the change in UQM and FLOPs to quantify the impact of longer contexts.  
   - **Multilingual check:** Use the Flores‑101 dev set (English→German) and compute BLEU as an auxiliary quality dimension; report whether the Pareto conclusions shift when a non‑English benchmark is added.  

8. **Resource‑aware Implementation Plan (10‑Week Timeline)**  
   - **Weeks 1‑2:** Environment setup, checkpoint download, tokenizer verification, per‑model FLOP profiling (αᵢ), AR‑scorer latency measurement.  
   - **Weeks 3‑4:** Implement static candidate generation loops; collect UQM and FLOP data for all \((N,S)\) combos; store raw benchmark scores.  
   - **Week 5:** Implement adaptive allocation strategy; gather data for dynamic \((N,S)\) conditions; perform sensitivity sweeps over \((S_0,k)\).  
   - **Week 6:** Bootstrap analysis (10 k resamples), GAM fitting with interaction terms, AUPC/hypervolume computation, multiple‑comparison correction.  
   - **Week 7:** Validation of AR NLL scorer (Spearman/Brier), candidate‑independence diagnostics (Sentence‑BERT).  
   - **Week 8:** Generalizability experiments (held‑out prompts, extra 3B DLM, alternative scorers, YAN flow‑matching model, instruction‑tuned DLMs, second‑GPU validation, sequence‑length ablation, multilingual BLEU).  
   - **Weeks 9‑10:** Write‑up, visualisation (Pareto plots, marginal‑gain interaction heatmaps, sensitivity plots), preparation of reproducibility package (Dockerfile, scripts, seeded random numbers, profiling logs, tokenizer verification script).  

---

**Rationale**

The revised method directly tackles the two open questions from the target paper while addressing every concern raised in the reviews.

*Accuracy headroom (L6):* By systematically varying denoising steps (**S**) and candidate count (**N**) and estimating marginal quality gains via a GAM that includes an \(S \times \log_2(N)\) interaction term, we quantify whether investing compute in refinement or in diversity yields a higher return. The adaptive two‑stage allocation strategy provides a principled, inference‑time mechanism to shift compute from exploration to refinement based on cheap AR scores, directly testing whether a hybrid approach can dominate the static Pareto frontier.

*Compute‑normalization gap (L9):* Architecture‑specific FLOP multipliers (αᵢ) are obtained empirically from actual MAC counts, eliminating the bias of a universal constant. Wall‑clock latency validation ensures that FLOP estimates reflect real hardware consumption. Efficiency is expressed as quality per FLOP (or per ms), and Pareto fronts summarized with AUPC/hypervolume give a threshold‑free, statistically comparable summary of the quality‑compute trade‑off.

*Lightweight AR reranker:* The AR base model’s NLL is used as a cheap, likelihood‑based proxy for candidate quality. We empirically validate this proxy (Spearman correlation with benchmark success, Brier score from isotonic regression) and verify that scorer overhead remains negligible. The method also explores alternative scorers (distilled AR, MLM) to determine if the benefit is scorer‑agnostic.

*Novelty beyond prior work:* Unlike Jacobi Forcing (which distills AR behavior into a parallel decoder **during training**) or TESS 2 (which applies a fixed reward guidance at inference), our adaptive strategy dynamically reallocates inference‑time compute **after** seeing cheap AR scores, embodying a divergent‑convergent think‑and‑refine loop without any training cost. The GAM‑based marginal‑gain analysis with interaction terms and the AUPC metric provide fresh analytical lenses on the denoising‑step vs. candidate‑selection trade‑off.

*Rigor & validity:*  
- Tokenizer handling is restricted to families with identical vocabularies, guaranteeing a true 1‑to‑1 ID mapping without weight changes.  
- UQM uses min‑max normalization per benchmark across all conditions, preserving interpretability and avoiding distributional assumptions.  
- FLOP estimation is grounded in measured MACs; latency regression with \(R^2>0.95\) confirms the model’s fidelity to real hardware.  
- Bootstrap significance testing with Bonferroni correction controls family‑wise error across multiple FLOP levels.  
- All hyper‑parameters, seeds, library versions, and tokenizer verification scripts are explicitly documented for full reproducibility.  
- Sensitivity analyses on adaptive parameters and sequence‑length ablations test the robustness of conclusions.

*Generalizability:*  
- We test across five distinct DLMs spanning two AR families, on coding, mathematical, and commonsense benchmarks, plus held‑out prompts, an additional 3B DLM, alternative scorers, a continuous‑diffusion/flow‑matching model (YAN), instruction‑tuned DLMs, a second GPU architecture, longer sequence lengths (up to 512 tokens), and a multilingual BLEU benchmark.  
- This breadth ensures that observed trends are not artifacts of a single model, benchmark, sequence length, or language.

Overall, the method delivers a clear, reproducible, and statistically sound roadmap for practitioners who must allocate a fixed inference budget between denoising refinement and candidate selection in diffusion language models, thereby closing the accuracy headroom and providing a compute‑normalized efficiency assessment that was missing in prior work.