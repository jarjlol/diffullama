**Problem:**  
How does the choice of noise schedule (e.g., linear, cosine, context‑adaptive, and a simple learned schedule) affect the quality‑efficiency frontier of scaled diffusion language models adapted from autoregressive checkpoints, and does this relationship differ across model families (GPT‑2‑based vs. LLaMA‑based) and scales (127 M – 7 B)?

**Rationale:**  

The target paper shows that continual pre‑training of open‑source autoregressive (AR) checkpoints yields competitive diffusion language models (DLMs) such as DiffuGPT and DiffuLLaMA, but it also highlights several limitations that motivate an inference‑only investigation of the noise schedule:

1. **Undertrained models (L7).** The adapted DLMs are reported to be undertrained relative to their AR counterparts, leaving a quality gap. A more effective noise schedule can reduce the difficulty of the denoising process at inference time, potentially narrowing this gap without additional training.

2. **Non‑compute‑normalized efficiency reporting (L9).** The original work measures only wall‑clock latency, ignoring the fact that different noise schedules change the amount of computation per denoising step (e.g., by altering the proportion of tokens that remain noisy). By reporting a compute‑normalized metric—approximated FLOPs = model size × steps × (effective token count) × sequence length—we obtain a fairer quality‑efficiency trade‑off curve for each schedule.

3. **Narrow infilling evaluation (L8).** Infilling was assessed only on single‑line HumanEval with gold prefix and suffix. We will evaluate a broader suite of infilling tasks (multi‑line HumanEval, MBPP, and open‑ended text completion) to see whether certain schedules generalize better to realistic masking scenarios.

4. **Architectural generality (L10).** The adaptation recipe was applied only to GPT‑2 and LLaMA families up to 7 B. Noise schedules interact with the model’s attention pattern (causal vs. bidirectional) and may benefit one family more than the other—for instance, context‑adaptive schedules that preserve bidirectional context could align better with LLaMA‑based models, while simple linear schedules might suit GPT‑2‑based models. Testing across families and scales will reveal such interactions.

5. **Connection to alternative routes (L10, L5).** Related work shows that representation‑alignment or uniform‑noise adaptation can be competitive with continued denoising training. By treating the noise schedule as an inference‑only “knob,” we explore whether a simple change at sampling time can approach the performance of those alternative routes without retraining.

6. **Statistical rigor (L11).** We will run multiple seeds (e.g., 5) for each configuration and report mean ± standard deviation, addressing the lack of variance estimates in the original paper.

7. **Instruction‑following deficiency (L5).** The authors deferred instruction tuning despite evidence that it improves performance. We will include an instruction‑following benchmark (IFEval) to determine whether certain schedules better preserve the model’s ability to follow prompts.

**Experimental design (inference‑only, feasible under constraints):**  

- **Models:** DiffuGPT‑S/M (GPT‑2‑based), DiffuLLaMA‑6.74 B, Dream‑7B, DiffuCoder‑7B, LLaDA‑8B (LLaMA‑based).  
- **Noise schedules:** (i) linear βₜ, (ii) cosine βₜ (as in improved DDPM), (iii) context‑adaptive schedule used in Dream 7B (noise level conditioned on token‑wise entropy), (iv) a simple learned schedule (a small MLP predicting βₜ from timestep, trained on a held‑out slice of the adaptation corpus—this step uses only the released checkpoint’s forward pass and does not modify weights).  
- **Denoising steps:** 16, 32, 64 (to capture the quality‑efficiency curve).  
- **Sequence length:** fixed at 128 tokens (typical for infilling and reasoning prompts).  
- **Metrics:**  
  - *Quality:* perplexity on WikiText‑103, exact‑match pass@k on HumanInfilling (k = 1,10), MAUVE on open‑ended generation (Prompts from C4), IFEval instruction‑following score.  
  - *Efficiency:* approximated FLOPs per generated token (model size × steps × average noisy token fraction × seq length) and wall‑clock latency per token (batch size = 1).  
- **Analysis:** For each model family and scale, plot quality vs. compute‑normalized efficiency for each schedule; compute the area under the curve (AUC) to summarize the frontier; test whether schedule effects interact significantly with family and scale using two‑way ANOVA.  

**Feasibility:** All experiments use released checkpoints; no pre‑training, continual pre‑training, or weight updates are required. A single NVIDIA RTX 6000 Pro Blackwell (96 GB) can batched inference for the proposed configurations within the ten‑week timeline (GPU‑enabled members parallelize over models and schedules; CPU‑only members handle metric computation, statistical analysis, and write‑up).  

**Significance:** This work provides the first systematic, inference‑only characterization of how noise scheduling shapes the quality‑efficiency trade‑off of scaled DLMs across families and scales. By offering a training‑free lever to mitigate undertraining, improve compute‑normalized efficiency, and enhance infilling and instruction‑following performance, the results give practitioners a concrete knob for deploying adapted DLMs. Moreover, uncovering family‑ and schedule‑specific interactions can guide future DLM design (e.g., selecting schedules that better match the model’s attention bias), thereby extending the adaptation paradigm introduced in the target paper without costly retraining.