**Problem:**  
*Does the shift operation inherited from autoregressive training introduce a positional bias that degrades bidirectional infilling in adapted diffusion language models, and can simple inference‑time modifications (e.g., removing or symmetrizing the shift) mitigate this bias without retraining?*  

**Rationale:**  
The target paper shows that removing the shift operation causes a large drop in performance (‑11.9 to ‑15.2 accuracy points), yet the shift’s effect has never been examined at scale or linked to autoregressive positional biases. Because the shift is directly carried over from autoregressive training dynamics, it may cause the model to treat early and late token positions asymmetrically during the denoising process, harming tasks that require true bidirectionality (e.g., multi‑span infilling, bidirectional reasoning).  

We can test this hypothesis using only the released checkpoints (DiffuGPT‑S/M, DiffuLLaMA, Dream‑7B, etc.) by:  

1. **Probing positional bias** – extract hidden states at each denoising step for a set of probe sentences and train lightweight linear classifiers (no model updates) to predict token position (absolute or relative) from the representations. Comparing classification accuracy between the standard model and a version where the shift term is subtracted from the timestep embedding will reveal whether the shift encodes positional information.  

2. **Infilling evaluation** – run multi‑span infilling benchmarks (e.g., the FIM‑Plus or Multi‑Span Infill dataset) that require filling two or more gaps simultaneously. Measure exact‑match and token‑level F1 scores under the default shift and under a shift‑removed inference setting (implemented by feeding timestep t − shift instead of t).  

3. **Uncertainty and pass@k analysis** – generate multiple completions per prompt (e.g., 16 samples) and compute hit‑rate vs. single‑answer accuracy on reasoning benchmarks (GSM8K, CSQA) to see whether mitigating the shift reduces the gap between oracle and model performance, as suggested by L6.  

4. **Compute‑normalized efficiency** – measure wall‑clock latency and FLOPs per denoising step for both configurations, enabling a fair comparison with autoregressive baselines (GPT‑2/LLaMA) under the same compute budget.  

These experiments require only forward passes, hidden‑state extraction, and lightweight probing—all feasible on a single RTX 6000 Pro Blackwell within ten weeks, leveraging the team’s GPU members for inference and CPU members for probe training and analysis.  

If the shift is found to carry autoregressive positional bias and its removal improves bidirectional generation without prohibitive cost, the study will directly address limitation L2, offer a zero‑training‑cost mitigation strategy, and provide deeper insight into how autoregressive pretraining influences diffusion‑based language models. This contributes both theoretically (understanding the role of the shift) and practically (a simple inference‑time tweak that can be adopted by practitioners using existing checkpoints).