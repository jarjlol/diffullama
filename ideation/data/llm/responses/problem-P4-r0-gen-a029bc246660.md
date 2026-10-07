**Problem:**  
How does the choice of attention‑masking strategy (causal, bidirectional, or hybrid) applied at inference time affect the quality‑efficiency trade‑off of scaled diffusion language models that have been adapted from autoregressive checkpoints, and does this relationship differ across model families (GPT‑2‑based vs. LLaMA‑based) and model scales (127 M – 7 B)?

**Rationale:**  
The target paper shows that adapting open‑source autoregressive (AR) checkpoints to masked diffusion yields competitive models (DiffuGPT, DiffuLLaMA) without costly pre‑training from scratch. However, several limitations point to an under‑examined inference‑time design choice: the attention mask used during the denoising process.  

1. **Connection to reported weaknesses** –  
   - *Shift operation* (L2) and *attention‑mask annealing* (L1) are both tied to how the model attends to context during diffusion. The authors annealed the mask during training but removed it at scale, claiming minimal impact, yet their own ablation showed the benefit grows with scale.  
   - The *[MASK]* token handling (L3) and the *adaptation recipe* (L4) were selected via proxy tasks, suggesting that the true influence of attention conditioning on the final diffusion dynamics has not been validated directly.  
   - Recent follow‑up work (e.g., UNIFUSION, Don’t Retrain, Align, TESS 2) explores alternative training objectives or representation alignment, but few studies systematically vary the *inference‑time* attention pattern while keeping the adapted checkpoint fixed.  

2. **Why attention masking matters for diffusion LMs** –  
   In masked diffusion, each denoising step conditions on a partially noised sequence. The attention mask determines which positions can influence each other:  
   - **Causal mask** preserves the autoregressive ordering the model saw during pre‑training, potentially retaining AR‑specific biases (relevant to L2).  
   - **Bidirectional mask** allows full context exchange, which may improve infilling and bidirectional reasoning but could conflict with the causal priors learned in AR pre‑training.  
   - **Hybrid mask** (causal for the prefix, bidirectional for the masked region) mirrors the training objective described in the target paper (causal attention within the observed prompt, full attention within the masked target) and is hypothesized to give the best trade‑off.  

   By varying this mask at inference we can directly test whether the model’s performance hinges on preserving AR‑style causality, benefits from full bidirectional context, or requires a hybrid compromise—without any additional training.  

3. **Feasibility under the given constraints** –  
   - **Inference‑only**: All experiments can be run on the released checkpoints (diffusionfamily/diffugpt‑s/m, diffusionfamily/diffulla­ma, LLaDA‑8B, Dream‑7B, DiffuCoder‑7B). No fine‑tuning, continual pre‑training, or AR‑to‑diffusion adaptation is needed.  
   - **Compute**: A single NVIDIA RTX 6000 Pro Blackwell (96 GB) suffices for batched inference; changing the attention mask only alters the mask tensor passed to the model’s forward pass, incurring negligible overhead.  
   - **Timeline**: Implementing mask wrappers, running perplexity/generation benchmarks (WikiText‑103, HumanEval infill, ARC/Hellaswag, AlpacaEval), measuring latency/FLOPs, and analyzing results can be completed within ten weeks with a team of seven (three GPU‑enabled members for inference, four for CPU‑side analysis, evaluation harness construction, and statistical reporting).  
   - **Available resources**: Tokenizers and model architectures are already compatible with the checkpoints; we can reuse existing evaluation scripts from the target paper’s repository and related works (e.g., Dream, TESS 2).  

4. **Significance** –  
   - **Scientific**: Clarifying how attention masking influences the diffusion process will reveal whether the success of AR‑to‑diffusion adaptation relies on retaining causal structure, exploiting bidirectional context, or needing a nuanced hybrid approach. This informs future adaptation recipes and can explain the observed shift‑operation and annealing effects.  
   - **Practical**: Identifying the optimal inference‑time mask could yield immediate quality or speed gains for deployed diffusion LMs (e.g., better fill‑in‑the‑middle performance, lower latency for a given quality target) without retraining.  
   - **Broader impact**: The findings will generalize to other adaptation strategies (uniform‑noise diffusion, representation alignment, classifier‑free guidance) by highlighting a shared, low‑cost lever for tuning the quality‑efficiency frontier of diffusion language models.  

In summary, systematically probing inference‑time attention masking offers a clear, original, and feasible research direction that directly addresses several limitations of the target paper, leverages only the released checkpoints, and promises both theoretical insight and practical improvements for scaled diffusion language models.