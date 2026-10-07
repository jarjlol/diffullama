**Problem:**  
How does the denoising‑step budget affect the quality‑efficiency trade‑off of scaled diffusion language models adapted from autoregressive checkpoints, and does this relationship differ across model families (GPT‑2‑based vs. LLaMA‑based) and scales?

**Rationale:**  
The target paper demonstrates that continual pre‑training of open‑source autoregressive (AR) models yields competitive diffusion language models (DLMs) such as DiffuGPT and DiffuLLaMA. However, its evaluation fixes a single inference budget (e.g., 256–1000 denoising steps) and compares wall‑clock latency to AR baselines without normalizing for compute, leaving it unclear whether reported gains stem from superior modeling or simply from more inference steps. Moreover, the paper’s infilling and reasoning assessments are narrow (single‑line HumanEval infilling, limited reasoning benchmarks), so it is unknown whether increasing steps improves structured, bidirectional generation such as whole‑function code synthesis or long‑range dependency tasks.  

Several limitations highlighted in the target paper point directly to these gaps:  

* **L8 – Infilling claims are narrow:** Only single‑line infilling with gold prefix/suffix is evaluated, leaving the broader capability of DLMs for structured code generation untested.  
* **L9 – Efficiency claim is not compute‑normalized:** Latency is reported at batch size 1 against a single AR configuration, without FLOPs‑ or step‑normalized comparison, making it impossible to tell if quality improvements require disproportionate compute.  
* **L6 – Accuracy headroom remains unexplored:** Hit‑rate over generated candidates exceeds single‑answer accuracy, suggesting that more inference steps (or better selection) could close the gap, but this is never examined.  

Because our resources restrict us to inference‑only experiments on released checkpoints, we can systematically vary the denoising‑step count at inference time—a zero‑cost manipulation that directly probes the step‑quality‑latency relationship. We will evaluate five publicly available DLMs that span different architectures and scales:  

* DiffuGPT‑S (127 M) and DiffuGPT‑M (355 M) – GPT‑2‑based  
* DiffuLLaMA (6.74 B) – LLaMA‑based  
* Dream‑7B – a strong open diffusion LLM  
* LLaDA‑8B – another recent diffusion LM  
* DiffuCoder‑7B – a code‑specialized diffusion LM  

For each model we will:  

1. **Sweep denoising steps** (e.g., 16, 32, 64, 128, 256, 512, 1000) and record:  
   * **Quality metrics** – perplexity on WikiText‑103, pass@k on HumanEval (single‑line and whole‑function infilling), GSM8K reasoning accuracy, and commonsense (SIQA/WinoGrande) scores.  
   * **Efficiency metrics** – wall‑clock latency per token, FLOPs estimated from step count × model size, and memory footprint.  
2. **Plot quality‑efficiency curves** to identify step budgets where diminishing returns appear and to compare the curves across model families and scales.  
3. **Analyze whether larger models achieve higher quality at lower step budgets**, testing the hypothesis that scaling shifts the frontier favorably.  
4. **Investigate structured infilling** by generating whole functions from natural‑language specifications (using MBPP or HumanEval‑style prompts) and measuring pass@k as a function of steps, directly addressing L8.  
5. **Explain the accuracy headroom** (L6) by measuring how candidate selection (e.g., reranking with a small scorer) improves performance as steps increase.  

This study is feasible within the ten‑week timeline: all experiments involve only forward passes on a single RTX 6000 Pro Blackwell (96 GB), which comfortably fits the largest checkpoint (6.74 B) with batch size 1. The seven‑person team can split tasks (model harness implementation, evaluation script development, latency measurement, analysis, and write‑up).  

**Significance:** By providing the first compute‑normalized, step‑scaled analysis of multiple adapted diffusion LMs, the work will clarify whether the promising results of diffusion language models stem from intrinsic modeling gains or merely from increased inference compute. It will also deliver concrete guidance for practitioners on choosing denoising budgets when deploying scaled DLMs, and it will illuminate whether diffusion models truly excel at bidirectional, structured generation tasks beyond the narrow infilling settings examined to date. This directly addresses the identified limitations (L6, L8, L9) and advances the understanding of how scaling autoregressive checkpoints into diffusion models impacts the quality‑efficiency trade‑off.