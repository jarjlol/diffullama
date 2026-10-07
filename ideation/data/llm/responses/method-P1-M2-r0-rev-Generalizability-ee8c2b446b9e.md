**Review:** The method demonstrates moderate generalizability within the discrete diffusion language model (DLM) paradigm, supported by testing across five architectures and two model families, but its applicability is bounded by strong architectural assumptions and a narrow benchmark scope.

**Feedback:** 
*Strengths:* The study spans multiple DLM families (DiffuGPT, DiffuLLaMA, Dream, LLaDA, DiffuCoder) and validates robustness through alternative lightweight scorers and held-out benchmarks (HumanEval‑plus, MBPP). The explicit FLOP‑normalized efficiency framework is reusable for other inference‑time allocation studies.

*Limitations:* (1) The guidance mechanism requires white‑box gradient access to the diffusion latent $\mathbf{z}_t$, which excludes black‑box or compiled/optimized deployments where internal states are unavailable. (2) Evaluation is restricted to English text benchmarks (code, math, commonsense), limiting claims about multilingual or multimodal generalization. (3) The “unseen” DLM is drawn from the same dLLM zoo rather than a fundamentally different architecture (e.g., flow‑matching or continuous diffusion). (4) The AR scorer is limited to decoder‑only transformers; the method’s compatibility with encoder‑decoders or non‑AR scorers is untested. (5) The compute model (FLOPs ≈ α·|θ|·L) may not transfer to hardware‑specific optimized kernels or memory‑bound regimes.

*Recommendations:* To strengthen generalizability, test on multilingual benchmarks, continuous/flow‑based DLMs, and black‑box inference settings where only final logits are accessible. Additionally, validate the FLOP‑to‑latency model on different hardware generations.

Rating (1-5): 3