**Review:**

The proposed method is a well-structured, inference-only experimental protocol that directly addresses the research problem of whether AR-representation preservation in DLMs determines the effectiveness of lightweight reranking versus additional denoising steps. The method demonstrates strong alignment with the target paper (DiffuLLaMA/DiffuGPT) and related work (REPR-ALIGN, Jacobi Forcing, TESS 2, PreDiff-LM), and incorporates several thoughtful design choices:

**Strengths:**
- Architecture-matched scorers eliminate a key confounding variable
- Behavioral validation (Step 7) provides construct validity for the CKA preservation metric
- Non-linear step-gain modeling (Step 8) avoids the assumption of linear diminishing returns
- Honest acknowledgment of n=5 limitations with appropriate exploratory framing
- Comprehensive ablation suite (scorer size, alternative preservation metrics, hidden-state extraction points, candidate diversity)
- Bootstrap CIs with prompt-level resampling properly reflect uncertainty

**Weaknesses:**
- With only 5 DLMs, even bootstrap CIs for Spearman's ρ have very wide uncertainty; the core predictive claim may remain underdetermined regardless of methodology
- The Unified Quality Score (simple average across code, math, and commonsense benchmarks) may obscure domain-specific effects where preservation matters differently
- The layer-weighting scheme (exponential decay with λ=0.5) is somewhat arbitrary despite ablation coverage
- Efficiency ratio ε_i assumes FLOPs proxy for real-world cost, which may not hold across architectures
- The oracle headroom computed from K=8 candidates may underestimate true headroom

**Feedback:**
The method is scientifically sound and well-integrated with existing literature. To strengthen validity, consider: (1) explicitly discussing whether n=5 DLMs can ever support a meaningful test of the preservation-gain hypothesis, potentially reframing the primary contribution as a proof-of-concept with generalizable methodology rather than a definitive causal claim; (2) reporting domain-specific UQS breakdowns alongside the composite metric to detect heterogeneous effects; (3) justifying the exponential layer-weighting form with reference to known representation evolution across DLMs layers (e.g., lower layers = syntax, upper layers = semantics). The protocol is reproducible and feasible within constraints.

**Rating (1-5): 4**