**Review:**

The method demonstrates a thoughtfully modular and model-agnostic design, with interchangeable preservation metrics (CKA, SVCCA, Procrustes, linear probing), swappable scorers, and a unified quality metric that can accommodate alternative benchmarks. The ablation suite (Step 9) systematically tests robustness to scorer choice, hidden-state extraction point, layer weighting, normalization reference, and candidate diversity—strong evidence that the authors have considered whether findings are artefacts of specific implementation choices. The behavioral validation (Step 7) adds construct validity by linking geometric preservation to functional AR-like behavior.

However, several constraints limit true generalizability:

1. **Construct scope**: The core hypothesis—AR representation preservation predicts reranking benefit—only meaningfully applies to DLMs converted from AR backbones via continual pretraining. Models like Dream 7B and LLaDA 8B, which were not AR-to-DLM conversions, muddy this construct. The method doesn't discuss how preservation would be defined or measured for flow-matching-based DLMs or DLMs trained from scratch.

2. **Empirical ceiling (n=5)**: Despite honest framing as exploratory, five DLMs severely constrain any predictive or generalizable claim about the preservation-gain relationship. The bootstrap CIs reflect prompt-level uncertainty but cannot compensate for model-level under-sampling.

3. **Narrow deployment context**: English-only benchmarks (HumanEval, GSM8K, SIQA, WinoGrande) limit applicability to multilingual or non-text settings. The efficiency framework (reranking vs. more denoising steps) is specific to the diffusion sampling paradigm and doesn't extend to alternative inference-time strategies (flow matching, speculative decoding, early exiting).

4. **Hardware bottleneck**: The single RTX 6000 Pro (96 GB) constraint caps model size, potentially excluding larger DLMs where preservation dynamics might differ qualitatively.

5. **Scorer limitation**: The method only tests AR scorers; it's unclear whether the preservation-gain relationship holds with non-AR scorers (e.g., a discriminative model or another DLM as scorer).

**Feedback:**

To strengthen generalizability, the authors should: (a) clarify which DLMs in the study genuinely have AR backbones and justify inclusion of models that don't; (b) explicitly define how preservation would be measured for non-AR-converted DLMs (e.g., flow-matching models); (c) consider adding at least one multilingual or domain-shifted benchmark to test whether preservation scores transfer across data distributions; (d) discuss whether the efficiency-comparison framework (reranking vs. extra steps) is portable to other inference paradigms; and (e) acknowledge that the n=5 design, while honestly framed, fundamentally limits the scope of any generalizable claim—perhaps reframing the study as a proof-of-concept that motivates larger-scale validation.

**Rating (1-5): 3**

The method exhibits a reasonable level of adaptability within its defined scope (discrete AR-converted DLMs on English benchmarks), with modular design and systematic ablation checks. However, the core construct does not transfer to all DLMs, the empirical validation is limited to five models, and the efficiency framework is tied to a specific inference paradigm—preventing broader claims of generalizability.