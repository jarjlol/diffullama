# Generation Step 1: Outline (QUAL-SG Task 2, RAG-based Survey Generation)

Survey topic: "Diffusion Language Models: Adaptation from Autoregressive Models,
Continual Pre-training, and Non-Autoregressive Text Generation"

Per QUAL-SG's design, the generator first produces a structured outline from the
retrieved top-K papers (to avoid a brief, unstructured one-shot survey), then
expands each section using the referenced abstracts as grounding context.

1. Introduction
2. Background: From Diffusion Models to Non-Autoregressive Text Generation
3. Text Diffusion Models
   3.1 Continuous Diffusion for Text
   3.2 Discrete and Masked Diffusion for Text
   3.3 Initializing Diffusion Language Models from Pretrained (Masked) LMs
   3.4 Scaling Masked Diffusion Language Models (2024-2026)
4. Adapting and Continually Pre-training Language Models
   4.1 Continual / Domain-Adaptive Pre-training of Autoregressive LMs
   4.2 Architecture Transfer: From Transformers to Alternative Architectures
   4.3 Adapting Autoregressive Models into Diffusion Models
5. Non-Autoregressive Text Generation
   5.1 Foundational Non-Autoregressive Machine Translation
   5.2 Semi-Autoregressive and Position-Aware Models
   5.3 Non-Autoregressive Generation Beyond Translation
   5.4 Constrained and Insertion-Based Generation
6. Recent Developments Since DiffuLLaMA (2025-2026): Scaling, Reasoning, and Applications
7. Open Challenges and Future Directions
8. Conclusion
