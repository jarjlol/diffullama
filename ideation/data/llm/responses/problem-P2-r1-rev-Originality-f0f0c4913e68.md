## Systematic Evaluation of Originality

### Step 1: Reading the Problem & Context

The research problem investigates how inference-time noise schedule choice (linear, cosine, learned) affects the quality–efficiency trade-off of scaled DLMs adapted from AR checkpoints, across model families (GPT-2 vs. LLaMA) and scales (127M–7B). It is framed as a training-free, inference-only empirical study.

**Key contextual observations from the existing literature:**

- **Dream 7B** (Related #1) already introduced "context-adaptive token-level noise rescheduling" as a core technique, directly exploring how noise scheduling affects DLM quality.
- **UNIFUSION** (Related #3) established connections among different noise objectives and corruption kernels, showing that noise parameterization is an active and nuanced area.
- **Jacobi Forcing** (Related #10) explored inference-time paradigm shifts for diffusion models, demonstrating that inference-time optimization is a recognized research direction.
- The broader diffusion literature (DDPM, DDIM, DPM-Solver) has extensively studied noise schedules in continuous diffusion, establishing that schedule choice fundamentally shapes the quality–efficiency frontier.

### Step 2: Generating Review & Feedback

**Analysis of Originality:**

The core question — "does noise schedule choice matter for DLMs?" — is **not novel**. It is a natural and well-motivated question, but:

1. **Noise schedule effects in diffusion models are well-established.** The impact of linear vs. cosine schedules on diffusion trajectories has been studied extensively in the continuous diffusion literature and partially in the DLM literature (Dream 7B, UNIFUSION).

2. **Dream 7B already explored noise rescheduling for DLMs.** Their "context-adaptive token-level noise rescheduling" directly addresses how schedule design affects DLM performance, reducing the novelty of the proposed investigation.

3. **The problem is fundamentally an empirical sweep.** It proposes trying different schedules on existing models and measuring outcomes. While the cross-family and cross-scale dimensions add specificity, the core methodology (inference-time schedule variation + metric collection) is incremental rather than transformative.

4. **No new methodology, architecture, or theory is introduced.** The problem does not propose a new noise schedule family, a new training paradigm, or a new theoretical framework for understanding why schedules matter differently across architectures.

5. **The framing as addressing "limitations L7 and L9"** of the target paper is somewhat contrived — it defines the problem relative to one paper's shortcomings rather than identifying a genuine gap in the field.

**Genuine but modest contributions:**
- The systematic cross-family comparison (GPT-2 vs. LLaMA) for schedule sensitivity is not extensively done.
- Compute-normalized efficiency metrics (perplexity/GFLOP) provide a more rigorous framing than raw perplexity.
- The hypothesis about architectural curvature matching is interesting but speculative.

### Step 3: Rating

**Rating: 2**

The problem exhibits **minimal originality**. While it adds some specificity through cross-family comparison and compute-normalized analysis, the core question — how noise schedules affect DLM quality-efficiency trade-offs — is a natural and somewhat obvious extension of well-established ideas in diffusion modeling. Dream 7B and UNIFUSION have already explored noise scheduling in the DLM context, and the broader diffusion literature has extensively studied this topic. The problem is a well-motivated empirical follow-up rather than a novel challenge or unique perspective.

---

**Review:** The research problem is well-defined and practically motivated, but its originality is limited. The core investigation — how inference-time noise schedules affect DLMs — is a natural extension of established diffusion modeling principles and has already been partially explored by Dream 7B (noise rescheduling) and UNIFUSION (noise kernel connections). The problem does not introduce novel methods, architectures, or theoretical frameworks; it is primarily an empirical sweep across existing models and schedules.

**Feedback:** To strengthen originality, consider: (1) Identifying a genuine gap that existing noise schedule studies have not addressed — e.g., whether schedules interact with specific architectural innovations (MoE, Mamba) in ways not captured by GPT-2/LLaMA studies; (2) Proposing a novel schedule design principle informed by the observed cross-family differences, rather than merely sweeping over predefined schedules; (3) Connecting the schedule-family/architecture interaction to a deeper theoretical understanding of why certain schedules suit certain architectures, moving beyond empirical observation toward mechanistic explanation. Without such additions, the problem risks being perceived as a benchmark-style follow-up rather than a contribution that sets a new direction.

**Rating (1-5): 2**