# Limitations of DiffuLLaMA and the Continuation Research Problem

**Base paper:** Gong et al., *Scaling Diffusion Language Models via Adaptation from Autoregressive
Models* (DiffuLLaMA), ICLR 2025.

**Methodology:** produced by a lightweight multi-agent pipeline (Extractor → Analyzer → Citation
Agent → Judge → Master) combining techniques from three papers: BAGELS' explicit/implicit span
extraction (Al Azher et al., 2025, arXiv:2505.18207), LimitGen's four-aspect limitation taxonomy and
RAG-grounded generation (Xu et al., 2025, arXiv:2507.02694), and the role architecture of Multi-Agent
LLMs for Generating Research Limitations (Al Azher et al., 2026, arXiv:2601.11578). Full methodology,
including every intermediate agent's raw output and what was filtered and why, is in `data/` and
`REPORT.md`.

## Research gaps and limitations

- **[L1] The attention-mask-annealing ablation contradicts the paper's own justification for
  dropping it.** The paper's Table 3 shows annealing's benefit growing with scale — +2.1 accuracy
  points at 124M, +2.5 at 355M — yet the paper calls this "minimal impact" and omits annealing
  entirely at 7B "to simplify implementation using flash-attention 2," a ~20x extrapolation past the
  largest model it was actually tested on. *(Experimental Design — Lack of Ablation Studies)*

- **[L2] The shift-operation ablation is roughly five times larger than the annealing effect, yet
  receives far less scrutiny.** Removing the shift operation costs -11.9 to -15.2 accuracy points
  (vs. annealing's +2.1/+2.5), but the shift is inherited directly from AR training dynamics and its
  own potential side effects at scale are never examined the way annealing's are. *(Experimental
  Design — Lack of Ablation Studies)*

- **[L3] The `[MASK]` token implementation is an admitted workaround, inconsistently applied and
  confounded with model size.** The paper states it "should" expand the vocabulary but instead
  reuses an existing token "considering practical issues on implementation" — except for DiffuGPT-M,
  which does get a genuinely new token. DiffuGPT-M is also the best-performing model at its scale,
  and this is never disentangled from the mask-token treatment. *(Methodological — Inappropriate
  Method)*

- **[L4] Instruction tuning is explicitly deferred despite the paper's own evidence that it's
  needed.** The paper reports that adding chain-of-thought prompting *hurts* DiffuLLaMA's accuracy,
  attributes this to "the absence of instruction tuning," cites prior work showing diffusion LMs
  specifically benefit from it — and defers it to future work regardless. *(Experimental Design —
  Insufficient Baselines)*

- **[L5] The infilling-capability comparison is self-flagged as potentially unfair.** The paper
  withholds suffix information from the model for its infilling evaluation and explicitly notes
  this "might result in an unfair comparison" against baselines that could use it. *(Experimental
  Design — Inappropriate Datasets)*

- **[L6] The model is reported as undertrained, with no convergence shown.** The paper states there
  is "still scope for training more, since the model does not show signs of saturation" — meaning
  every reported result is a lower bound on the training recipe's actual potential, not a converged
  outcome. *(Experimental Design — Limited Datasets/training budget)*

- **[L7] No compute-parity (FLOPs/forward-pass) accounting is reported, despite diffusion's lack of
  KV-caching.** The paper reports wall-clock decoding latency against one AR baseline configuration,
  but diffusion models require full self-attention over the entire sequence at every denoising step
  with no caching — and no FLOPs-normalized or forward-pass-normalized comparison is given, so the
  wall-clock win doesn't establish whether an equally-optimized AR setup (e.g., speculative decoding)
  would close the gap. *(Result Analysis — Insufficient Metrics)*

- **[L8] High answer uncertainty is reported but not investigated.** hit@3 substantially exceeds
  single-answer accuracy across every reasoning benchmark (e.g., 57.7 vs. 23.6 on SATMath) — the
  paper's own diagnosis is "the current model exhibits high uncertainty about its responses," but no
  calibration or verifier-based mitigation is attempted. *(Result Analysis — Limited Analysis)*

- **[L9] "Code generation" is claimed as a capability but only fill-in-the-middle infilling is
  evaluated.** The abstract and introduction list code generation among DiffuLLaMA's strengths; the
  only code-related evaluation in the paper is HumanEval single-line infilling (prefix and suffix
  both given), not open-ended generation from a specification — a narrower task than the claim
  implies. *(Literature Review — Inaccurate Description of scope vs. claim)*

- **[L10] The adaptation recipe is validated on only two AR model families, up to 7B.** Every result
  is GPT2-derived or LLaMA2-derived. A fresh literature search for this analysis found no paper
  testing whether the recipe transfers to other AR backbones; the only closely related model found
  (LLaDA) is trained from scratch rather than adapted, sidestepping the question rather than
  answering it. *(Literature Review — Limited Scope)*

## Research problem

DiffuLLaMA's central claim — that continual pre-training can convert a pretrained autoregressive
model into a competitive diffusion language model — rests on an adaptation recipe with at least two
components (attention-mask annealing, the inherited shift operation) whose costs at the scale the
paper actually cares about (7B) are either extrapolated from models 20x smaller (L1) or not examined
at all despite being the larger effect in the paper's own ablation (L2). This is compounded by a
third, more localized design choice — the mask-token workaround (L3) — that is applied
inconsistently across model sizes in a way the paper never isolates from scale itself. Together,
these gaps point to a single underlying question the base paper opens but does not answer: **does
DiffuLLaMA's adaptation recipe, as validated at 124M-355M scale, actually hold at the 7B scale the
paper's headline results depend on, or does dropping the costlier parts of the recipe for
implementation convenience leave a quantifiable gap between DiffuLLaMA and what a fully-annealed,
properly-disentangled adaptation at 7B would achieve?** The continuation problem this motivates is a
direct, ablation-based audit of DiffuLLaMA's own adaptation recipe at scale: reintroduce attention-
mask annealing and a mask-token treatment disentangled from model size in a 7B (or largest-feasible)
adaptation run, and measure whether the recipe's own internal justifications — "minimal impact,"
implementation convenience — survive contact with the scale the paper was actually built for. This
framing is provisional pending the team's own direction decision (documented as still-open in this
project's `project-docs/02-decision-log.md`, item P-1); L4, L7, L8, and L10 above point toward
related but distinct continuation problems (instruction-tuning ablation, inference-cost accounting,
answer-calibration, and cross-architecture generalization, respectively) that remain available if the
team's eventual direction diverges from the annealing/mask-token audit this paragraph centers on.
