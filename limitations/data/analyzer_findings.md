# Analyzer Agent — inferred implicit limitations

Per the Multi-Agent paper's Analyzer role (arXiv:2601.11578 §IV.B): a methodological audit of the
whole paper to surface weaknesses the authors do not explicitly flag as limitations — confounded
variables, inadequate ablations, unaddressed scope. Each item below is grounded in a direct quote
or table from `anchor_fulltext.txt`, read fresh for this pass (not carried over from any prior
audit of this repo).

**A1 — The attention-mask-annealing ablation contradicts its own "minimal impact" conclusion.**
Table 3 (§4.5): annealing's contribution is **+2.1** accuracy points at 124M (43.3→45.4) and
**+2.5** at 355M (47.2→49.7) — the gain *grows*, not shrinks, with scale. The paper's own words:
*"The mask annealing has minimal impact, so we choose to omit it for 7B adaptation to simplify
implementation using flash-attention 2."* No ablation of any kind is run at 7B. The 20x
extrapolation (355M → 6.74B) runs against the direction the paper's own two data points suggest.

**A2 — The shift-operation ablation is far larger than the annealing one, yet gets one sentence.**
Same table: DD-without-shift scores 33.5 / 34.5 vs. full DD's 45.4 / 49.7 — a drop of **-11.9** and
**-15.2** points, roughly 5x the size of the annealing effect the paper does discuss at length. The
shift operation is inherited directly from AR training dynamics (§3.3, "we inherit the shift
operation from AR models"). Given how large this ablation is, the paper does not examine whether
the shift — an AR-native mechanism — introduces its own adaptation artifacts at scale, the same
class of question it raises for annealing but does not raise for shift.

**A3 — The mask-token confound.** Appendix B.3: DiffuGPT-S reuses vocabulary token 10541,
DiffuLLaMA reuses token 811, but DiffuGPT-M gets *"a new [MASK] token"* at tokenid=50257 (GPT2's
vocab boundary — i.e., an actually-appended token, not a repurposed word). Table 1 (§4.3, referenced
from §4.4's analysis) shows DiffuGPT-M is the best-performing model of the three at its scale. The
one model with the theoretically "correct" mask-token treatment is also the best performer, and this
is never disentangled from model size — the paper does not report a same-size ablation isolating the
mask-token treatment.

**A4 — No compute-parity accounting for diffusion's lack of caching.** §4.5's inference-speed
discussion states diffusion models require *"self-attention over the entire sequence at each
iteration"* with no KV-cache. The paper reports wall-clock decoding-latency comparisons (Fig. 4) but
never reports FLOPs-per-token or forward-pass counts. A wall-clock win at T=256 says nothing about
whether that same time budget spent on more AR sampling (e.g., speculative decoding, batching)
would close the gap — the comparison is against a single unaccelerated AR baseline, not the
strongest available AR inference setup.

**A5 — High answer uncertainty is reported, not resolved.** §4.4, Table 2: hit@3 substantially
exceeds single-answer accuracy across every task (e.g., SATMath: 23.6 few-shot vs. 57.7 hit@3). The
paper's own diagnosis: *"the current model exhibits high uncertainty about its responses."* No
follow-up experiment (better decoding, confidence calibration, or verifier-based selection) is run
to close this gap — it is reported as an observation, not investigated as a limitation with a
mitigation attempt.

**A6 — Chain-of-thought prompting *hurts* the model, and the fix is explicitly deferred.**
Same section: adding step-wise CoT solutions *"leads to a drop in performance, likely due to the
absence of instruction tuning."* This means one of the most standard techniques for eliciting
reasoning in AR LLMs actively degrades DiffuLLaMA, and the diagnosed fix (instruction tuning) is the
same capability explicitly deferred to future work (§4.4, "We will leave instruction tuning as the
future work") — so the paper identifies its own fix and does not attempt it.

**A7 — "Code generation" is claimed in the abstract but only infilling is evaluated.** The
abstract/intro (lines 111, 721) claim code-generation capability among DiffuLLaMA's strengths. The
only code-related evaluation in the paper is HumanEval *single-line infilling* (App. B, Table 8) —
fill-in-the-middle given both prefix and suffix, not open-ended function generation from a
specification (the setting "code generation" ordinarily refers to, e.g. HumanEval pass@1 from
scratch). The claimed capability and the evaluated task are not the same thing.

**A8 — Validation is confined to two AR model families at up to 7B.** Every adaptation result in
the paper is GPT2-derived (DiffuGPT) or LLaMA2-derived (DiffuLLaMA), maxing out at 6.74B. The paper's
own framing (§5 Related Work) is that "the adaptation of diffusion models from AR LLMs remains
unexplored" prior to this work — but having opened that direction, it validates the recipe on only
one architecture family at the largest scale tested, leaving open whether the recipe transfers to
architecturally different AR backbones (e.g., non-LLaMA attention variants, mixture-of-experts
models) or scales further.
