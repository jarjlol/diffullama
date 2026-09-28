# Extractor Agent — LLM refinement pass over regex candidates

Per BAGELS Sec 3.2 step 2: filter the 14 regex-keyword hits in `extractor_candidates.txt` down to
genuine self-admitted limitations, discarding noise. Rule followed: select and lightly clean up
verbatim spans only — no paraphrasing, altering, or inventing content. Each verdict states *why*.

## KEPT (5) — genuine author-acknowledged limitations

**E1** [L248, §3.1] *"During training, we do not compute integral loss in Eq.6 for efficiency
consideration; instead, we sample t for each data point."* — An approximation adopted for
efficiency, with no ablation reported quantifying the approximation's effect on the trained model.

**E2** [L411, §4.1] *"there is still scope for training more, since the model does not show signs
of saturation."* — Self-acknowledged undertraining; all reported DiffuGPT results are a lower bound
on what the training recipe could achieve, not a converged result.

**E3** [L511, §4.3] *"we do not provide the suffix information to the model, which might result in
an unfair comparison"* [for infilling evaluation]. — Explicit, author-flagged fairness caveat on
one of the paper's own headline comparisons (infilling capability vs. AR baselines).

**E4** [L604, §4.4] *"We will leave instruction tuning as the future work"* — despite citing
evidence (Ye et al., 2023) that diffusion LMs specifically benefit from it, and despite the same
paragraph reporting that chain-of-thought prompting *degrades* DiffuLLaMA's accuracy, attributed to
"the absence of instruction tuning." No instruction-tuned DiffuLLaMA variant is evaluated anywhere
in the paper.

**E5** [L1429, App. B.3] *"In theory, we should expand the original vocabulary... However,
considering practical issues on implementation, we can alternatively select an existing word from
the vocabulary to serve as the [MASK] token."* — Explicit acknowledgment that the "correct" approach
was not used, for implementation-convenience reasons.

## DISCARDED (9) — false positives from the keyword scan

| Line | Keyword hit | Why discarded |
|---|---|---|
| L22 (abstract) | "limitation" | Describes *prior* DLMs' limitations, motivating this paper's contribution — not self-referential |
| L50 (Intro) | "limitation" | About AR models in general, motivating the shift to diffusion |
| L61 (Intro) | "limitation" | Same — AR limitations DLMs address, not DiffuLLaMA's own |
| L103 (Intro) | "limitation" | Describes what the paper *does* (broaden eval beyond perplexity), framed as contribution |
| L417/418 (§4.2) | "does not" | Critiques *prior work's* perplexity-only evaluation, motivating this paper's broader eval suite |
| L655 (§4.5) | "cannot" | About AR models' memory-bound decoding — framed as diffusion's *advantage*. (But see Analyzer A4 below — the sentence's second half, about diffusion's own lack of caching, is a genuine implicit limitation the keyword scan didn't isolate cleanly enough to extract as its own span) |
| L698 (§5) | "remains unexplored" | The paper's own framing of the gap *it claims to fill* — a contribution claim, not a limitation |
| L912, L969 (References) | "cannot", "limitation" | Bibliography entries (title text), not paper content — confirms the regex scan has no section-boundary awareness and will hit reference lists |
| L1376 (App. B) | "do not" | A deliberate, reasonable design choice (match pre-training distribution) rather than an admitted shortcoming |

## What the regex scan missed entirely (motivates the Analyzer agent)

**Cross-checked directly against §4.5 Discussions (lines 608-655) independently of the keyword list:**
the paper's own ablation table (Table 3) reports the attention-mask-annealing gain as **+2.1** (GPT2-S,
43.3→45.4) and **+2.5** (GPT2-M, 47.2→49.7) — then states *"The mask annealing has minimal impact, so
we choose to omit it for 7B adaptation to simplify implementation using flash-attention 2."* None of
this sentence contains a KEYWORDS hit ("minimal impact... simplify" has no limitation-adjacent term),
so the regex Extractor cannot find it — a genuine, documented failure mode of keyword-based extraction,
not a hypothetical one. This is exactly why BAGELS/LimitGen/the Multi-Agent paper all pair extraction
with a separate reasoning-based Analyzer step rather than relying on regex alone.
