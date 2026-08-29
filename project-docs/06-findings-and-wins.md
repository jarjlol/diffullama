# Novel findings and best moments

What this project has actually discovered. Several of these are publishable-grade observations that came
out of verification work rather than experiments — which is itself notable for an AI-augmented research
project.

---

## 🏆 Top three

### W-1 — Aryan caught a paper that would have caused a desk rejection
**The single highest-value contribution to the project so far.**

The Structure-Guided ReMasking design doc claimed to be first to use program structure for selecting
remasking positions. **CDC** ([arXiv:2605.16829](https://arxiv.org/abs/2605.16829), 16 May 2026) does
exactly that — its own words: *"the localization set anchored on a witness node rather than on a heuristic
score."*

Three claims were deleted before they reached a reviewer. And CDC's Fig. 8(b) turned out to contain a
design correction the team would otherwise have paid for in wasted implementation (see W-4).

**Why it counts:** this is precisely the task described in the team's own Assignment Part 2 —
"Continuous Gap Liveness Verification." The project demonstrated its own proposed contribution on itself.

### W-2 — The anchor paper's annealing justification runs against its own data
DiffuLLaMA omitted attention mask annealing at 7B, justifying it as *"minimal impact"* — based on
ablations at 124M and 355M only.

| | GPT2-S (124M) | GPT2-M (355M) |
|---|---|---|
| annealing gain | **+2.1** | **+2.5** |

**The gain grows with scale.** The authors extrapolate "minimal" ~20× beyond their largest test, in the
direction the trend argues against.

Sharpened further: [2512.06776](https://arxiv.org/abs/2512.06776) tested annealing at 7B — but for
**block**-diffusion, a different adaptation target — and found it *hurts*. So:

| Scale | Target | Effect |
|---|---|---|
| 124M / 355M | full-attention | **helps** (+2.1 / +2.5) |
| 7B | block-diffusion | **hurts** (−4.6) |
| **7B** | **full-attention** | **never tested by anyone** |

Two confounded variables — scale and adaptation target — and DiffuLLaMA sits exactly in the untested cell.
**This is the basis of the recommended project direction.**

### W-3 — The off-by-one was proven, not argued
`audit/test_shift_offset.py` establishes `canvas_position = returned_index + 1` exactly, in pure stdlib
Python with no GPU, model, or torch dependency — and demonstrates the failure concretely: targeting the
token at `returned[2]` masks `'b'` when `'c'` was intended.

This is the test the design doc says must exist *before* the selector. It now does, and it passes.

---

## 🔬 Technical findings

### W-4 — CDC's scope ablation is a *middle* optimum, on a *security* benchmark
| Scope | func-sec@1 |
|---|---|
| **Parent+Leaf** | **34.3** |
| Use–Def Slice | 26.9 |
| Token-Window | 24.1 |

Two things nobody had extracted from this:
1. It is an **interior optimum** — the deployed neighbourhood beats both broader *and* tighter
   alternatives. Not "tighter is better."
2. The metric is on **CWEval, a security benchmark**, and CDC routes *functional* correctness through an
   entirely different mechanism. **Whether the optimum transfers to functional bugs is unmeasured** — and
   that is a genuine open research question, not a reframing of a preempted one.

### W-5 — Targeted remasking saves zero compute
`model.py:139` forwards the entire sequence every denoising step regardless of how many positions are
masked. Repairing 5 tokens costs exactly what regenerating 500 costs at equal step count.

**Consequence:** any repair method's only possible win is *quality*, never efficiency — which makes
best-of-*N* resampling a far harsher baseline than the design doc rated it. Unless steps-to-convergence
differs, which **nobody has measured** — and that is an unclaimed efficiency argument sitting in plain sight.

### W-6 — The confidence baseline would have been unfairly weak
`model.py:116` computes `x0_scores` as the log-prob of the **sampled** token — sampling luck, not model
certainty. The max-based form sits commented out at line 113.

A confidence baseline built on it would be weaker than the published methods it represents, **biasing
results in favour of the proposed method**. Caught before any experiment ran.

### W-7 — Dream ships random-order decoding *and* greedy sampling
`Dream-org/Dream-v0-Instruct-7B`'s config: `"alg": "origin"` (random branch), `"temperature": 0.0`.

Two silent-failure modes in one file: any Dream baseline run without explicitly setting `alg="entropy"`
quietly runs random-order; and at temperature 0, a remasked span *deterministically reproduces itself*
unless surrounding context changed — meaning a repair loop can silently no-op.

### W-8 — The mask-token confound nobody disentangled
DiffuLLaMA reuses vocabulary token 811 as `[MASK]`; DiffuGPT-S reuses 10541; but DiffuGPT-M gets a
**proper new token** — and is also the **better performer**. Confounded with model size, never ablated,
and the "low frequency" justification is asserted without measurement.

### W-9 — The inference attention mask is a no-op that blocks flash attention
With `attn_mask_ratio=1.0`, `get_anneal_attn_mask` returns an all-zeros additive mask — mathematically
identical to no mask at all — but as a dense 4-D float tensor that forces the eager attention path.
Free speedup available for a few lines of change.

### W-10 — The generation-order literature has an adapted-model shaped hole
Order-statistics measurement and causal commitment interventions are both already published, at scale
([2601.15593](https://arxiv.org/abs/2601.15593) across 58 benchmarks and 8 MDLMs up to 100B;
[2608.05687](https://arxiv.org/abs/2608.05687) on LLaDA/Dream).

**But every one of them studies from-scratch models. None studies an *adapted* one.** Does adaptation leave
any-order capability vestigial? That question is open and is uniquely licensed by this project's anchor.

---

## 🛠 Engineering wins

### W-11 — Literature-review pipeline improved on its own reported limitation
The QUAL-SG implementation's `REPORT.md` §8 named its keyword-regex relevance scorer as the single biggest
weakness. Replacing it with real semantic judgment:

| | Original | LLM-judged |
|---|---|---|
| Matched / ground truth | 9 / 33 | **11 / 33** |
| F1 | 0.222 | **0.272** |
| Off-topic papers in top-48 | several | **zero** |

Kept in separate `*_llm_judged` files rather than overwriting, because the generated survey cites by
position — swapping the reference set would desync every citation marker. The non-monotonic caveat
(DiffusionBERT dropped out despite scoring 5/5) is documented rather than hidden.

### W-14 — Both provided SOTA papers implemented, and the evaluation caught two measurement bugs
The instructor offered two papers; both are now implemented. QUAL-SG covers generation;
[2412.13612](https://arxiv.org/abs/2412.13612)'s three-task framework covers evaluation
(`litreview/scripts/eval_framework_2412.py`, REPORT.md §11).

Results: reference accuracy **1.000**, hallucination rate **0.000** over 47 scored references (40 via
OpenAlex, 7 rescued via an arXiv fallback).

**The informative part is the Task 2 / Task 3 split**, which validates separating the two:

| | Introduction (Task 2) | Full survey (Task 3) |
|---|---|---|
| ROUGE-1 | **0.357** | 0.187 |
| Lexical coverage | 0.304 | **0.705** |

The Introduction reads most like the anchor's §5 because it is length-comparable; the full survey covers
far more of §5's content because it is ~5× longer. Neither number alone characterises the survey.

**And building it surfaced two measurement bugs** (M-19, M-20), each of which would have produced a
confident wrong number — one of them a fake 100% hallucination rate.

**Honest limit:** Task 1 is close to tautological here. Our references are *retrieved*, not LLM-generated,
and are verified against the same databases they came from. Recorded as such in REPORT.md §11.5 rather than
presented as a result.

### W-12 — Six ground-truth papers recovered from a recall failure
D3PM, SEDD, Argmax Flows and three others were **never retrieved at all** — not mis-ranked, never fetched.
Diagnosed by inspecting the raw candidate pool rather than blaming the ranker, then fixed with queries
matching their actual phrasing.

### W-13 — A stratification described in docs but absent from code
`REPORT.md` claimed `merge_finalize.py` stratified foundational vs recent papers before ranking. It did
not. Running it as-committed reproduced the exact recency-dilution failure the report had already
diagnosed once — including a paper on *tire architecture design* out-ranking Mamba.

---

## 📋 Methodological notes worth keeping

- **Verification found more than experiments would have.** Nearly every finding above came from reading
  code and papers carefully, not from running models. For a project whose theme is AI-augmented research,
  that is itself a result.
- **Three separate off-by-N bugs** appeared in this project (`05-mistakes-and-bugs.md` §B). Assert lengths
  on every hand-built mapping.
- **Recorded disagreements are more useful than resolved ones.** Where an automated pass reached a
  conclusion that was wrong (STaRR "falsifying" the taxonomy), the disagreement is written down rather
  than silently overridden — so a future reader can check who was right.
