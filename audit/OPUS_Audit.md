# OPUS Audit — Structure-Guided ReMasking design doc

> **STATUS: SCRATCH / THROWAWAY.** This branch (`audit/opus-review`) and this `audit/` folder exist to
> hold verification work and open questions while the project direction is being settled. **Delete the
> branch once decisions are made** — nothing here is meant to ship. Nothing in `main` depends on it.

**Audited:** Neel's `Structure-Guided ReMasking for Masked Diffusion Language Models` design doc,
against this repository's actual code, the published literature, and Aryan's CDC finding.
**Date:** 2026-08-28.
**Method:** direct code reading (not summary-trusting) + two independent literature-verification passes,
every cited paper resolved to a working link before inclusion.

---

## 0. Summary

The design doc is unusually well-grounded — **all seven code references in Appendix A are correct**, and
the two "Critical" risks it self-identifies (slice blowup, off-by-one shift) are the right two to have
flagged. What follows is what it got wrong or missed, in descending order of consequence.

Three findings change design decisions:

1. **§4.1's sampler claim is factually wrong in both directions** — and creates a silent-failure mode.
2. **Aryan is right about CDC.** The core mechanism is preempted by a May 2026 preprint. Verified.
3. **The confidence baseline as specified would be unfairly weak**, biasing results toward the proposed
   method in a way a reviewer would catch immediately.

---

## 1. Code reference verification — all correct

Every `Appendix A` reference checked against the actual files at this commit:

| Doc reference | Claim | Verdict |
|---|---|---|
| `model.py:80` | `generate_samples()` | ✅ correct |
| `model.py:102` | `maskable_mask = ~src_mask` | ✅ correct (`init_maskable_mask = maskable_mask = ~src_mask`) |
| `model.py:120` | shift: `x0 = cat([x[:,0:1], x0[:,:-1]])` | ✅ correct |
| `model.py:156` | final `x0 = x0[:,1:]` | ✅ correct |
| `model.py:133` | random unmasking order | ✅ correct (`rand_like(x0) < p_to_x0`) |
| `model.py:143` | `x0_scores` computed but unused | ✅ correct — assigned at 116/143, shifted at 121/148, **never read, never returned** |
| `inf_diffullama.py:65` | worked `src_mask` example | ✅ correct |

§5.2's central claim also holds: `src_mask` is a genuine freeze/remask primitive and does permit
arbitrary non-contiguous remasking without touching the sampler. The architectural premise is sound.

### 1.1 The shift, stated concretely

§5.3 says "∓1 for the shift", which is directionally ambiguous. The exact relation, traced through the code:

- `model.py:110` — `logits[i]` predicts position `i+1` (inherited AR shift)
- `model.py:120` — right-shift realigns `x0` to canvas coordinates
- `model.py:156` — drops canvas position 0 from the returned sequence

Therefore: **`canvas_position = returned_index + 1`**

Write that literal formula into §5.3. It is the thing the invariant test must assert.

---

## 2. Errors and omissions found in code review

### 2.1 The confidence baseline would be unfairly weak — **fairness problem**

`model.py:116` computes `x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1))` — the log-prob of the
**sampled** token. Line 113 shows the max-based form (`scores.max(-1)`) sitting commented out.

So `x0_scores` measures *how lucky this particular sample was*, not *how certain the model is*. A
low-confidence baseline built on it conflates model uncertainty with sampling noise, making the baseline
weaker than the published methods it is supposed to represent — and biasing the comparison **in favour of
the proposed method**. This is exactly the kind of thing a reviewer rejects on.

**Fix:** use max softmax probability (or margin/entropy) for the confidence baseline, not the
sampled-token log-prob.

### 2.2 Targeted remasking saves **zero** compute

`model.py:139` forwards the entire sequence at every denoising step, regardless of how many positions are
masked. No KV cache, no sparsity benefit. Repairing 5 tokens costs exactly what regenerating 500 costs at
equal step count.

Consequences:
- Risk 6 (best-of-*N* beats targeted repair) is **worse than rated**. At equal step budget, *R* rounds of
  repair costs the same as *R* entirely fresh drafts.
- The only available win is **quality**, never efficiency — unless steps-to-convergence differs (see §5.3).

### 2.3 The oracle baseline is probably ill-defined

§6.1 masks "the gold-diff region." But HumanEval/MBPP ship *canonical solutions*, not gold diffs against
this model's draft. A 7B model's failed attempt and the canonical solution are frequently different
algorithms, so their diff approaches 100% of the function and the oracle degenerates to "regenerate
everything" — bounding nothing. See §5.2 for a replacement.

### 2.4 `x0_scores` is never returned

`model.py:158` returns only `x0`; no per-step history is recorded anywhere. §5.2's "the DiffuLLaMA adapter
is consequently thin" holds for masking but **not** for the confidence or instability baselines, both of
which require modifying `generate_samples`.

### 2.5 Statistical power is overstated

§6.3 claims ~4–5 pp resolvable on ~560 paired problems. Two corrections:
- The analysis unit is **failed drafts** (~280 if pass rate ≈ 50%), not 560 problems.
- 9 methods ⇒ **36 pairwise comparisons**. Holm across that family pushes the effective threshold to
  ≈0.0014 for the smallest *p*.

**Fix:** pre-register 3–4 primary comparisons (ours vs low-confidence, vs local-window, vs best-of-*N*);
declare the remaining 32 exploratory.

### 2.6 Compute estimate is light by roughly 2–4×

`inf_diffullama.py:24` defaults `_attn_implementation="eager"`, and `get_anneal_attn_mask` supplies a dense
4-D float mask which blocks flash attention. At T=32–64 full-sequence forwards per repair, ~0.4 s/repair is
optimistic; ~1–1.5 s is more realistic → grid nearer **30–80 GPU-hours** than 10–20. Still feasible, but the
stated headroom isn't there. §7 also asserts a 96 GB system — **unconfirmed, verify before planning on it.**

### 2.7 Missing failure mode: remasking may regenerate identical tokens

Nothing in the doc addresses the case where a remasked span is refilled with exactly the same code.
Sampling at `logits_temp=0.9`/`topp=0.9` gives some variation, but confident spans will often reproduce
themselves and the repair loop silently no-ops. "Tokens changed" is listed as a *cost* metric — it should be
a **Phase 0 diagnostic**, because a near-zero change rate kills the project without any visible error.

### 2.8 Offset mapping over indentation-sensitive code

§5.3 flags the shift correctly, but a second hazard stacks on it: LLaMA's SentencePiece tokenizer uses `▁`
prefix markers and glues leading whitespace to the first token of a line. Python indentation is *semantic*.
The invariant test must cover indented multi-line statements specifically, not just expression spans.

### 2.9 Unaddressed practical gaps

- **Canvas construction for code.** Generation is fixed-length; nothing specifies how `gen_len` is chosen,
  or how the function end is handled (too short truncates, too long trails garbage).
- **Body extraction / post-processing.** Standard for HumanEval, unmentioned — and it gates AST parsing.
- **Parse rate.** AST parsing is a hard prerequisite for the entire structural method. Add it to Phase 0;
  it's one line to measure and it binds before any other criterion.
- **`R` (rounds) is never given a numeric value** in the compute table.

---

## 3. Novelty: Aryan's CDC finding is correct

**Verified:** [arXiv:2605.16829](https://arxiv.org/abs/2605.16829) — *Constrained Code Generation with
Discrete Diffusion* (Shao, Cardei, Xie, Fioretto, Wang), 16 May 2026. **v1 preprint, unrefereed, no venue.**
Full text read; Aryan's description was accurate in every particular.

### 3.1 What is genuinely preempted

CDC's **MDFI** operator: builds a partial Code Property Graph (AST+CFG+DFG) mid-denoising → identifies an
offending node → takes its AST/dataflow neighbourhood → lifts to token spans → applies a budget cap →
remasks → continues denoising. The paper's own words:

> *"This is the same form as the standard partial-mask state ... with the localization set anchored on a
> witness node rather than on a heuristic score."*

That sentence is §1's thesis, already published. Also gone: **"structure beats fixed-window/random"** as a
novel *finding* — their Fig. 8(b) reports it, and a GradGuide ablation shows random editing ≈ vanilla. Same
backbone family too (Dream-Coder, DiffuCoder).

**Three claims must be deleted outright:**
1. first to use program structure to select remasking positions
2. first non-contiguous structure-derived remask set
3. structure-beats-window/random as a novel result

### 3.2 What is *not* preempted

1. **CDC has no confidence-based remasking baseline at all.** Its baselines are vanilla denoising,
   grammar-constrained diffusion, security prompting, AR re-prompting. No low-confidence, no entropy, no
   instability, nothing at matched budget *k*. The unified-scorer grid is an experiment that does not exist.
2. **Purely static — no execution feedback.** MDFI is seeded by a static security witness. Nothing anywhere
   seeds a structural remask from a failing test, traceback, or assertion. Searched hard; found nothing.
3. **Constrained generation, not repair.** CDC intervenes mid-trajectory on one incomplete program. No
   iterate-until-pass loop over already-failed programs, and **zero repair-benchmark numbers** — no
   HumanEvalFix, DebugBench, QuixBugs, Defects4J.
4. **Security vs functionality split.** CDC routes *security* through the graph and *functional* correctness
   through a learned surrogate with entropy/confidence saliency. Structure for semantic bugs is the gap its
   own architecture leaves open — and its +8.5 pp super-additivity result hints the split isn't principled.
5. It never cites the remasking-policy literature and does not present itself as a remasking policy.

### 3.3 The finding that should change the design

CDC Fig. 8(b) ablates neighbourhood scope on CWEval:

| Scope | Score |
|---|---|
| Parent+Leaf AST | **34.3** |
| Use–Def Slice | 26.9 |
| Token-Window | 24.1 |

**A tight AST neighbourhood beats the broader use-def slice by 7.4 pp.** §3.2 is built on backward slicing.
There is now published evidence that aggressive slicing actively *hurts* — which is the doc's own Risk 1,
confirmed empirically by someone else before a line of code has been written.

Treat as a design correction, not merely a novelty problem. Phase 0's slice-size measurement is more
load-bearing than written, and the §9 pivot to the hybrid is more likely than the doc implies.

### 3.4 Near-misses — cite, don't fear

| Paper | Link | Relation |
|---|---|---|
| AnCoder | [2602.17688](https://arxiv.org/abs/2602.17688) | AST guides unmask **order**, not remasking; no repair |
| TreeDiff | [2508.01473](https://arxiv.org/abs/2508.01473) | AST corruption at **training** time only |
| CodeDiffuSe | [Springer](https://link.springer.com/content/pdf/10.1007/s44443-025-00237-6.pdf) | Repair via **oracle** localization of a known buggy span; inference remask is entropy+parser, not dependency graph. Lower-rigor venue — some numbers look anomalous |
| DAPD | [2603.12996](https://arxiv.org/abs/2603.12996) | "Dependency" = self-attention MRF, **not** external program structure |
| Remask, Don't Replace | [2604.18738](https://arxiv.org/abs/2604.18738) | Purely probabilistic — **supports** the taxonomy |
| STaRR | [2601.04205](https://arxiv.org/abs/2601.04205) | Temporal variance + spatial deviance of confidence — the instability baseline |
| Re-evaluating Confidence Remasking | [2606.12232](https://arxiv.org/abs/2606.12232) | Finds WINO ≈ no benefit over plain confidence unmasking — strong motivating citation |

---

## 4. Fact-check: backbones, benchmarks, baselines

### 4.1 **§4.1's sampler claim is wrong in both directions** — highest-priority correction

**LLaDA unmasks *highest*-confidence-first, not lowest.** `generate.py` does
`torch.topk(confidence, k=num_transfer_tokens)`. The config name `remasking='low_confidence'` refers to which
tokens *stay masked*. The doc read the name, not the code. (Default is also semi-autoregressive,
`block_length=32`.)

**Dream and Dream-Coder do not default to confidence ordering at all.**
`Dream-org/Dream-v0-Instruct-7B`'s `generation_config.json` ships `"alg": "origin"` — the **random** branch
(`torch.rand(*x0.shape) < p_transfer`). `Dream-Coder-v0-Instruct-7B` has **no `alg` key**, falling back to the
same random default. `apple/DiffuCoder-7B-Instruct` likewise. Confidence variants (`entropy`,
`maskgit_plus`, `topk_margin`) are strictly **opt-in**; the READMEs pass `alg="entropy"` explicitly.

Two consequences:

1. **The motivation collapses.** "DiffuLLaMA is random-order while modern dLLMs are confidence-ordered" is
   true only for LLaDA. Dream is *also* random by default, so DiffuLLaMA is not the outlier.
2. **New silent-failure risk — same class as the off-by-one.** Any Dream / Dream-Coder baseline run without
   explicitly setting `alg="entropy"` **silently runs random-order**, executes cleanly, produces plausible
   numbers, and is not the model whose published results are being compared against.
   → Add to §10 beside Risk 2. The adapter must **assert `alg` is explicitly set** and fail loudly otherwise.

Combined with §2.1 and §2.4 above, §4.1's "DiffuLLaMA already computes the needed scores, so this is a small
change" is wrong on both counts and the whole section needs rewriting.

### 4.2 Numbers to correct

| Doc | Actual |
|---|---|
| ~560 problems (EvalPlus) | **542** (HumanEval+ 164 + MBPP+ 378). 560 ≈ deprecated pre-v0.2.0 count of 563 — quoting it signals a stale version |
| Dream-7B, ~14 GB | **7.6B, 15.2 GB bf16** (Qwen2.5-7B backbone). On a 16 GB card that is the difference between fitting and not |
| Dream HF IDs | Doc's IDs **do not exist**. Real: `Dream-org/Dream-v0-Instruct-7B`, `Dream-org/Dream-Coder-v0-Instruct-7B` |
| DiffuGPT-small 127M | Hub artifact `diffusionfamily/diffugpt-s` is **124M** (127M is the paper's figure) |
| DiffuLLaMA-7B ~14 GB | 6.74B, **13.5 GB** — fine, slightly conservative |
| MBPP ~500 | Correct **only** for the test split (task_ids 11–510). Full set 974, sanitized 427 — state which |

Verified as written: LLaDA-8B-Instruct (8.0B, 16.03 GB), HumanEval = 164, MBPP's 3-asserts-per-problem
(distribution is `{3: 974}`, exact), and EvalPlus's `base_input`/`plus_input` split supporting the
visible/hidden design.

**Caveat on the visible/hidden split:** EvalPlus stores test *inputs* evaluated differentially against
`canonical_solution`, **not** literal assert strings. §6's harness must render them into asserts itself.

Also relevant: **DiffuCoder** is Apple's ([arXiv:2506.20639](https://arxiv.org/abs/2506.20639)), same first
author as DiffuLLaMA, same architecture family as Dream. Natural fourth backbone.

### 4.3 Taxonomy holds — one correction to a verification pass

An automated check claimed STaRR "falsifies" §1's claim that existing policies use only probabilistic or
spatial signals. **That reading is wrong and the distinction matters.** STaRR's *temporal variance* and
*spatial deviance* are both computed from token-confidence dynamics — internal model signals, not external
program structure. STaRR is an **instance of** the probabilistic/spatial class, not a counterexample.

§1's taxonomy stands. What changes: the "temporal instability" baseline row has **canonical published
forms** that must be cited and implemented faithfully rather than invented —
**STaRR** ([2601.04205](https://arxiv.org/abs/2601.04205)) and
**RCR / Running Confidence Remasking** from MDPO ([2508.13148](https://arxiv.org/abs/2508.13148)).
Implementing a strawman version of a baseline that has a real published form is the same fairness failure as
§2.1.

---

## 5. Ideas worth adopting

### 5.1 Turn CDC from competitor into baseline

CDC's static-neighbourhood selection slots directly into the unified scorer as another (α,β,γ,δ,ε) setting.
The question becomes *"does execution-grounded dynamic dependency information beat static structure?"* — which
CDC cannot answer about itself and needs this harness to answer. Its existence becomes evidence the direction
matters rather than evidence of being scooped.

### 5.2 Synthetic-corruption testbed — fixes the oracle, decouples two confounds

Take canonical HumanEval solutions, inject known single-edit bugs (flipped comparator, wrong variable,
off-by-one). This yields **ground-truth wrong tokens**, and therefore:

- a well-defined oracle (replacing the broken one in §2.3)
- direct selector **precision/recall**, independent of whether the backbone can repair anything
- clean separation of *"my selector is bad"* from *"this model can't fix anything"* — the exact confound §9's
  oracle criterion tries to resolve

Cheap, runs against the `mock.py` adapter, buildable before any GPU access.

### 5.3 Steps-to-convergence as the efficiency claim

Since masking fewer tokens does not cut per-step cost (§2.2), test whether it cuts **step count** — does a
20-token repair converge in 8 steps where full regeneration needs 32? If so, it recovers a compute argument
against best-of-*N* that the current design has no way to make. Nothing in the doc measures this.

### 5.4 Drop the dense attention mask

With `attn_mask_ratio=1.0`, `get_anneal_attn_mask` returns an **all-zeros** additive mask — mathematically
identical to no mask at all (verify: `bernoulli(1.0)` → all ones → `logical_or` → all ones → `1.0-1.0 = 0.0`).
Passing `None` with flash attention instead should give a solid speedup at seq 512, more at longer lengths,
for a few lines of change.

### 5.5 Reframe that survives CDC

In descending order of survivability:

1. **Execution-grounded seeding.** State it in §1: *"CDC establishes that static-analysis witnesses can anchor
   structural remasking; we ask whether dynamic failure evidence — traceback, assertion line, first divergent
   variable — anchors it better."*
2. **The policy study.** First controlled comparison of remasking-position policies at matched budget:
   {random, window, low-confidence, instability, AST-neighbourhood, def-use slice} × *k* ∈ {8,16,32,64}. CDC
   fills one cell on one benchmark; cite it as an instance, not as the study.
3. **Functional bugs, not security** — the split CDC's own super-additivity result suggests isn't principled.
4. **Radius as the object of study.** Turn Fig. 8(b) into the question: how far along the dependency graph
   should a remask set extend, and does the optimum move when localization is dynamic rather than static?

---

## 6. Open items

- [ ] **Confirm the 96 GB system actually exists** and whether it is one card or several (§7 shards trivially
      if several, but the batch-32 assumption changes).
- [ ] Numeric value for `R` (repair rounds).
- [ ] Decide 2 vs 3 backbones after the §4.1 rewrite — the sampler-normalization cost changed.
- [ ] **Scope decision pending** — see the honest scope assessment appended to this document.

---

# Part 2 — Scope assessment and cross-reference

Added after Part 1. Cross-references Neel's design doc, Aryan's CDC finding, and Aalhad's six proposed
research bets. The question this part answers, asked directly: **is the post-CDC reframe of
Structure-Guided ReMasking real, or is it cope?**

## 7. Coverage caveat — **now resolved, see Part 3**

*Original note: two verification passes covering bets 1, 2, 5, 6 died mid-run on an API spend limit.*

**Resolved.** The searches were re-run and all six bets are now verified. **Part 3 supersedes the
provisional assessments in this Part for bets 1, 2, 3, 5, and 6.** Bet 4's assessment below stands
unchanged and was fully verified from the outset.

| Bet | Status | Verdict (see Part 3) |
|---|---|---|
| 1 — multi-GPU single-request decode | ✅ verified | **NOT RECOMMENDED** — premise false |
| 2 — dual-mode serving | ✅ verified | **RISKY** — thesis published 3 months ago |
| 3 — do dLLMs plan? | ✅ verified | **VIABLE only when narrowed** |
| 4 — conversion / dropped training step | ✅ verified | **STRONG** |
| 5 — honest cost accounting | ✅ verified | **NOT RECOMMENDED** — premise decayed |
| 6 — mask-token shortcut | ✅ verified | **VIABLE, pairs with bet 4** |

## 8. Is the Structure-Guided ReMasking reframe cope?

**Honest verdict: partially, yes.** The gap is real, but the effort-to-novelty ratio collapsed when CDC
appeared, and the specific technical approach now has published evidence against it.

### 8.1 The case that it is cope

1. **The delta is one variable on someone else's pipeline.** CDC: static witness → graph neighbourhood →
   token spans → budget → remask. Proposed: dynamic witness → graph neighbourhood → token spans → budget →
   remask. Everything downstream of the seed is identical.
2. **The engineering cost did not shrink with the claim.** To test that one-variable delta, the team must
   still build ~80% of CDC's machinery — code property graph, neighbourhood lifting, budget capping, token
   offset mapping across a shifted tokenizer — plus a sandbox and dynamic tracing CDC didn't need. Maximum
   engineering, minimum claim.
3. **The chosen structural signal has evidence against it.** CDC Fig. 8(b): Parent+Leaf AST **34.3** vs
   Use–Def Slice **26.9**. §3.2 is built on backward slicing. Someone already published that the broader
   slice is *worse*.
4. **The fallback framing needs a surprising result.** "First controlled policy comparison at matched budget"
   is a study paper. Study papers land when the result overturns something. A confirmatory result
   ("structure helps somewhat") is a thin workshop paper.
5. **Effort was already aggressive before CDC.** 3 backbones × 9 methods × 542 problems × 3 seeds, plus
   sandboxing, AST/dataflow, dynamic tracing, offset mapping — for 5 undergraduates in ~14 weeks alongside
   other coursework.

### 8.2 The case that it is genuine

1. **Execution-grounded seeding is verifiably unoccupied.** A dedicated search across arXiv, Semantic
   Scholar, ACL Anthology, Springer and a dLLM survey list found **no** paper seeding a structural remask
   from a failing test, traceback, or assertion.
2. **CDC has no confidence baseline at matched budget.** That comparison genuinely does not exist anywhere.
3. **Repair-after-failure for dLLMs does not exist.** No HumanEvalFix / DebugBench / QuixBugs / Defects4J
   numbers in CDC or elsewhere in this space.
4. **For a course project targeting a workshop, concurrent work is survivable.** "We identified this
   independently, found concurrent work, and isolated the sub-question it left open" is a legitimate and
   well-regarded framing at Agents4Science scale — this is *not* a top-tier venue submission where the
   novelty bar would kill it.

### 8.3 Net

Not pure cope — the gap is real and verified. But it is now **a materially harder project for a materially
smaller claim**, with a known-weak core signal. If the team keeps it, the honest version is:

> *"CDC showed static-analysis witnesses can anchor structural remasking. We ask whether dynamic failure
> evidence anchors it better, and we run the matched-budget policy comparison CDC does not."*

That sentence is defensible. It is also visibly a follow-up, and should be written as one rather than
dressed up as a first.

## 9. Aalhad's six bets

### Bet 4 — "what conversion does to the weights" — **VERIFIED, and stronger than stated**

Aalhad's claim: *"The anchor paper dropped a key training step for engineering convenience and validated it
only at small scale."*

**Confirmed verbatim from the anchor paper.** Two quotes:

> "For efficient implementation we enable flash-attention 2 (Dao, 2024) and directly use bi-directional
> attention **without attention mask annealing**."

> "For DD loss, removing attention mask annealing and shift operations both degrade performance, indicating
> the efficacy of our approaches. **The mask annealing has minimal impact, so we choose to omit it for 7B
> adaptation to simplify implementation using flash-attention 2.**"

**The detail Aalhad did not state, and it is the whole opportunity.** The ablation (Table 3):

| Setting | GPT2-S | GPT2-M |
|---|---|---|
| DD w/o anneal | 43.3 | 47.2 |
| DD (full) | 45.4 | 49.7 |
| **Annealing gain** | **+2.1** | **+2.5** |

The benefit of annealing **grows** from 124M to 355M. The authors call this "minimal impact" and extrapolate
to a model ~20× larger than the largest one they tested it on. The scaling trend runs *against* their own
justification.

Note also a widespread misreading in secondary sources (search-engine summaries and review sites state
"annealing has minimal impact **on the 7B model**"). The paper says no such thing — annealing was **never
tested at 7B**. The "minimal impact" finding is entirely from GPT2-S/M. That conflation is itself worth a
paragraph in a write-up.

**Adjacent work to check before committing** (found, not yet read in full):
- *From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs*,
  [arXiv:2512.06776](https://arxiv.org/abs/2512.06776) — reportedly finds annealed attention mask "not
  performant" for AR→**Block**-Diffusion adaptation. Different target architecture, but closest known work.
  **Read this first.**
- *UNIFUSION*, [arXiv:2607.24507](https://arxiv.org/abs/2607.24507) — AR→discrete-diffusion adaptation under a
  unified reverse-rate objective. Adjacent.

**Why this fits better than SGR for this specific team:**

| | Structure-Guided ReMasking | Bet 4 (annealing audit) |
|---|---|---|
| Course fit | Tangential to anchor paper | **Directly interrogates the anchor paper** — literally "reproduce, then improve" |
| Novelty grounding | Follow-up to a preprint | A quotable, verified inconsistency in the anchor's own text |
| Engineering risk | Very high (CPG, tracing, sandbox, offset mapping) | Low — training runs + evaluation, no exotic infrastructure |
| Silent-failure surface | Two known (off-by-one, `alg="origin"`) | Minimal |
| Result value | Weak if confirmatory | **Interesting either way** — if annealing matters at scale, that's a finding against the anchor; if not, that's a validated scaling law |
| Preemption risk | Already realised (CDC) | Low, and cheap to re-check |

**Verdict: STRONG.** The single best-fitting direction on the table for this team.

**Concrete shape:** train DiffuGPT-scale adaptations with and without annealing at ≥3 sizes (124M, 355M, and
one larger — 774M/1.5B if compute allows), fit the trend in the annealing gain, and test whether the
extrapolation to 7B holds. Everything needed is in this repo already. This also *is* a reproduction of the
anchor paper, satisfying project stage 2 and stage 3 simultaneously.

### Bet 3 — "do diffusion LLMs actually plan?" — **PARTIALLY OCCUPIED**

Not verified in depth (agent died), but at least four papers already work this ground, all surfaced in this
repo's own litreview retrieval pool:

- *On the Reasoning Abilities of Masked Diffusion Language Models* (2025) — characterises what MDMs can
  provably solve, connecting to CoT and padded looped transformers
- *Theoretical Benefit and Limitation of Diffusion Language Model* (2025)
- *Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation* (2026) — a direct
  falsification-shaped result on the any-order claim
- *Do Language Models Plan Ahead for Future Tokens?* ([arXiv:2404.00859](https://arxiv.org/abs/2404.00859))
  — cited by the anchor paper itself

Aalhad's "low effort, publishable either way" is optimistic. The obvious version of this experiment is
substantially done. A course-scale contribution would need a specific causal intervention nobody has run —
which requires the novelty search that did not complete.

**Verdict: RISKY until re-searched.** Good instinct, likely crowded. Do not commit before verifying.

### Bets 1, 2, 5, 6 — **NOT ASSESSED**

Novelty searches did not complete. Reasoning-from-premises only, flagged as unverified:

- **Bet 1 (multi-GPU single-request decode).** One premise concern worth checking: AR models *can* split a
  single request across GPUs (tensor/pipeline parallelism). The real distinction is presumably
  *sequence-dimension* partitioning, which diffusion permits because positions have no sequential dependency.
  As stated, the premise is at minimum imprecise. Also: MLSys/EuroSys framing sits awkwardly against a course
  requiring NLP/CV with GenAI emphasis, and against an AI-for-science workshop target.
- **Bet 2 (dual-mode serving).** "Untested" needs checking against block-diffusion work (BD3-LMs, SDAR,
  CtrlDiff, Sequential Diffusion Language Models) and dLLM KV-cache work (Fast-dLLM, dLLM-Cache, FlashDLM),
  several of which already interpolate AR and parallel decoding.
- **Bets 5, 6.** Correctly self-ranked as supporting work. Bet 6 (mask-token handling) is worth 10 minutes of
  checking — LLaMA-2 has no native mask token, and this repo does
  `resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)`, so there is *something* real there, but
  likely an implementation note rather than a project.

## 10. Cross-reference — how the three teammates' work fits together

- **Neel** built the QUAL-SG litreview pipeline and the SGR design doc. The design doc's code grounding is
  excellent (all 7 references correct); its literature grounding is where the errors are (§4.1 sampler, §4.2
  numbers) — the predictable failure mode when a doc is written from papers' prose rather than their code.
- **Aryan** found CDC. This is the highest-value single contribution to the project so far: it prevents a
  desk-rejectable novelty claim, and its Fig. 8(b) is a design correction the team would otherwise have paid
  for in wasted implementation.
- **Aalhad** proposed six alternatives. Bet 4 is verified and is, on this analysis, the strongest direction
  available. His closing instruction — *"re-verify novelty against arXiv at kickoff: this field publishes ~10
  preprints weekly"* — is exactly right and is precisely what Aryan's CDC find demonstrates.

**These three artefacts are consistent with each other, not competing.** Aryan's finding constrains Neel's
doc; Aalhad's bet 4 offers a lower-risk destination for the same anchor paper. The disagreement between them
is about *how much risk to carry*, not about facts.

Aalhad's own **novelty-verification skill** should be run against bet 4 before kickoff, and re-run at
milestone boundaries — not once. CDC was posted three months before this audit; whatever preempts bet 4 may
not exist yet today.

## 11. Recommendation

Stated plainly, as a judgement rather than a decree — the team invested real work in SGR and this is their
call:

1. **Primary: bet 4 (annealing / conversion audit).** Verified open, directly interrogates the anchor paper,
   low engineering risk, interesting either way, and doubles as the reproduction stage the project requires.
2. **Parallel de-risking: SGR Phase 0 only.** Run the failure taxonomy, slice-size distribution, and parse
   rate — a few days of CPU work, no GPU. If the numbers are strong *and* CDC Fig. 8(b) does not replicate on
   Python/functional bugs, SGR becomes viable again on evidence rather than hope. This is Aalhad's own "use a
   fast parallel result to de-risk the timeline" logic, applied to the direction already invested in.
3. **Before any commitment:** re-run the novelty search on bet 4 (read arXiv:2512.06776 first), and confirm
   the 96 GB system actually exists.

If the team prefers to keep SGR as primary, the honest framing in §8.3 is the one to use, the three claims in
§3.1 must be deleted, and §3.2 should move from backward slicing to a tight AST neighbourhood in light of
Fig. 8(b).

---

*Verification standard used throughout: every paper resolved to a working arXiv/proceedings link before
inclusion; code claims checked by reading the files in this repository at this commit, not by trusting
summaries. Where an automated verification pass reached a conclusion I disagreed with (§4.3), the
disagreement is recorded rather than silently resolved. Where verification did not complete (§7), that is
stated rather than papered over.*

---

# Part 3 — completed novelty audit of all six bets

Supersedes Part 2's provisional assessments for bets 1, 2, 3, 5, 6. Bet 4 (§9) stands unchanged.

## 12. A venue constraint nobody has accounted for

**Agents4Science requires AI to be the primary author** (https://agents4science.stanford.edu/submissions.html).
That is a hard constraint orthogonal to every novelty question in this document, and it does not appear in
Neel's design doc, Aalhad's bets, or the project brief. It also *rewards* the direction this project is
already taking — the litreview pipeline and this audit are both AI-executed research artefacts. **Confirm
this requirement with the instructor before scoping anything**, because it changes what "the deliverable"
means, not just what the topic is.

## 13. Bet 1 — multi-GPU single-request decode — **NOT RECOMMENDED**

### Premise is false as written

*"AR models can't split one request across GPUs"* is incorrect. Tensor parallelism splits every
single-request forward pass across GPUs — that is the standard vLLM/TRT-LLM deployment. Pipeline
parallelism splits by layer. And **decode-time context parallelism for a single request already shipped in
vLLM** (https://vllm.ai/blog/2026-08-07-decode-context-parallelism), partitioning one request's KV cache
along the sequence dimension.

The salvageable version is much narrower: in AR decode one step produces *one* token position, so there is
no sequence-dimension *work* to partition (CP partitions KV history, not new-position computation). A dLLM
step computes logits for many masked positions at once, so those are genuinely partitionable. That is a
claim about **activation width per step**, not about "can't split across GPUs." As written it would be
desk-rejected in the first paragraph by any systems reviewer.

It is also weaker than it sounds — a 7B dLLM denoising 256 positions is already **compute-bound**, so extra
GPUs buy less than the pitch implies.

### Already published

| Work | Link | Preemption |
|---|---|---|
| **dInfer** | [2510.08666](https://arxiv.org/abs/2510.08666) | **Heavy.** TP + expert parallelism for dLLM inference explicitly **at batch size 1** on 8×H800 — the exact regime claimed as the gap. Paper states EP "is effective even at a batch size of 1 — unlike in AR models" |
| **Sangam** | [2607.04206](https://arxiv.org/abs/2607.04206) | **Heavy.** This *is* the scheduler + cost model paper, from an established systems group |
| **Optimus** | [2605.24832](https://arxiv.org/abs/2605.24832) | Saturation-aware runtime selection of decoding granularity |
| **HERALD** | [2606.21633](https://arxiv.org/abs/2606.21633) | Block-diffusion serving, CPU-GPU cooperative KV retrieval (Stoica) |
| **dLLM-Serve** | [2512.17077](https://arxiv.org/abs/2512.17077) | Production dLLM serving system design |
| **DiffusionGemma in vLLM** | [blog](https://vllm.ai/blog/2026-06-10-diffusion-gemma) | dLLMs now natively supported in the mainstream stack — the "nobody serves dLLMs" gap is closed |

Sequence-dimension partitioning of *denoising positions* for a single dLLM request across GPUs was **NOT
VERIFIED** as existing — a thin sliver of genuine novelty. But it sits inside a subfield with four serving
papers in nine months.

**The mechanism already exists anyway.** DeepSpeed-Ulysses and Ring Attention are architecture-agnostic
attention-partitioning schemes, and a dLLM step is exactly the bidirectional full-sequence forward pass they
were designed for. Nothing new to invent; the residual contribution is "cost model + scheduler," which is
precisely what Sangam and Optimus publish.

### Hardware — **corrected 2026-08-29**

Confirmed: **2× RTX 6000 Pro Blackwell (96 GB each, 192 GB total)**, plus access to the **Sharanga cluster
at BITS Hyderabad** (specs unconfirmed).

This **partially retracts** the earlier hardware objection. A 2-GPU experiment is runnable, so the thesis is
no longer unmeasurable. But it does not rescue the direction:

- **2 GPUs cannot support a scaling argument.** dInfer's numbers come from 8×8 H800. A two-point
  measurement (1 GPU vs 2) cannot characterise a scheduler or a cost model, which is the entire claimed
  contribution.
- **The other two objections are untouched by hardware.** The premise is still false as written, and dInfer
  still published TP+EP for dLLMs at batch size 1 with Sangam publishing the scheduler.

If Sharanga turns out to have ≥8 interconnected GPUs, the *feasibility* picture changes again — but novelty
would still be the binding constraint, so confirm Sharanga's specs for the sake of the other directions
rather than to revive this one.

**Verdict: NOT RECOMMENDED** — unchanged, but now on novelty and premise grounds alone rather than hardware.

## 14. Bet 2 — dual-mode serving — **RISKY**

### "Untested" is false — it was published three months ago, by name

**FLARE: Diffusion for Hybrid Language Model**, [arXiv:2606.01774](https://arxiv.org/abs/2606.01774)
(Adobe / Georgia Tech). Abstract, verbatim: *"enabling one checkpoint to support both AR-style verified
decoding and diffusion-style parallel denoising."* Modes are AR-Trust vs Diffusion-Trust, chosen at
inference, from "a single trained checkpoint that only switches its sampling path." It reports throughput
gains over dLLM baselines in single-GPU concurrent serving.

That is Bet 2's thesis statement, near word-for-word.

Supporting cluster, all verified: **BD3-LMs** ([2503.09573](https://arxiv.org/abs/2503.09573), ICLR 2025
**Oral**) interpolates AR↔diffusion via block size with KV caching; **Eso-LMs**
([2506.01928](https://arxiv.org/abs/2506.01928)) adds KV caching to MDMs while preserving parallel
generation; **SDAR** ([2510.06303](https://arxiv.org/abs/2510.06303), ACL Findings 2026) converts a trained
AR model into blockwise diffusion; **CtrlDiff** ([2505.14455](https://arxiv.org/abs/2505.14455)) picks block
size per step via RL; **SDLM** ([2509.24007](https://arxiv.org/abs/2509.24007)) commits adaptive length per
step; **AdaBlock-dLLM** ([2509.26432](https://arxiv.org/abs/2509.26432)) does it training-free. KV-cache
line: **Fast-dLLM** ([2505.22618](https://arxiv.org/abs/2505.22618), NVlabs), **dLLM-Cache**
([2506.06295](https://arxiv.org/abs/2506.06295), ICML 2026), **FlashDLM**
([2505.21467](https://arxiv.org/abs/2505.21467)).

### Block-trained models already give both modes for free

Block size 1 → autoregressive with working KV cache. Block size L → full parallel diffusion. Anything
between is a point on the curve. The genuinely underexplored piece is a **per-request SLO-aware router**,
and that is ~80% covered by Optimus (load-driven) and Sangam. The remaining delta is a load-balancing
heuristic, not a research contribution.

### An anchor-specific blocker

**DiffuLLaMA is a full-attention, non-block model.** It has no native AR mode with a working KV cache. To
get "one checkpoint, two modes" from it you would first have to block-ify it — i.e. re-implement
SDAR/BD3-LM. That is the whole semester spent reproducing two existing papers *before* the novel part
starts.

**Verdict: RISKY.** Feasible on the hardware (its one real advantage), but as pitched it is a reproduction.

## 15. Bet 3 — do dLLMs actually plan? — **VIABLE only when narrowed**

### The claims are real and thin

Verified: **LLaDA** claims it "addresses the reversal curse, surpassing GPT-4o in a reversal poem completion
task" — one task, GPT-4-generated pairs, the single strongest cherry-picked-looking claim in the set.
**Dream 7B** asserts "superior planning abilities" with no benchmark named. **DiffuLLaMA — the anchor —
makes no explicit planning claim at all**, only "filling in the middle without prompt re-ordering."

### But the falsification test is already run, twice, at scale

| Paper | What it already establishes |
|---|---|
| [2601.15593](https://arxiv.org/abs/2601.15593) *Parallelism and Generation Order in MDLMs* | **The order-statistics measurement, done.** Kendall's tau + Average Finalization Parallelism over **58 benchmarks, 8 MDLMs up to 100B**. Concludes MDLMs do not realize arbitrary-order generation; order is near-left-to-right in practice |
| [2608.05687](https://arxiv.org/abs/2608.05687) *Answer First, Reason Later* | **The causal intervention, done.** Manipulates commitment order, delays answer-token commitment, measures GSM8K on LLaDA variants + Dream-7B |
| [2601.13228](https://arxiv.org/abs/2601.13228) *AR Models Rival Diffusion at ANY-ORDER Generation* | AR models with two-stream attention beat diffusion at infilling — removes the "only diffusion can do any-order" premise |
| [2605.29123](https://arxiv.org/abs/2605.29123) *The Confidence Shortcut* | Confidence-based decoding misaligned with logical-flow trajectories |
| [2510.13117](https://arxiv.org/abs/2510.13117) *On the Reasoning Abilities of MDLMs* (ICLR 2026) | Theory; MDMs ≡ padded looped transformers. Note this one **supports** diffusion |
| [2502.09622](https://arxiv.org/abs/2502.09622) *Theoretical Benefit and Limitation* | Efficiency win holds for perplexity, vanishes for sequence error rate |

"Low effort, publishable either way" is **not true** — the null result is already published, at a scale this
team cannot approach.

### What survives, and why it is actually good

**Every one of those papers studies from-scratch models (LLaDA, Dream). None studies DiffuLLaMA — an
AR→diffusion *adapted* model.** The open question the anchor uniquely licenses:

> Does adaptation leave the any-order capability **vestigial**?

Apply 2601.15593's order metrics and 2608.05687's commitment intervention to DiffuLLaMA vs LLaDA vs Dream
with decoding held fixed. If adapted models are measurably more left-to-right than from-scratch ones, that
is a clean, novel, anchor-derived result that also *explains why adaptation is cheap*.

Inference-only, no training, all three models fit in 96 GB at bf16, parallelizes across five people.

**Verdict: VIABLE**, narrowed. The original headline framing would be desk-rejected by anyone who has read
2601.15593.

## 16. Bet 5 — honest cost accounting — **NOT RECOMMENDED**

Premise was true in 2024-25 and is **largely obsolete now**. Wall-clock reporting became standard by 2026:
dInfer ([2510.08666](https://arxiv.org/abs/2510.08666)), Fast-dLLM, and NVIDIA's Nemotron diffusion line all
report measured throughput; NVIDIA explicitly reasons about metric choice. The community had this argument
and moved on.

The strongest version of the argument is already a **theorem**, not a benchmark
([2502.09622](https://arxiv.org/abs/2502.09622): the efficiency win vanishes for sequence error rate).

Joules-per-token for dLLMs specifically is a real gap, but it is a gap because it is low-value, and it
**cannot be measured credibly on a shared cluster** — energy measurement needs exclusive GPU access and
stable power sampling. A shared-node number is one no reviewer will trust. Compounding it: no vLLM batching
support for dLLMs means the AR baseline is unfairly advantaged, and fixing that is an engineering project,
not a research one.

**Verdict: NOT RECOMMENDED.**

## 17. Bet 6 — mask-token shortcut — **VIABLE, pairs with bet 4**

### The claim is half right — documented, but unanalyzed

"Undocumented" is wrong; it is in Appendix B.2. Verbatim:

> "In theory, we should expand the original vocabulary by adding an additional dimension to include a
> special token as [MASK] token. However, considering practical issues on implementation, we can
> alternatively **select an existing word from the vocabulary** to serve as the [MASK] token. It is
> preferable that this chosen word has a particularly low frequency of occurrence in corpus. For DiffuGPT-S
> we use tokenid=10541 and for **DiffuGPT-M we set a new [MASK] token with tokenid=50257**. For DiffuLLaMA,
> we set tokenid=811."

Confirmed in this repo: `DiffuLLaMA-training/train.py:304` defaults `--mask_token 811`;
`LLaMA-Factory/src/llamafactory/model/loader.py:145,152` hardcodes 10541 for DiffuGPT-S and 811 for
DiffuLLaMA.

### Three things nobody checked

1. **No evidence** that token 811 or 10541 are actually low-frequency. Asserted, never measured.
2. Reusing a real token forces the model to disambiguate *"this is a MASK"* from *"this is literally that
   word"* — a plausible source of degradation, never tested.
3. **DiffuGPT-M is both the only model with a proper new mask token and the better performer** (DD row: 49.7
   vs DiffuGPT-S's 45.4). Confounded with model size, never disentangled.

**Verdict: VIABLE** — too small alone, but a clean cheap ablation that plugs directly into bet 4.

## 18. Revised recommendation — one project, three questions

The three surviving directions are the same paper, and all three are **forced through DiffuLLaMA
specifically** rather than bolted onto it:

| | Question | Type | Cost |
|---|---|---|---|
| Bet 4 | Does dropping attention-mask annealing at 7B cost anything? The ablation gain *grows* with scale (+2.1 → +2.5) while the authors extrapolate "minimal" 20× beyond their largest test | adaptation shortcut | small training runs |
| Bet 6 | Does reusing a vocabulary token as `[MASK]` cost anything? The one model that got it "right" is also the best one, confounded with size | adaptation shortcut | cheap ablation |
| Bet 3 (narrowed) | Does adaptation leave any-order capability vestigial vs from-scratch models? | adaptation consequence | inference only |

**Proposed title: "What does AR→diffusion adaptation actually cost?"**

Why this beats every alternative on the table:

- **Course fit is exact.** It *is* reproduce-the-anchor-then-interrogate-it. Stages 2 and 3 in one.
- **No exotic infrastructure.** No sandboxing, no CPG, no NCCL, no multi-node, no offset-mapping hazard.
- **Interesting whichever way it lands.** If the shortcuts cost something, that is a finding against a
  published ICLR paper. If they don't, that is a validated scaling law nobody had.
- **Three independent shots.** If one question dies, the other two still carry a paper — unlike SGR, where
  the whole project rides on one claim that CDC already dented.
- **Preemption risk is low and cheap to re-check**, unlike the dLLM-serving subfield (a dozen 2026 papers)
  or the planning-falsification subfield (two 2026 papers at 100B scale).

### If the team keeps Structure-Guided ReMasking instead

Then: delete the three claims in §3.1, adopt the §8.3 framing, move §3.2 from backward slicing to a tight
AST neighbourhood (CDC Fig. 8b), fix §4.1 per §4.2, and run Phase 0 **first**.

### Also worth noting

The systems audit surfaced an unrelated but genuinely good idea: use DiffuLLaMA's actual distinguishing
capability — infilling without prompt reordering, per the anchor's own abstract — on a **scientific** task
(structured scientific text, molecular/protein sequence infilling, constrained scientific-notation
generation). Single-GPU, course-appropriate, genuinely AI-for-science given the venue, and not crowded.
Unverified for novelty; flagged as a lead, not a recommendation.

## 19. Final ranking

1. **Bets 4 + 6 + 3-narrowed as one adaptation-audit project** — STRONG
2. **Structure-Guided ReMasking, reframed per §8.3** — VIABLE, higher effort, smaller claim, one dented leg
3. **Scientific-infilling pivot** — promising lead, needs a novelty search
4. **Bet 2** — RISKY; would be a reproduction as pitched
5. **Bets 1 and 5** — NOT RECOMMENDED; false/decayed premises, hardware-blocked

---

*Part 3 verification: four independent search passes, every paper resolved to a working link before
inclusion. Two claims were checked directly against this repository's own copy of the anchor paper's full
text (§9 annealing, §17 mask token) rather than against any summary of it.*

---

# Part 4 — invariant test result, and the plan

## 20. Off-by-one test: **written, run, passing**

`audit/test_shift_offset.py`. Pure stdlib — no torch, no transformers, no GPU, no model download,
because the shift is integer index arithmetic and can be verified exactly without any of that.
(`torch`/`transformers` are in fact not installed locally; this test does not need them.)

**Result: all checks pass.** The invariant is now proven rather than argued:

```
canvas_position = returned_index + 1
```

Exact, and tokenizer-independent — it follows from `model.py` L110→L115→L120→L124→L156 alone.

**The bug is demonstrated concretely, not just described.** Targeting the token seen at `returned[2]`:

| | canvas position | token actually masked |
|---|---|---|
| naive mapping (no shift) | `canvas[2]` | `'b'` ← **wrong, silently** |
| correct mapping (+1) | `canvas[3]` | `'c'` ← intended |

The test then runs both through the simulated sampler and confirms the naive version regenerates a token
the user never selected, while the +1 version hits exactly the intended one.

Also covered and passing: `src_mask` freeze semantics (frozen prefix survives unchanged; generation starts
at `canvas[len(prefix)]` = `returned[len(prefix)-1]`), and AST→char-span mapping over all 19 nodes of a
sample program, exact against `ast.get_source_segment`.

**What this does NOT settle — flagged in the test's own output.** Step 5 uses a *stub* tokenizer with clean
offsets. The real LLaMA tokenizer is SentencePiece: it glues leading whitespace into tokens and marks word
starts with U+2581, and **Python indentation is semantic**. The logic is validated; the real tokenizer's
behaviour on indented statements is not. Re-run step 5 with
`AutoTokenizer.from_pretrained('diffusionfamily/diffullama', use_fast=True)` and
`return_offsets_mapping=True` before trusting the chain on real code. Test 1's invariant needs no such
re-check.

## 21. SOTA assignment — what can still be added

The assignment is **substantively complete** (retrieval + generation + overlap comparison against the
anchor's §5, all 48 references verified real, plus the LLM-judge upgrade taking F1 0.222 → 0.272). Two
additions would strengthen it, in priority order:

1. **Implement the second paper the instructor provided.** The email offered two papers; QUAL-SG
   ([2508.17647](https://arxiv.org/abs/2508.17647)) was implemented for *generation*. The other —
   *Large Language Models for Automated Literature Review* ([2412.13612](https://arxiv.org/abs/2412.13612))
   — is an **evaluation** framework covering reference generation, abstract writing, and review
   composition. Using it to evaluate the QUAL-SG-generated survey uses both provided papers, and turns a
   single-method implementation into generate-then-evaluate. Highest value per effort.
2. **Report the LLM-judge extension as a finding, not a footnote.** REPORT.md §10 already documents it
   honestly including the non-monotonic caveat. It is a genuine ablation of QUAL-SG's own relevance-scoring
   step and reads as a contribution rather than a fix.

Deadline still unknown. **Check Quanta directly** — forum posts do not reach Gmail, which is why the earlier
inbox search found nothing.

## 22. Plan — everything remaining

### A. Blocking, do first (this week)

| # | Task | Why it blocks | Owner |
|---|---|---|---|
| A1 | **Find the SOTA deadline on Quanta**, then submit the litreview | Work is done and sitting unsubmitted | anyone |
| A2 | **Confirm Sharanga's specs** (GPU count, interconnect, queue policy) | Changes what scale of training is possible for the recommended direction | anyone |
| A3 | **Confirm Agents4Science edition + deadline** in writing | The instructor already confirmed AI-primary-author; the *timeline* still gates everything | anyone |
| A4 | **Team decision: adaptation-audit vs Structure-Guided ReMasking** | Everything downstream forks here | all |

### B. If the adaptation-audit direction is chosen (recommended)

| # | Task | Cost |
|---|---|---|
| B1 | Reproduce DiffuGPT-S/M training from this repo — establishes the baseline and satisfies project stage 2 | days |
| B2 | Re-run the annealing ablation at 124M and 355M to confirm Table 3's +2.1 / +2.5 | days |
| B3 | Extend to a third scale (774M / 1.5B) — **the actual contribution**: does the annealing gain keep growing? | ~1 week on 2×96 GB |
| B4 | Mask-token ablation (bet 6): reused vocab token vs new token, at matched scale, disentangling the DiffuGPT-M confound | cheap, pairs with B2 |
| B5 | Order-metric comparison (bet 3 narrowed): DiffuLLaMA vs LLaDA vs Dream, decoding held fixed, using 2601.15593's metrics | inference only |
| B6 | Read [2512.06776](https://arxiv.org/abs/2512.06776) in full and position against it — it tests annealing at 7B but for **block**-diffusion, and its ablation is only 4,000 iterations | hours |

### C. If Structure-Guided ReMasking is chosen instead

| # | Task |
|---|---|
| C1 | **Phase 0 first** — failure taxonomy, slice-size distribution, **parse rate**, and **change rate** (§2.7) |
| C2 | Rewrite §1 novelty claim (delete the three preempted claims, adopt §8.3 framing) |
| C3 | Rewrite §4.1 entirely — sampler facts are wrong; add the `alg="origin"` assertion guard |
| C4 | Move §3.2 from backward slicing to a tight AST neighbourhood, per CDC Fig. 8(b) |
| C5 | Re-run `audit/test_shift_offset.py` step 5 with the **real** LLaMA tokenizer |
| C6 | Fix the confidence baseline to use max-prob, not sampled-token logprob (§2.1) |
| C7 | Replace the broken oracle with the synthetic-corruption testbed (§5.2) |
| C8 | Cite and faithfully implement STaRR / RCR for the instability baseline (§4.3) |

### D. Verification debt — outstanding regardless of direction

| # | Item | Status |
|---|---|---|
| D1 | CDC Fig. 8(b) numbers (34.3 / 26.9 / 24.1) | **Single-sourced.** Load-bearing for the "abandon slicing" recommendation — read the figure directly before acting on it |
| D2 | CDC is a v1 preprint from May 2026 | Re-check for updates/acceptance at kickoff |
| D3 | Dream's `alg="origin"` default, LLaDA's topk direction | Agent-verified, not personally verified. Cheap to confirm from the config files |
| D4 | Compute estimate ("2–4× optimistic") | FLOPs arithmetic, not measurement. One hour of benchmarking settles it |
| D5 | Bet 1's sequence-partitioning sliver | Came back COULD NOT VERIFY — unresolved either way |
| D6 | The scientific-infilling pivot (§18) | Zero novelty checking. Best venue fit of any idea raised; needs a search before it can be ranked |
| D7 | `mock.py` / DiffuGPT-small local path | Never exercised |

### E. Standing practice

Re-run Aalhad's novelty-verification skill at every milestone, not once. CDC was posted three months before
this audit and nearly invalidated a semester's framing; whatever preempts the chosen direction may not exist
yet today.

---

*Part 4 note: the hardware objection to bet 1 (§13) was partially retracted on 2026-08-29 when the
configuration was confirmed as 2×96 GB plus cluster access. The verdict did not change, but the reasoning
did, and the retraction is recorded rather than quietly edited away.*
