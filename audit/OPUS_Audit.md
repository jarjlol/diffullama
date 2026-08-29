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

## 7. Coverage caveat — read this before trusting Part 2

Two automated verification passes covering Aalhad's bets **1, 2, 5, and 6 died mid-run** (API spend limit)
and returned nothing usable. What follows is therefore **unevenly verified**, and is marked as such:

| Bet | Verification status |
|---|---|
| 3 — do dLLMs plan? | **Partially verified** — adjacent papers identified from an earlier retrieval pool, not individually fetched |
| 4 — conversion / dropped training step | **Fully verified** against the anchor paper's own text, quoted below |
| 1, 2, 5, 6 | **NOT VERIFIED.** No novelty search completed. Assessments below are reasoning-from-premises only |

Do not treat bets 1, 2, 5, 6 as novelty-checked. They need the search re-run before anyone commits.

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
