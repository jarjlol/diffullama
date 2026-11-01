# Structure-Guided ReMasking for Masked Diffusion Language Models
### Design & Implementation Specification

**Status:** design draft, pre-implementation
**Anchor paper:** *Scaling Diffusion Language Models via Adaptation from Autoregressive Models* (DiffuLLaMA), ICLR 2025
**Scope:** semester research project
**Companion doc:** [`DIFFULLAMA_PROJECT_SUMMARY.md`](DIFFULLAMA_PROJECT_SUMMARY.md) (original proposal)

---

## 1. Thesis

Masked diffusion language models can revise any token at any position, but every published
remasking policy chooses *which* tokens to revise using signals that are either **probabilistic**
(low confidence, confidence instability) or **spatial** (random, fixed window around an error).
None of them use the **semantic dependency structure of the generated artifact itself**.

For code, that structure is available for free: an AST, def-use chains, and an execution trace.
A bug's cause is frequently far from its symptom in token space but adjacent in dependency space.

> **Claim under test:** given a fixed budget of *K* tokens to remask and a fixed number of model
> forward passes, spending that budget on *dependency-related* positions repairs generated code
> more often than spending it on low-confidence, unstable, or nearby positions.

The contribution is deliberately narrow. Model-agnostic adaptive remasking is already an active
area and we do **not** claim it. We claim the use of an *external semantic dependency graph* to
choose a **non-contiguous** repair scope, evaluated under strict budget parity across multiple
backbones.

---

## 2. Research questions

**RQ1 (primary).** Under matched remasking budget and matched forward-pass budget, does
dependency-guided token selection yield a higher repair rate than probabilistic and spatial
selection baselines?

**RQ2 (generality).** Does any observed effect hold across ≥2 independently trained masked
diffusion backbones, or is it an artifact of one model's sampler?

**RQ3 (mechanism).** Does the benefit come from *non-contiguity* specifically — i.e. does the
method beat a contiguous-region baseline that masks the same number of tokens?

**RQ4 (cost).** What is the repair rate per additional forward pass, and how does it compare to
simply resampling the whole program?

### Hypotheses

- **H1.** Dependency-guided selection > confidence/window selection on repair rate at equal *K*.
- **H2.** The effect is largest on failures with a *localizable* symptom (uncaught exceptions,
  where a traceback names a line) and smallest on silent wrong-output failures.
- **H3.** A **hybrid** scorer (dependency structure ∩ low confidence) outperforms either pure
  signal. *We consider H3 more likely to hold than H1 in its pure form; see §10, Risk 1.*

---

## 3. Method

### 3.1 The unifying formulation

The central design decision of this project: **all selection policies, ours and every baseline,
are the same function under different parameters.**

For each candidate token position *p* in the generated region, compute

```
score(p) = α · structural_relevance(p)     # dependency distance from the failure site
         + β · (1 − confidence(p))          # model's own uncertainty
         + γ · instability(p)               # variance of confidence across denoising steps
         + δ · proximity(p)                 # token distance to the failing line
         + ε · noise(p)                     # tie-breaking / random policy
```

then remask the top-*K* positions.

Every method in the study is a setting of (α, β, γ, δ, ε):

| Method | α | β | γ | δ | ε | Notes |
|---|---|---|---|---|---|---|
| Random | 0 | 0 | 0 | 0 | 1 | lower bound |
| Low-confidence | 0 | 1 | 0 | 0 | 0 | standard dLLM remasking |
| Temporal instability | 0 | 0 | 1 | 0 | 0 | confidence-variance methods |
| Local window | 0 | 0 | 0 | 1 | 0 | contiguous; addresses RQ3 |
| AST subtree | 1* | 0 | 0 | 0 | 0 | *structure = enclosing subtree |
| **Dependency slice (ours)** | 1* | 0 | 0 | 0 | 0 | *structure = backward slice |
| **Hybrid (ours)** | 1* | β>0 | 0 | 0 | 0 | slice-constrained confidence |
| Oracle | — | — | — | — | — | masks the gold-diff region (upper bound) |

**Why this matters.** *K* is held constant by construction, so no method can buy a win with a
larger edit budget. Ablations are free — they are parameter sweeps, not new code paths. And there
is one selection code path to test rather than seven, which is a large reduction in bug surface
for a project whose central risk is a silent indexing error (§5.3).

### 3.2 Structural relevance

`structural_relevance(p)` is derived from a dependency graph over the generated program, seeded at
the localized failure site and propagated backward.

**Graph construction (static).** Nodes are AST statements; edges are:
- def → use (variable definition to each reference)
- data-flow (assignment right-hand side to left-hand side)
- call → argument and call → callee-definition
- control dependence (a statement to its governing `if`/`for`/`while` test)
- AST parent ↔ child

**Refinement (dynamic).** Static slices over-approximate badly on short programs (§10). Execute
the failing visible test under `sys.settrace`, recording:
- which lines actually executed (branches not taken are excluded from the slice)
- frame-local deltas per line, which recovers *writes* by diffing `frame.f_locals`

The working slice is then `static_backward_slice ∩ executed_lines`, which is substantially tighter
than either component. True dynamic slicing (tracking reads at bytecode level) is a possible
refinement but is not required for v1.

**Scoring.** Statements at graph distance *d* from the failure site receive relevance decaying in
*d* (e.g. `1/(1+d)`), which is then broadcast to every token position covered by that statement.
This gives a graded score rather than a binary in-slice/out-of-slice partition, so top-*K*
truncation degrades gracefully when the slice is larger than *K*.

### 3.3 Failure localization

Localization quality differs sharply by failure mode, and this drives H2:

| Failure mode | Localization signal | Precision |
|---|---|---|
| Syntax error | `SyntaxError.lineno/offset` | Exact, but AST unavailable — fall back to window around the error, no slicing possible |
| Uncaught exception | Traceback innermost frame in generated code | Exact line |
| Assertion / wrong output | *None* — only "the answer is wrong" | Must slice backward from `return`; worst case |
| Timeout | Last executed line from trace | Approximate |

The third row is the hard case and probably the most common one (§10, Phase 0).

### 3.4 The repair loop

```
draft ← generate(prompt)
for round in 1..R:
    result ← verify(draft, visible_tests)          # sandboxed
    if result.passed: break
    site  ← localize(result)
    G     ← build_dependency_graph(draft)          # static ∩ dynamic
    P     ← top_K( score(p) for p in generated_positions )
    draft ← refine(draft, remask=P, freeze=¬P, steps=T)
final_score ← verify(draft, HIDDEN_tests)          # never used to guide repair
```

`R`, `K`, and `T` are held identical across all methods.

---

## 4. Backbones

| Model | Params | bf16 weights | Role |
|---|---|---|---|
| LLaDA-8B-Instruct | 8B | ~16 GB | Primary |
| Dream-7B / Dream-Coder-7B | 7B | ~14 GB | Primary (Dream-Coder for code) |
| DiffuLLaMA-7B | 7B | ~14 GB | Historical baseline (this repo's anchor) |
| DiffuGPT-small | 127M | ~0.3 GB | **Local development only** — validates mechanics, not quality |

### 4.1 Sampler normalization — a required correction

The three backbones do **not** share an unmasking policy:

- **DiffuLLaMA** unmasks in **random order** — [`model.py:133`](model.py:133) selects positions with
  `rand_like(x0) < p_to_x0`. It computes `x0_scores` at [`model.py:143`](model.py:143) but
  **never uses them** for selection.
- **LLaDA / Dream** unmask **lowest-confidence-first**, the modern dLLM default.

Comparing methods across backbones without fixing this confounds "structure helps" with "these
models sample differently." **All backbones must run one common unmasking policy** implemented in
the adapter layer. We standardize on confidence-order (the current field default); DiffuLLaMA
already computes the needed scores, so this is a small change. Any deviation from a model's
released sampler must be documented, and DiffuLLaMA should additionally be reported under its
native random-order sampler as a sanity check.

---

## 5. System architecture

```
sgr/
  adapters/
    base.py          # DiffusionLMAdapter ABC — the ONLY model-specific surface
    diffullama.py    # wraps this repo's generate_samples()
    llada.py
    dream.py
    mock.py          # CPU-only fake model for tests (see §5.5)
  structure/
    ast_graph.py     # AST + def-use + control-dependence graph
    dynamic_trace.py # sys.settrace executed-lines + local-write recovery
    slicer.py        # backward slice, graded by graph distance
    offsets.py       # source span -> token index mapping  (SHIFT-AWARE, see §5.3)
  verify/
    sandbox.py       # subprocess isolation, rlimits, timeout, no network
    localize.py      # traceback / SyntaxError / assertion -> source site
  select/
    scorer.py        # the single weighted scorer of §3.1; all methods are configs
  loop.py            # the repair loop of §3.4
  bench/
    humaneval.py  mbpp.py  runner.py  metrics.py
```

### 5.1 Adapter interface

```python
class DiffusionLMAdapter(ABC):
    mask_token_id: int

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int,int]]]:
        """Token ids + char offset spans. MUST use a fast tokenizer."""

    def generate(self, prompt_ids, gen_len, steps, seed) -> Draft:
        """Draft carries: token ids, per-position final confidence,
           and per-step confidence history (for the instability baseline)."""

    def refine(self, ids, remask_positions, steps, seed) -> Draft:
        """Re-mask exactly `remask_positions`, freeze everything else, denoise."""

    def forward_count(self) -> int:
        """Cumulative model forwards — used to VERIFY budget parity empirically."""
```

### 5.2 The repair primitive already exists in this repo

`generate_samples()` in [`model.py:80`](model.py:80) accepts a `src_mask`, and
[`model.py:102`](model.py:102) sets `maskable_mask = ~src_mask`. Positions with `src_mask = 1` are
never masked or updated; the rest are masked and denoised.

That is exactly the required primitive. Passing the current draft as `input_ids` together with
`src_mask = 1` everywhere except the positions to repair yields arbitrary non-contiguous
structure-guided remasking **with no changes to the sampler**. The DiffuLLaMA adapter is
consequently thin.

### 5.3 The offset-mapping hazard (highest-risk component)

The mapping chain is:

```
AST node (lineno, col_offset, end_lineno, end_col_offset)
  → char span in the generated string        [line-start prefix sums]
  → token indices                            [tokenizer offset_mapping]
  → canvas positions                         [+ prompt_len, and ∓1 for the shift]
```

**The shift.** DiffuLLaMA is trained with an inherited autoregressive shift: position *i* predicts
the token at *i+1*. See [`model.py:120`](model.py:120)
(`x0 = cat([x[:,0:1], x0[:,:-1]])`) and the final `x0 = x0[:,1:]` at
[`model.py:156`](model.py:156). **A mapping that ignores this is off by one and silently masks the
wrong tokens** — producing a method that looks correct, runs end to end, and quietly underperforms.
This is the most likely way for the project to fail invisibly.

**Required invariant test.** For every AST node in a corpus of programs, map node → token indices,
decode those tokens, and assert the decoded text covers the node's original source text. This is
cheap, runs on CPU, and catches the shift bug immediately. It should be written *before* the
selector.

### 5.4 Sandboxing

The harness executes model-generated code. HumanEval's stock runner is not adequate. Requirements:

- separate subprocess per candidate, hard wall-clock timeout
- `RLIMIT_AS` (address space) and `RLIMIT_CPU`
- no network, minimal environment, non-writable working directory
- crash/timeout treated as an ordinary failure signal, never as a harness error

### 5.5 Local development without a GPU

The development laptop is a 6 GB RTX 3060 — it **cannot** hold an 8B model, and 4-bit quantization
would distort the logits that the confidence baselines depend on, making local comparisons
scientifically meaningless.

Everything outside `adapters/` is CPU-only: slicing, tracing, offset mapping, sandboxing, scoring,
metrics. These are developed and unit-tested locally against:

1. **`mock.py`** — a fake adapter that "denoises" by revealing tokens from a known target program.
   This permits an end-to-end assertion that the loop repairs a *deliberately corrupted* program,
   with zero GPU.
2. **DiffuGPT-small (127M)**, which runs locally through the same `generate_samples` path and
   exercises real tokenization, masking, and refinement. It will not write correct code; it
   validates mechanics only.

The GPU-dependent surface is therefore reduced to model loading and the refine loop.

---

## 6. Data & evaluation protocol

**Datasets.** HumanEval (164) and MBPP (~500). Use **EvalPlus** (HumanEval+ / MBPP+).

**Visible/hidden split.** Repair may consult *visible* tests only; final correctness is measured on
*hidden* tests that never influence repair.
- HumanEval has no natural split → **base tests visible, EvalPlus extra tests hidden.**
- MBPP has 3 asserts → 1 visible, 2 hidden (or the same base/plus scheme).

This split is the integrity backbone of the whole evaluation and must be enforced in code, not by
convention — the hidden suite should be unreachable from the repair path.

### 6.1 Baselines

The seven policies of §3.1, **plus two additions absent from the original proposal:**

- **Best-of-*N* resampling.** Sample *N* fresh drafts, keep any that passes visible tests, matched
  to the repair methods on total forward passes. For code repair this is a brutally strong
  baseline and reviewers will ask for it immediately. *If targeted repair cannot beat resampling at
  equal compute, the method has no practical case.* This addresses RQ4.
- **Oracle remasking.** Mask exactly the gold-diff region. This bounds what *any* selector can
  achieve and separates "our selector is weak" from "this backbone cannot repair."

### 6.2 Metrics

**Primary:** pass@1 on hidden tests · repair rate = repaired / failed-drafts ·
**regression rate** = broken / initially-passing drafts · net gain = repaired − broken.

**Cost:** model forward passes (logged, not assumed) · wall-clock · tokens remasked · tokens changed.

**Diagnostic:** compile rate · slice size as a fraction of function tokens · failure-mode
breakdown · localization accuracy where gold diffs exist.

### 6.3 Statistics

Drafts are **generated once per (backbone, seed, problem) and shared by all methods.** This cuts
compute several-fold *and*, more importantly, makes every comparison **paired** — all methods
repair identical drafts.

- Paired **McNemar test** per method pair, Holm correction across the family.
- 3 seeds; report mean ± CI, not single runs.
- With ~560 paired problems, differences of roughly ≥4–5 pp are resolvable; **smaller effects will
  not be**. The design should not claim resolution it lacks, and the doc states this in advance to
  avoid post-hoc rationalization of a null result.

---

## 7. Experiment matrix & compute

Grid: 2–3 backbones × 9 methods × ~560 problems × 3 seeds.

**Compute estimate on the 96 GB system.** Diffusion generation is fixed-length and uses no KV
cache, so it batches extremely well — batch 32 at sequence length 512 is comfortable within 96 GB
alongside 16 GB of weights.

| Stage | Count | Est. per sample | Total |
|---|---|---|---|
| Drafts (cached, shared) | ~3.4k | ~1 s (batched) | ~1 h |
| Repairs (T = 32–64 steps) | ~27k | ~0.4 s | ~3 h |
| Verification | CPU, parallel | — | overlapped |

**A full clean grid is on the order of 10–20 GPU-hours**, with ample headroom for re-runs and
pilot sweeps. If the 96 GB is several smaller cards rather than one, the grid is embarrassingly
parallel across (backbone × seed) and shards trivially.

**Conclusion: compute is not the bottleneck for this project.** The bottlenecks are (a) the
research risk in §10 and (b) implementation correctness, especially §5.3. Effort should be
allocated accordingly — this argues strongly for the Phase 0 probe below rather than an early
push on infrastructure.

---

## 8. Milestones

| Phase | Weeks | Deliverable | Gate |
|---|---|---|---|
| **0. Feasibility probe** | 1–2 | Failure taxonomy + slice-size distribution + oracle repair rate | **Go/no-go, §10** |
| 1. Core harness | 3–4 | Adapter ABC, mock adapter, sandbox, offset mapping + invariant test | Round-trip test green |
| 2. Structure module | 4–5 | AST graph, dynamic trace, slicer | Slices validated on hand-built cases |
| 3. Scorer + baselines | 6–7 | Unified scorer, all 9 policies, budget-parity assertions | Forward counts equal across methods |
| 4. Main runs | 8–10 | Full grid, 3 seeds, 2–3 backbones | — |
| 5. Analysis & writeup | 11–14 | Paired stats, ablations, paper | — |

---

## 9. Phase 0: the probe that decides the project

**Before building any repair logic**, run this. It is cheap and it tests the premise.

Generate drafts for HumanEval + MBPP with one backbone, then measure:

1. **Failure taxonomy** — what fraction are (a) syntax error, (b) uncaught exception,
   (c) wrong output with no exception, (d) timeout?
2. **Slice size** — median backward-slice size as a fraction of function tokens, per bucket.
3. **Oracle repair rate** — with gold-diff masking, how often can the backbone actually fix its
   own draft?

### Decision criteria (fixed in advance)

- **If ≥60% of failures are bucket (c) *and* median slice ≥80% of function tokens:**
  the pure dependency-slice method cannot differentiate itself from "regenerate everything."
  → **Pivot: the hybrid (slice-constrained confidence) becomes the primary method**, and the
  paper's claim narrows to "structure is a useful *prior* over which uncertain tokens to revise."
  This is still a defensible contribution and preserves the project.
- **If oracle repair rate < 15%:** the backbone, not the selector, is the bottleneck. No selection
  policy will show a meaningful effect. → Switch to a code-specialized backbone
  (Dream-Coder-7B) or narrow the claim to the localizable-failure subset.
- **If bucket (b) is substantial (≥30%) with tight slices:** the strongest version of the project
  is available — proceed as specified.

Recording these criteria *before* seeing the data is deliberate; it prevents the result from being
rationalized after the fact.

---

## 10. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Backward slices ≈ the entire function.** HumanEval solutions are 5–15 lines; slicing from `return` may pull in everything, collapsing the method into "regenerate all." | **Critical** | Fixed-*K* ranking makes structure a graded prior, not a region (§3.1); dynamic ∩ static tightens slices; Phase 0 measures it before commitment |
| 2 | **Off-by-one from the training shift** silently masks wrong tokens (§5.3) | **Critical** | Round-trip invariant test written before the selector |
| 3 | **Length rigidity.** Masking *K* tokens regenerates exactly *K*; fixes needing more tokens are unreachable | High | Measure what fraction of gold fixes are length-compatible; dilate masks to line boundaries for slack |
| 4 | **Silent-wrong-output failures cannot be localized**, forcing whole-function slices | High | H2 predicts this; report stratified by failure mode rather than aggregate only |
| 5 | **Sampler differences confound cross-model claims** (§4.1) | High | Normalize unmasking policy in the adapter layer |
| 6 | **Best-of-*N* resampling beats targeted repair** at equal compute | Medium | Included as a baseline; if it wins, that is a legitimate and publishable negative result about repair vs. resampling |
| 7 | **Underpowered comparisons** — small effects on n≈560 | Medium | Paired design on shared drafts, 3 seeds, McNemar; resolution limits stated in advance (§6.3) |
| 8 | Executing model-generated code | Medium | Sandbox per §5.4 |

---

## 11. Open questions for advisor review

1. **Claim strength.** Is "structure helps *within* a repair framework" sufficient, or must the
   method also beat best-of-*N* resampling at equal compute to be worth writing up?
2. **Scope.** Two backbones (LLaDA + Dream) or three (adding DiffuLLaMA)? Three strengthens RQ2 but
   costs the sampler-normalization work in §4.1.
3. **Pivot authority.** If Phase 0 triggers the §9 pivot, is moving the primary claim to the hybrid
   method acceptable, or should the project change direction more substantially?
4. **Extension.** Is the equation-dependency / math-reasoning extension in scope this semester, or
   explicitly deferred?
5. **Negative results.** If H1 fails but the Phase 0 diagnostics are informative, is a
   characterization paper ("when does structure help dLLM repair?") an acceptable outcome?

---

## Appendix A — Key code references in this repository

| Location | Relevance |
|---|---|
| [`model.py:80`](model.py:80) `generate_samples()` | The denoising loop to be wrapped |
| [`model.py:102`](model.py:102) `maskable_mask = ~src_mask` | The freeze/remask primitive — enables arbitrary non-contiguous masking |
| [`model.py:120`](model.py:120), [`model.py:156`](model.py:156) | The shift; source of the off-by-one hazard (§5.3) |
| [`model.py:133`](model.py:133) | Random unmasking order — differs from LLaDA/Dream (§4.1) |
| [`model.py:143`](model.py:143) `x0_scores` | Confidence, already computed but unused; needed for baselines |
| [`inf_diffullama.py:65`](inf_diffullama.py:65) | Worked example of constructing `src_mask` |
| [`inf_diffugpt.py`](inf_diffugpt.py) | Local 127M development path (§5.5) |
