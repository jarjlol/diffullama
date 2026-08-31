# Proposed Option C2 — Trace-Guided Post-Hoc Repair for Frozen Diffusion Code Models

> **Status: proposed, not decided.** This records the direction discussed on 2026-08-31. The team must
> still choose between this and Option B.

## The question

Can an actual failed execution select a better repair region for a frozen diffusion code model than model
confidence or simple positional heuristics?

The initial backbone is DiffuLLaMA. A completed generated program is run against visible tests. On failure,
a small external controller uses the failure report, execution trace, and a lightweight program structure to
choose at most `K` tokens to reopen. DiffuLLaMA regenerates only those masked positions with its native
sampler. The process repeats for at most `R` attempts, initially `R = 10`.

```text
prompt → DiffuLLaMA generates complete code → visible tests
                                              │ fail
                                              ▼
                   traceback + executed lines + AST/def-use links
                                              │
                                              ▼
                                  choose <= K token spans
                                              │
                                              ▼
                            remask spans → native dLLM repair
                                              │
                                              └── re-run visible tests (<= R attempts)
```

## Terminology

- **Functional correctness:** the program compiles/runs and gives the expected result on the task’s tests.
  A wrong output, failed assertion, exception, or timeout is a functional failure. The test harness, not an
  LLM, determines it.
- **AST:** a syntactic tree: assignments, calls, conditions, expressions, and returns. It says how code is
  written, not what executed.
- **Dynamic trace:** source lines actually executed for one input, in order; it can also include exception
  and lightweight value information.
- **Trace-guided static slice (v1):** mark AST nodes whose lines executed in the failing run, then expand
  through conservative static def-use and control links. This is not a full dynamic dataflow slice; that
  would require much more runtime instrumentation.

## Why diffusion is useful here

Autoregressive models generally regenerate a suffix or emit a new patch. A masked diffusion model can
preserve most visible code and reopen an arbitrary, potentially non-contiguous set of positions. The
project asks whether external execution evidence can decide *which* positions to reopen. It does not claim
that localized revision itself is new.

## Relationship to closest work

| Work | What it does | Why C2 differs |
| --- | --- | --- |
| [CDC](https://arxiv.org/pdf/2605.16829) | Applies constraints during reverse denoising. GradGuide uses a trained correctness surrogate and gradients; MDFI uses static-security-analysis witnesses, AST/dataflow neighborhoods, remasking, optional insertion, and hints. | C2 starts after a complete program fails an actual test, uses no learned surrogate or gradient, and studies functional repair-location selection under equal budgets. |
| [CDLM](https://arxiv.org/pdf/2512.15596) | Post-trains a DLM on visible synthetic corruptions so low confidence identifies tokens to iteratively remask. Tests grade the benchmark but do not guide each revision step. | C2 keeps the backbone frozen and uses external test-trace evidence rather than learned token confidence. |
| [Self-Debugging](https://openreview.net/pdf?id=KuPixIqPiq), [NExT](https://proceedings.mlr.press/v235/ni24a.html), [RePair](https://aclanthology.org/2024.findings-acl.973/), [RepairAgent](https://arxiv.org/abs/2403.17134) | Use execution feedback, traces, or iterative program repair with autoregressive models/agents. | C2 is not the first test-driven repair loop. Its target is frozen diffusion-model remasking and a controlled span-selection comparison. |

## Exact contribution claim

> We study whether execution-trace-guided, structure-aware span selection improves **post-hoc,
> fixed-budget repair** by frozen diffusion code models over random, confidence, traceback-window, and
> static-only remasking policies.

This is an empirical systems contribution. It should not claim:

- first test-driven iterative code repair;
- first structure-guided remasking for diffusion code models; or
- guaranteed or near-100% correctness.

## Method

### 1. Generate and run visible tests

Generate one complete candidate. Execute it in a sandbox against a visible repair-test set. Preserve the
failure report: expected/actual output, exception/timeout, traceback, and executed source lines.

### 2. Create the repair-region candidates

Parse the candidate. Map executed lines to AST nodes. Add conservative links for local definitions and
uses, returned expressions, calls, and enclosing conditions/loops. Rank spans using execution relevance,
failure proximity, and static dependency relevance; align selected source spans to model-token positions.

The shift invariant in `03-established-facts.md` F-3 is mandatory: for this backbone, canvas position is
returned-token index + 1. Assert every mapping before masking.

### 3. Remask and regenerate

Mask only selected spans; all other code remains fixed. Invoke the backbone’s native sampler for the chosen
repair schedule. The initial scope is **replacement-only**: no insertion/deletion or custom common sampler.

### 4. Bounded acceptance loop

For at most `R` attempts:

1. If all visible tests pass, stop.
2. Produce one or more allowed region candidates.
3. Generate a repair.
4. Accept it only if it repairs at least one visible failure and does not break a previously passing visible
   test; otherwise retain the best candidate and try another seed/region.

This makes the visible-test objective monotonic, but it is not proof of correctness.

## Required baselines

Every policy must share the backbone, original candidate, maximum remasked-token budget `K`, repair attempt
limit `R`, decoding configuration, and visible tests.

| Policy | What it tests |
| --- | --- |
| No repair | Original generation quality |
| Whole-program regeneration | Practical resampling baseline |
| Random `K` tokens | Whether extra samples alone explain gains |
| Traceback window | Failure location without program structure |
| Static AST/def-use slice | Structure without execution evidence |
| Confidence remasking | Internal model signal; only if exposed fairly per F-4 |
| Trace-guided AST/def-use slice | Proposed combined signal |

## Evaluation

Visible tests may guide repair. **Held-out tests may not.** Report:

- held-out repair success among initially failing candidates;
- visible-to-held-out gap, to expose test overfitting;
- regression rate on previously correct behavior;
- changed/remasked tokens and span count;
- model forwards, test executions, wall-clock time, and attempts per recovered program.

Use paired comparisons: each policy repairs the same starting candidate with matched seeds/budgets. Do not
treat ten attempts for one program as ten independent examples.

## Phase 0: the feasibility gate

Before building the controller, test 50–100 generated failures with manually selected plausible repair
spans. This answers whether frozen DiffuLLaMA can revise a known relevant region at all.

- If manual spans rarely repair code, a better selector cannot save the direction.
- If they often repair code, compare traceback windows and static slices before implementing richer tracing.
- Proceed to the full controller only if trace evidence improves held-out repair quality enough to justify
  its complexity.

## Main risks

1. Full-sequence diffusion forwards mean selective masking does **not** automatically save compute; quality
   and regression reduction are the primary hypotheses.
2. Some functional failures execute a broad, uninformative path.
3. Replacement-only remasking cannot express fixes requiring new statements.
4. A frozen model may simply reproduce the same tokens; temperature and sampling policy must be explicit.
5. Public-test success can overfit. Held-out testing is non-negotiable.
