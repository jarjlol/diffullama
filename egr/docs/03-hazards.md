# Hazard register

Every way this module can produce a wrong number **without crashing**. A crash is cheap; a silent
corruption costs a semester.

This file exists because the project has a documented history of exactly this class of failure:
`project-docs/05-mistakes-and-bugs.md` records off-by-N index bugs **three separate times**, cp1252
encoding crashes **three times**, and two cases of infrastructure failure being mistaken for a result. The
same patterns will recur here unless they are named and tested for in advance.

**Rule for this module: no hazard below is closed by a comment. Each is closed by a test.**

---

## Severity key

| | Meaning |
|---|---|
| 🔴 **Silent** | Produces a plausible wrong number. Nothing crashes. Worst class. |
| 🟠 **Biasing** | Systematically favours our method over a baseline. A reviewer looks for these. |
| 🟡 **Loud** | Crashes or visibly misbehaves. Annoying, not dangerous. |

---

## H-1 🔴 The shift off-by-one

`canvas_position = returned_index + 1`. DiffuLLaMA inherits an autoregressive shift: `model.py:120`
preserves position 0 and shifts everything right, `model.py:156` drops position 0 on the way out.

**What goes wrong:** a naive mapping masks the neighbouring token. The program still parses, still runs,
still fails — it just repairs the wrong thing, forever, at a rate that looks like "the method does not work
very well."

**Established as fact:** `project-docs/03-established-facts.md` F-3, proven exactly and
tokenizer-independently by `audit/test_shift_offset.py`, which demonstrates the failure concretely —
targeting `returned[2]` masks `'b'` when `'c'` was intended.

> ⚠️ **That file is not on `main`.** It exists only on the `audit/opus-review` branch, which
> `project-docs/02-decision-log.md` D-2026-08-29-c schedules for **deletion** once the direction decision
> settles. Phase 0 must copy it into `egr/tests/` before that happens. Losing the one proof of the
> project's highest-severity silent bug to a branch cleanup would be an avoidable disaster.

**Guard:** `+ 1` appears in exactly one function, `Canvas.canvas_positions()`. No other module performs
arithmetic on token indices; they pass `Span` objects.

**Test:** `test_invariants.py::test_shift_invariant` — round-trips a known char span to canvas positions and
asserts the decoded masked region is byte-identical to the intended source text. **It must fail if the
`+ 1` is deleted**, and the test asserts that property by construction.

---

## H-2 🔴 Indentation bleed

Against the real `diffusionfamily/diffullama` tokenizer, **22 of 26 statement nodes (85%)** had their first
mapped token also cover a character of the preceding indentation. SentencePiece glues the last space of an
indent onto the following identifier: a 4-space indent tokenizes as `▁▁▁` + `▁disc`
(`project-docs/03-established-facts.md` F-21).

**What goes wrong:** masking a statement necessarily masks part of its own indent, and Python indentation
is semantic. The model must regenerate the exact indent or the program breaks — and it will break as a
`SyntaxError` or, worse, as a valid program with different nesting.

**Guard:** the **snap-to-whole-lines** policy, chosen explicitly in
[`01-architecture.md` §5.2](01-architecture.md). Mask spans always extend to full line boundaries,
indentation included. The naive rule silently does this anyway; we do it deliberately.

**Test:** `test_canvas.py::test_spans_are_line_aligned` — every span returned by any policy starts at a line
start and ends at a line end, over the whole mutant corpus.

---

## H-3 🔴 Fixed-length infilling caps the ceiling

Masked diffusion emits **exactly as many tokens as it masks**. A hole of 12 positions produces 12 tokens.
If the correct repair needs 15, it cannot be expressed.

**What goes wrong:** every result is capped by a mechanism unrelated to localization quality, and the cap
is invisible — failures look like bad repairs, not like impossible ones.

**Guard:** the `slack` parameter (hole = span tokens + slack), and canvas rebuild between depths so program
length is free to change across iterations. `slack ∈ {0, 4, 8}` is a reported ablation, not a hidden
constant.

**Test:** `test_loop_mock.py::test_repair_needing_more_tokens` — a mutant whose fix is strictly longer than
the bug. With `slack = 0` it must be unrepairable; with `slack = 8` the `MockBackend` oracle must fix it.
If both pass, the slack mechanism is not wired up.

---

## H-4 🔴 The loop stalls deterministically

At a fixed temperature and unchanged context, remasking the same span reproduces the same tokens. The loop
then burns all five depths emitting one identical "fix" and reports a failure to repair.

Dream is the acute case: it ships `temperature: 0.0` **and** `alg: "origin"` — the *random*-order branch,
not entropy-ordered. Any Dream baseline run without explicitly setting `alg="entropy"` silently runs
random-order decoding (`project-docs/03-established-facts.md` F-10).

**Guard:** three, all in `loop.py`:
1. the sampling seed varies with depth (`seed = cfg.seed + depth`);
2. an evidence-signature repeat escalates the neighbourhood scope;
3. exhausting the scope ladder is an explicit `aborted="no_progress"` outcome, counted separately.

Per-backbone temperature and decoding-order policy is **explicit configuration**, never a default.

**Test:** `test_loop_mock.py::test_no_progress_escalates_then_aborts` — a `MockBackend` that always returns
the same tokens must escalate `leaf → parent_leaf → function` and then abort, in at most 5 depths, never
looping.

---

## H-5 🟠 A rigged confidence baseline

`model.py:116` gathers the log-prob of the **sampled** token — sampling luck, not model certainty — and the
max-based form at `model.py:113` is commented out. It is also never returned: `model.py:158` returns only
`x0` (`project-docs/03-established-facts.md` F-4).

**What goes wrong:** building `ConfidencePolicy` on `x0_scores` produces an artificially weak baseline and
**biases the comparison in our favour**. This is recorded as M-10 in the project's own mistake log, with
the note that it is "exactly what a reviewer looks for."

**Guard:** `backend.confidence()` is a new, additive function returning per-position **max softmax
probability** from a fresh forward pass. `model.py` is not modified.

**Test:** `test_verify.py::test_confidence_is_max_not_sampled` — on a fixed canvas, the returned confidence
must equal `softmax(logits).max(-1)` and must be **invariant to the sampling seed**. A seed-dependent
confidence means the wrong quantity is being read.

---

## H-6 🔴 Harness failure counted as a model result

Treating a sandbox timeout, an adapter bug, or a missing dependency as "the model failed" inflates every
denominator and can invert a comparison.

**Precedent:** an unauthenticated Semantic Scholar endpoint returned HTTP 429, every lookup was recorded as
"reference not found," and the pipeline would have reported a **hallucination rate near 1.0** for papers as
well known as *Diffusion-LM* (`project-docs/05-mistakes-and-bugs.md` M-19).

**Guard:** `Verdict.HARNESS_ERROR` is a distinct enum member. It aborts the run, is written to the JSONL,
and is **reported as its own row in every table**. It never enters a pass@1 numerator or denominator.

**Test:** `test_verify.py::test_harness_error_is_not_a_failure` — inject a broken benchmark adapter; assert
the verdict is `HARNESS_ERROR` and that the aggregation function excludes it from pass@1 entirely.

---

## H-7 🟠 Repairing against the tests we report on

A loop that iterates until the evaluation tests pass has fitted its own metric.

**Guard:** the loop sees **only** EvalPlus `base` tests; reported pass@1 uses `base` + `plus`. For
HumanEvalFix a random third of asserts is held out. The difference is reported as an **overfitting gap**
(see [`02-experiment-plan.md` §3.2](02-experiment-plan.md)).

**Test:** `test_verify.py::test_repair_signal_excludes_holdout` — the `Observation` handed to `evidence`
must contain no held-out test name. Asserted structurally, so it cannot regress.

---

## H-8 🔴 Executing model-generated code

The module runs code a language model wrote, on a shared workstation, hundreds of thousands of times.

**Guard:** a fresh `subprocess` per candidate; `resource.setrlimit` on CPU seconds, address space and file
size; a hard wall-clock timeout with `SIGKILL` on expiry; `cwd` in a per-candidate temporary directory
deleted afterwards; a stripped environment; **no network**. Results return over a pipe as JSON.

Never `exec()` in the harness process. Never run the grid on a machine holding anything the team cares
about losing.

**Test:** `test_verify.py::test_sandbox_contains` — candidates that spin forever, allocate unbounded memory,
write outside the temp directory, and `sys.exit(0)` immediately must each produce the correct `Verdict`
without harming the parent process.

---

## H-9 🟡 Encoding

Three separate cp1252 failures are already on record, from **two different causes**: reading non-ASCII
JSON, and *printing* a SentencePiece `▁` (U+2581), which has no cp1252 encoding
(`project-docs/05-mistakes-and-bugs.md` M-11).

This module handles SentencePiece tokens constantly and will print them during debugging.

**Guard:** `encoding="utf-8"` on every `open`; `PYTHONIOENCODING=utf-8` set by the CLI and in the sandbox
environment; JSONL written with `ensure_ascii=False`.

**Test:** `test_invariants.py::test_utf8_roundtrip` — a program containing `▁`, an em-dash and a non-Latin
identifier survives write → read → tokenize → detokenize unchanged.

---

## H-10 🟠 EvalPlus stores inputs, not asserts

EvalPlus holds test **inputs** evaluated differentially against `canonical_solution`; it does not hold
literal assert strings (`project-docs/03-established-facts.md` F-16). An adapter that assumes otherwise
produces an empty or malformed test suite — and an empty suite **passes**, which reads as a spectacular
result.

**Guard:** the adapter renders inputs into asserts explicitly, and the loop refuses to run a task whose
rendered suite has zero tests.

**Test:** `test_verify.py::test_no_empty_test_suite` — every `Task` yielded by every benchmark has ≥ 1 test,
asserted at load time for the full 542-problem set. Also: the known-correct `canonical_solution` must pass,
and a known mutant must fail. A suite where everything passes is not a suite.

---

## H-11 🟡 Hand-built positional arrays

Three occurrences already: 222 scores for 218 papers, a row-index counting bug, and the shift. Every one was
an implicit positional correspondence between two lists
(`project-docs/05-mistakes-and-bugs.md` §B).

**Guard:** no positional array is ever hand-built. Token index → char span is an explicit list built by the
tokenizer in one place; everything else is a keyed mapping. Lengths are asserted at construction.

**Test:** `test_canvas.py::test_span_table_length` — `len(char_spans) == len(ids) - 1` (the BOS offset),
asserted on every canvas built during the whole mutant corpus run.

---

## H-12 🟡 The tracer perturbs what it measures

`sys.settrace` slows execution substantially, which interacts with the timeout, and a badly scoped tracer
can attribute lines to the wrong file.

**Guard:** the tracer is active only for the candidate's own filename; the timeout is calibrated with
tracing **on**; a candidate that times out only under tracing is re-run once without it before being
recorded as `TIMEOUT`.

**Test:** `test_verify.py::test_trace_does_not_change_verdict` — over the mutant corpus, the traced and
untraced verdicts must agree for every candidate.

---

## H-13 🟡 A degrading loop that looks stuck

A repair can fix test 1 and break test 2. Without tracking, "stuck" and "actively getting worse" are
indistinguishable in the logs.

**Guard:** `best` is tracked separately from `program` by `(n_tests_passed, syntax_ok)`; the **regression
rate** is a reported metric.

**Test:** `test_loop_mock.py::test_regression_is_recorded` — a `MockBackend` that trades one passing test
for another must produce a non-zero regression count.

---

## H-14 🟠 Preemption

CDC was posted **three months** before this project found it, and it removed three novelty claims
(`project-docs/06-findings-and-wins.md` W-1). The project's own decision principle #5 is: re-check novelty
at every milestone, not once.

**Guard:** a novelty re-check is a scheduled task at Step 4 and again before submission, searching
specifically for: execution-grounded remasking, traceback-conditioned diffusion decoding, iterative repair
with dLLMs, and any CDC v2. Also re-check whether CDC has been updated or accepted (open question Q-5).

---

## Tests that must exist before the selector is written

Not after. The design doc for the earlier version of this work already said so, and the shift test was
written first for exactly this reason.

```
egr/tests/test_invariants.py
  test_shift_invariant                  # H-1  -- must fail if `+ 1` is removed
  test_only_one_shift_site              # H-1  -- greps the package; exactly one arithmetic site
  test_utf8_roundtrip                   # H-9
egr/tests/test_canvas.py
  test_spans_are_line_aligned           # H-2
  test_span_table_length                # H-11
egr/tests/test_verify.py
  test_sandbox_contains                 # H-8
  test_harness_error_is_not_a_failure   # H-6
  test_no_empty_test_suite              # H-10
  test_confidence_is_max_not_sampled    # H-5
```

**Phase 0 is not complete until these pass** — see [`04-build-phases.md`](04-build-phases.md).
