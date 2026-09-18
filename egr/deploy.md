# Deploying and running EGR

How to actually run the code in `egr/`, what state it's in right now, and exactly what to do
next when real compute becomes available. Read this after `README.md` (the idea) and
`docs/01-architecture.md` (the design) — this is the "how do I press the button" document.

> **This POC never loads DiffuLLaMA and never touches the network.** Every command below runs
> on a laptop CPU, deterministically, using a `MockBackend` stand-in for the real model and a
> small embedded fixture set standing in for the real HumanEval+ corpus. That was a deliberate
> instruction for this build, not an oversight — §5 below is the exact list of what to swap in,
> and where, once compute and network access are available.

---

## 1. Requirements

- **Python 3.11+** (this was built and tested on 3.14). Phases 0–2 need **only the standard
  library** — no `pip install` at all. Confirm with:
  ```bash
  python3 --version
  ```
- Nothing else. `torch`/`transformers`/`datasets`/`evalplus` are **not** required to run
  anything described in this document — they are only imported lazily, inside
  `DiffuLLaMABackend.__init__`, which this document never calls (see §5).

## 2. Setup

```bash
cd egr/..              # repo root — the directory that CONTAINS egr/
python3 -m unittest discover -s egr/tests -p "test_*.py"
```

No virtualenv, no `pip install -e .`, no `PYTHONPATH` juggling needed for local use — `python3
-m egr.cli ...` and `python3 -m unittest ...` both work from the repo root as-is. (If you do
package this up for CI, `PYTHONPATH` should point at the directory containing `egr/`; the
sandbox subprocess in `verify.py` already sets this itself, see §4.4.)

## 3. Run the tests

```bash
PYTHONIOENCODING=utf-8 python3 -m unittest discover -s egr/tests -p "test_*.py" -v
```

**28 tests, all passing, in under 5 seconds, on CPU.** `PYTHONIOENCODING=utf-8` matters — this
project has hit cp1252 crashes three times before (`project-docs/05-mistakes-and-bugs.md`
M-11), once specifically from a SentencePiece `▁` character, and the tests deliberately contain
non-ASCII text to catch a regression.

What's covered, mapped to the hazard register (`docs/03-hazards.md`):

| Test file | Hazards closed |
|---|---|
| `test_invariants.py` | H-1 (shift off-by-one), H-9 (encoding) |
| `test_canvas.py` | H-2 (indentation bleed), H-11 (hand-built positional arrays) |
| `test_localize.py` | the scope-ladder cross-parse bug found while building this (see §6) |
| `test_verify.py` | H-5 (rigged confidence), H-6 (harness error), H-8 (sandbox), H-10 (empty test suite) |
| `test_loop_mock.py` | H-3 (fixed-length constraint), H-4 (deterministic stall), H-13 (regression tracking) |

A note on H-1's proof: `test_shift_offset_reference.py` and `test_real_tokenizer.py` in this
directory are the original standalone audit scripts this project's design docs cite as the
proof of the shift invariant. They were rescued from the `audit/opus-review` branch (which was
scheduled for deletion) and are kept here for reference; `test_invariants.py` is the version
integrated into this package's own test discovery.

## 4. Run the CLI

### 4.1 The localization study (Phase 2 — no model, no GPU)

```bash
PYTHONIOENCODING=utf-8 python3 -m egr.cli localize \
    --benchmark mutants --n 200 \
    --policies ours,static,confidence,random \
    --out runs/localize.jsonl
```

Prints a top-1/top-3 accuracy table with Wilson intervals, broken down by evidence kind, and
writes one JSON row per scored mutant to `--out`. This is the first real result the build-phase
plan calls for, and it runs in about a second. **Current result, on the placeholder fixture set
(§5), is recorded in `project-docs/02-decision-log.md` entry P-6** — a preliminary GO, with the
exact caveats that make it preliminary.

### 4.2 The full repair loop, against MockBackend

```bash
PYTHONIOENCODING=utf-8 python3 -m egr.cli run \
    --benchmark mutants --backend mock --mock-mode oracle \
    --policy ours --max-depth 5 \
    --out runs/demo.jsonl
```

`--mock-mode` is one of:

| Mode | Behaviour |
|---|---|
| `oracle` (default) | fills each hole from the task's known-correct source — proves the loop closes at all |
| `noisy` | fills correctly with probability `--mock-p` (default 0.5), else emits a plausible wrong token |
| `stuck` | always emits the same filler token — exercises the no-progress escalation ladder |

Swap `--policy` for any of `ours | static | confidence | random | resample | none` and
`--benchmark` for `mutants | humanevalfix` (see §5 for why `humaneval_plus`/`mbpp_plus` will
raise instead of running). Every run is resumable: re-running the same `--out` path skips task
ids that already have a summary line, so a killed run picks up where it left off
(`project-docs/04-build-phases.md`'s checkpoint-and-resume requirement).

### 4.3 Reading the results

Each line in a `--out` file is one JSON record — one per attempt, plus a trailing summary line
per task. No bespoke analysis script is needed:

```python
import pandas as pd
df = pd.read_json("runs/demo.jsonl", lines=True)
summaries = df[df["summary"] == True]
print(summaries["solved"].mean())          # pass@1-style aggregate
print(df.groupby("depth")["verdict"].value_counts())
```

### 4.4 What actually happens when you run a task (for debugging)

```bash
PYTHONIOENCODING=utf-8 python3 -m egr.cli run --benchmark mutants --backend mock \
    --mock-mode oracle --policy ours --max-depth 5 --out runs/demo.jsonl --verbose
```

`--verbose` turns on DEBUG-level logging across every module (`egr/log.py`'s `setup()`, called
once by `cli.py`). Every module does `log = logging.getLogger(__name__)` and nothing else, so
this is the only switch you need — no per-module configuration. Useful log lines to grep for:
`solved at depth`, `no progress`, `escalating`, `no maskable region`, `HARNESS_ERROR`.

Candidate programs run in a fresh `python3` subprocess per check
(`egr/_sandbox_harness.py`), with a resource-limited, temp-directory sandbox (H-8). If you need
to inspect that in isolation:

```bash
mkdir -p /tmp/egr_debug && printf 'def f(x):\n    return x + 1\n' > /tmp/egr_debug/solution.py
printf 'def test_0():\n    from solution import f\n    assert f(1) == 2\n' > /tmp/egr_debug/tests.py
PYTHONPATH=. python3 egr/_sandbox_harness.py /tmp/egr_debug 5 256
```

---

## 5. Known simplifications — read this before trusting a number

Everything below was a deliberate scoping decision for this session (no GPU, no network), not
an oversight. Each row says exactly what to do once that constraint lifts.

| # | What's simplified | Where | What to do instead, and where |
|---|---|---|---|
| 1 | **DiffuLLaMA is never loaded.** `DiffuLLaMABackend` is a complete implementation (tokenize/detokenize/infill/confidence, wired to `model.py`'s real `generate_samples` and `get_anneal_attn_mask`, unmodified per P5) but its `__init__` refuses to run unless called with `allow_load=True` **and** `torch`/`transformers` are installed. | `egr/backend.py` | Install `torch`/`transformers`, run `python -m egr.cli run --backend diffullama --allow-load ...` on a GPU machine, and follow `docs/04-build-phases.md` Phase 3's smoke test (10 problems, assert the diff is confined to the masked span) **before** trusting any repair-curve number. |
| 2 | **The benchmark corpus is a 6-function placeholder**, not the real HumanEval+/MBPP+/HumanEvalFix. `mutants.py`'s `_FIXTURES` and `humanevalfix.py`'s reuse of them are hand-written, not downloaded. | `egr/benchmarks/mutants.py`, `humanevalfix.py` | Get network access or the `evalplus`/`datasets` package, then replace `_FIXTURES` with a real loader over HumanEval+'s 164 canonical solutions (each module's own docstring says this). Nothing downstream needs to change — every module only depends on the `Task` objects `tasks()` yields. |
| 3 | **`humaneval_plus.py` and `mbpp_plus.py` raise `RuntimeError` on use.** Track B (pure generation from a bare prompt, no given buggy seed) is also not wired into `loop.py`, which always verifies `task.seed_program` — this is separate future work, not just a missing dataset. | `egr/benchmarks/humaneval_plus.py`, `mbpp_plus.py` | Same fix as #2 for the dataset half. For Track B, decide how the model's own output becomes depth-0's `program` in `loop.py` before wiring either benchmark into a `--backend diffullama` **generation** run. |
| 4 | **`confidence` in the Phase 2 localization study reads a uniform, uninformative score.** `MockBackend.confidence()` always returns 0.5 everywhere — there is no real model to read confidence from. | `egr/backend.py::MockBackend.confidence`, `egr/cli.py::cmd_localize` | Per `docs/04-build-phases.md` Phase 2: load **DiffuGPT-S (124M)**, which fits in ~4 GB and stays GPU-free, and pass it as the backend for the `confidence` policy only. Re-run `egr.cli localize` and treat the new numbers as the real Phase 2 gate result — the current table is explicitly flagged preliminary in `project-docs/02-decision-log.md` P-6. |
| 5 | **The "matched budget" ceiling is a target, not a hard cap.** `policy._budget_trim` widens a neighbourhood by `slack` and checks a `budget + slack` ceiling, but does not cut an oversized span, because an earlier version did cut it (at an arbitrary token offset) and that silently broke H-2's whole-line invariant. | `egr/policy.py::_budget_trim` | Implement a genuinely line-safe trim: walk inward from one end by whole lines until under budget, never stopping mid-line. Needed for a fully rigorous B2 (random-span, matched-budget) comparison; not needed for the POC's own acceptance tests, which all pass without it. |
| 6 | **`slack` cannot always rescue a fix that needs more tokens**, specifically when the extra tokens are needed on the LAST line that differs and there is no unchanged trailing content left to widen into. Discovered while building `test_loop_mock.py::TestFixedLengthConstraint` — see the test's own docstring and `_budget_trim`'s "HARD LIMIT" note for the full derivation. | `egr/policy.py::_budget_trim`, `egr/backend.py::MockBackend._oracle_replacement_ids` | This is a property of masked diffusion's fixed-length infilling (H-3), not a bug — expected to matter for the real model too. The `slack ∈ {0, 4, 8}` ablation already planned in `docs/02-experiment-plan.md` is the right way to measure how often it binds in practice. If it binds often, the real mitigation is building the canvas with reserved trailing padding positions for growth (`Canvas.build()` does not do this) — a design change, not a fix, and should be scoped as its own step. |

## 6. What got fixed while building this (for the record)

Building the code surfaced four real bugs that the design docs' hazard register did not
specifically predict, on top of the ones it did (H-1 through H-13, all closed — see §3's
table). All four are fixed in the current code and locked in by a test:

1. **`neighbourhood()` parsed the program twice** (once for the witness, once for the ancestor
   walk via a fresh `_parent_map`), so `parent_leaf`/`function` scope silently degenerated to
   `leaf` — two `ast.parse()` calls on the same text return unrelated node objects, so an
   ancestor lookup keyed by node identity always missed. Fixed by making `neighbourhood()` do
   a single parse and never accept a node handed in from elsewhere. Locked in by
   `test_localize.py::test_scopes_differ_when_nesting_allows_it`.
2. **A syntax error left `neighbourhood()` with nothing to walk** (there is no AST when the
   program does not parse), so every syntax-evidence repair attempt found no maskable region —
   silently defeating exactly the case the README calls out as common for a base diffusion LM.
   Fixed by `localize._fallback_neighbourhood`, a text-only heuristic used only when `ast.parse`
   fails.
3. **`_budget_trim`'s original cut-at-a-token-offset trim broke H-2's whole-line invariant.**
   Cutting an oversized neighbourhood at an arbitrary token position (rather than a line
   boundary) corrupted whatever statement the cut landed inside. Fixed by making the function
   never cut, only log when the ceiling is exceeded (see §5 row 5).
4. **An off-by-one in `MockBackend._oracle_replacement_ids`'s line-range slice** pulled in one
   line more than the mask actually covered (`oracle_lines[start:end + 1]` where `end` was
   already the correct exclusive bound). Fixed; the corrected version is what made the oracle
   splice tests in §3 pass exactly, not approximately.

## 7. Implementation checklist

Maps `docs/04-build-phases.md`'s phases to what actually exists. ✅ = done and tested; ⏭️ =
deliberately deferred, see §5 for exactly what unblocks it.

- [x] **Phase 0 — hazards first.** `types.py`, `canvas.py`, `test_invariants.py`,
      `test_canvas.py`. `test_shift_offset_reference.py` rescued from `audit/opus-review`.
- [x] **Phase 1 — the whole pipeline, without a model.** `verify.py`, `trace.py`,
      `evidence.py`, `localize.py`, `policy.py` (all six policies), `backend.py`'s
      `MockBackend`, `loop.py`, `report.py`, `cli.py`, `benchmarks/{base,mutants,humanevalfix}.py`.
  - [x] `test_verify.py`, `test_localize.py`, `test_loop_mock.py` — 28/28 tests passing.
  - [x] `python -m egr.cli run --benchmark mutants --backend mock --mock-mode oracle` closes
        the loop and writes a well-formed JSONL run record (§4.2).
  - [x] Sandbox guard tests: runaway loop, unbounded allocation, write outside sandbox,
        immediate `sys.exit` — none crash the harness (`test_verify.py::TestSandboxContains`).
  - [x] Canonical solution passes, a known mutant fails, for every task the (placeholder)
        benchmark yields (`test_verify.py::test_canonical_passes_and_mutant_fails_for_every_mutants_task`).
  - [⏭️] The same, for the real 542-problem EvalPlus corpus — needs §5 row 2.
- [x] **Phase 2 — the localization study.** `python -m egr.cli localize` runs, produces a
      top-1/top-3 table with Wilson intervals and a per-evidence-kind breakdown.
  - [x] Go/no-go recorded in `project-docs/02-decision-log.md` — **preliminary GO**, with the
        exact caveats (n=15 placeholder mutants, uninformative `confidence` column) that keep
        it from being the real gate result. See §5 rows 2 and 4 for what changes that.
- [⏭️] **Phase 3 — real backend, smoke test, measurement.** `DiffuLLaMABackend` is fully
      implemented in `backend.py` but has never been instantiated (§5 row 1). The "measure,
      then delete the arithmetic compute budget" step in `docs/02-experiment-plan.md` §8 has
      not happened — that section is still an estimate.
- [⏭️] **Phase 4 — the repair curve (Track A).** Blocked on Phase 3 and on §5 row 2 (real
      HumanEvalFix data).
- [⏭️] **Phase 5 — Track B, both backbones.** Blocked on Phase 4, and additionally on deciding
      how Track B feeds `loop.py` (§5 row 3) and on a Dream-Coder backend implementation (not
      started — only `DiffuLLaMABackend` exists).
- [⏭️] **Phase 6 — ablations and write-up.** Blocked on Phase 5.

**Bottom line: the plan's entire GPU-free critical path (Phases 0–2) is implemented, tested,
and has produced its first (preliminary) result. Nothing GPU-dependent has been attempted, per
explicit instruction — §5 is the handoff note for the session that has compute.**
