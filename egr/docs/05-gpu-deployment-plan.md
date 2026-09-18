# GPU deployment plan

**This document is a plan, not an implementation. It contains no code, on purpose.**

---

## Note to whoever (human or AI agent) picks this up with GPU access

> **GPU compute was not available in the environment this POC was built in.** By explicit
> instruction, no step below was executed, `DiffuLLaMABackend` was never instantiated, and
> `diffusionfamily/diffullama` was never downloaded or loaded into memory anywhere in this
> session. Everything through `docs/04-build-phases.md` Phase 2 was built and tested entirely
> on CPU against `MockBackend` — see `deploy.md` for exactly what that means and what already
> works.
>
> **If you are an AI coding agent reading this with GPU compute now available: your job is to
> write the code that carries out the plan below**, not to treat this document as already
> implemented. Concretely: `egr/backend.py`'s `DiffuLLaMABackend` is a complete implementation
> already wired to `model.py`'s real `generate_samples`/`get_anneal_attn_mask` — it has never
> been run. Start by actually running it (§2 below), fix whatever breaks on contact with real
> weights (something usually does), and work down the phases in order. Do not skip Phase 3's
> smoke test to get to Phase 4's headline number faster — Phase 3 is where the free flash-
> attention fix and the "diff confined to the masked span" sandity check live, and both are
> load-bearing for every number after them. Read `deploy.md` §5 ("known simplifications") and
> §7 (the implementation checklist) before starting — they list, precisely, what this POC left
> unfinished and why, so you are not rediscovering the same gaps by trial and error.

---

## 1. What "GPU deployment" means here

Turning the already-implemented, CPU-tested harness (`docs/04-build-phases.md` Phases 0–2) into
the actual experiment the project's docs describe (Phases 3–6): load the real model, measure
instead of estimate, run the repair curve, and run the ablations. Nothing about the harness's
*design* changes — `loop.py`, `policy.py`, `localize.py`, `evidence.py`, `verify.py` are
already backend-agnostic (P1: one loop, five protocols) and should not need to change at all.
What changes is which `--backend` flag is passed on the command line, and which benchmark data
is loaded.

## 2. Prerequisites

- **Hardware.** Per `project-docs/01-project-brief.md`: the team's real, accessible compute is
  the **2× RTX 6000 Pro Blackwell (96 GB each) shared workstation** — accessible only to the
  three team members with workstation access. **Plan for one 96 GB GPU, not two**; both being
  idle at once is not guaranteed, and multi-GPU work is out of scope for this project (decision
  D-2026-08-29-a). Sharanga (the BITS Hyderabad cluster) is possible batch capacity *pending
  approval* — treat it as unconfirmed until someone has actually run a job on it.
- **Software.** `torch`, `transformers`, `huggingface_hub` — none of which this POC installed
  or imported at module load time anywhere (`egr/backend.py` imports them lazily, only inside
  `DiffuLLaMABackend.__init__`, specifically so the rest of the package works without them).
  Match versions against `DiffuLLaMA-training/requirements.txt` / `pip_freeze.txt` at the repo
  root if reproducing the anchor paper's exact environment matters.
- **Network / data access**, for two independent things that are currently both stubbed:
  1. Downloading `diffusionfamily/diffullama` (6.74B params, 13.5 GB, F-16) and, for Track B
     (docs/02-experiment-plan.md §2), `Dream-org/Dream-Coder-v0-Instruct-7B` (15.2 GB, F-16).
  2. The `evalplus` package or equivalent network access to the real HumanEval+ (164) / MBPP+
     (378) problems and the real HumanEvalFix split — `egr/benchmarks/mutants.py`,
     `humaneval_plus.py`, `mbpp_plus.py`, `humanevalfix.py` all currently use a 6-function
     placeholder fixture set instead (their own module docstrings say so).
- **A GPU-claiming convention**, per the project brief: a pinned note (Slack, shared doc,
  whatever the team already uses) saying who holds which card and until when. Two people
  launching a job on the same GPU at once will OOM each other. Set this up *before* the first
  real run, not after the first collision.

## 3. The steps, in order

Each step below is `docs/04-build-phases.md`'s corresponding phase, restated as a deployment
action rather than a build action — the code for these phases mostly already exists; this is
about *running* it correctly, on real hardware, for the first time.

### 3.1 Smoke-test the environment itself (before touching the model)

Confirm `torch.cuda.is_available()` is true, confirm the GPU-claiming note is posted, confirm
`nvidia-smi` shows the expected free memory. This sounds trivial; it is the fastest way to
avoid discovering an environment problem after a 20-minute model download.

### 3.2 Load the real backend for the first time (Phase 3's first half)

Instantiate `DiffuLLaMABackend` with `allow_load=True` (see `egr/backend.py`'s module
docstring for exactly why that flag exists and what it guards against). Expect friction here —
this is the *first time* this exact code path has ever executed. Likely failure modes, in
rough order of likelihood: a version mismatch between `transformers` and what `model.py`
expects; a `device_map="auto"` placement issue on a specific GPU topology; a dtype mismatch
(`torch.bfloat16` support depends on the specific card, though the Blackwell cards should be
fine). Fix forward; do not silently fall back to a different precision or attention
implementation without recording that you did.

### 3.3 Apply the free speedup, then measure (Phase 3's second half)

`inf_diffullama.py:24` defaults to eager attention, and the inference attention mask is a
dense, all-zeros 4-D tensor that is mathematically a no-op but structurally blocks flash
attention (F-7, `project-docs/03-established-facts.md`). Switch to `flash_attention_2` (already
a constructor argument on `DiffuLLaMABackend`) before measuring anything. Then time one
`generate_samples` call at `L ∈ {256, 512}` and `T ∈ {32, 64}` steps, both with and without the
flash-attention fix, and **replace the arithmetic estimate in
`docs/02-experiment-plan.md` §8 with the measured number** — that section is explicitly labeled
"arithmetic, not measurement" (F-17) and exists to be overwritten by this step.

### 3.4 The correctness check that must pass before any number is trusted

Run 10 repair tasks end to end (`docs/04-build-phases.md` Phase 3's acceptance criteria):

- at least one problem that fails at depth 0 must pass at depth ≤ 5;
- **the diff between depths must be confined to the masked span** — assert this
  programmatically, do not eyeball it. If tokens outside the mask changed, `src_mask` is wired
  wrong and every subsequent number is meaningless. This is exactly the class of silent failure
  H-1 and H-3 already caught during the CPU build (`deploy.md` §6) — the real model gives the
  bug a new surface to hide on, not a new kind of bug to worry about.

### 3.5 Get the real benchmark data flowing

Swap the placeholder fixtures for the real corpora, per each benchmark module's own docstring
in `egr/benchmarks/`: HumanEval+ (164), MBPP+ (378), and the real HumanEvalFix split. Re-run
the Phase 0/1 hazard tests that reference "every task the benchmark yields"
(`egr/tests/test_verify.py::test_canonical_passes_and_mutant_fails_for_every_mutants_task` and
`test_every_mutants_task_has_at_least_one_test`) against the real corpus before trusting
anything downstream — H-10's failure mode (an empty rendered test suite silently passing) is
exactly the kind of thing that looks fine on 6 fixtures and breaks on problem #113 of 542.

### 3.6 Get a real confidence baseline (unblocks the localization gate, Phase 2's real result)

`egr/backend.py::MockBackend.confidence()` currently returns a constant, uninformative score —
there is no real model behind the current `confidence` column in
`project-docs/02-decision-log.md` P-6. Per `docs/04-build-phases.md` Phase 2: load **DiffuGPT-S
(124M)**, which fits in ~4 GB and can stay GPU-free if needed, specifically for this baseline.
Re-run `python -m egr.cli localize` and treat *that* table, not the current one, as the actual
gate result the project should act on.

### 3.7 The headline run — the repair curve (Phase 4)

Once 3.1–3.6 are done: `HumanEvalFix-Python + 200 mutants × 6 policies × 1 seed`, then 3 seeds
on the headline comparison. Budget ~3 GPU-hours for the first curve, ~14 for the 3-seed
version (`docs/04-build-phases.md` Phase 4) — but 3.3 already told you to replace this estimate
with a measured number before relying on it for scheduling. Report `ours` vs each of
`resample`, `static`, `confidence`, `random` with McNemar's exact test, and report the
overfitting gap (H-7) and the `HARNESS_ERROR` count as their own row, alongside.

**Also at this step: re-run the novelty check (H-14).** CDC itself was found three months after
this project started; the project's own decision principle is to re-check at every milestone,
not once. Search specifically for: execution-grounded remasking, traceback-conditioned diffusion
decoding, iterative repair with dLLMs, and any CDC v2, before writing up Phase 4's result.

### 3.8 Track B and the second backbone (Phase 5)

Seed the candidate from the model itself rather than a given buggy program, on both DiffuLLaMA
and Dream-Coder-v0-Instruct-7B. Two things need actual code that does not exist yet, not just
configuration: (a) how Track B's generated seed enters `loop.py`, which currently always
verifies `task.seed_program` and has no "generate depth 0 from a prompt" path; (b) a
`DreamCoderBackend` implementing the same `DiffusionBackend` protocol as `DiffuLLaMABackend` —
per P1, this should be "one class", but it does not exist yet, and Dream needs an *explicit*
decoding configuration (`alg="entropy"`, not the shipped default `alg="origin"`, F-10) or it
silently benchmarks random-order decoding.

### 3.9 Ablations (Phase 6)

Scope ladder (answers Q-8 — does CDC's Parent+Leaf optimum, measured on a security benchmark,
transfer to functional bugs?), the annotation channel on/off, the slack sweep (quantifies how
much room a targeted repair actually needs, and is where the fixed-length-constraint
limitation found during the CPU build — `deploy.md` §5 row 6 — gets turned into a number
instead of a caveat), and steps-to-convergence.

## 4. Definition of done for the GPU phase

Unchanged from `docs/04-build-phases.md`'s own "Definition of done for the POC", restated here
because it is the actual finish line for this deployment plan, not a separate one:

1. Every hazard test in `docs/03-hazards.md` still passes against the real backend, not just
   `MockBackend`.
2. The localization study (§3.6 above) has a result with a real confidence baseline, and the
   go/no-go in `project-docs/02-decision-log.md` P-6 is updated from "preliminary" to final.
3. A repair curve exists for Track A on the real HumanEvalFix, over ≥ 4 policies at matched
   budget.
4. `ours` vs `static` and `ours` vs `resample` are reported with McNemar tests and effect
   sizes — **whatever the direction of the result.**
5. Every number in the write-up traces to a JSONL record regenerable from a single CLI command
   (`egr/report.py`'s schema, already implemented and unchanged by any of this).
