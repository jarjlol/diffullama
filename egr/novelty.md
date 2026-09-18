# Novelty — where EGR sits relative to the anchor paper and CDC

One page, answering one question: **what exactly is new here, what is not, and why should
anyone believe the difference matters?** Read this after `README.md` §2–3, which cover the
same ground in narrative form — this document exists to make the lineage and the claim
checkable at a glance, in one place, rather than spread across a README, a decision log, and a
facts file.

---

## 1. The one-sentence claim

**Failure evidence from actually running a candidate program is a better anchor for choosing
what a diffusion LM should re-generate than either static structural analysis or the model's
own confidence — and nobody has measured this, at any budget, on any repair benchmark.**

That is the whole claim. It is deliberately narrow. Everything below is about which parts of it
are genuinely open and which parts already belong to someone else.

## 2. The lineage in one picture

```
DiffuLLaMA (anchor paper, ICLR 2025)
  -> proves: a diffusion LM adapted from an AR checkpoint CAN condition on a frozen mask and
     infill a hole. Ships no whole-function code-generation number; its only code result is
     single-line infilling by a separately-trained variant.
       |
       v
CDC (arXiv:2605.16829, May 2026)
  -> proves: for CONSTRAINED GENERATION, choosing WHERE to remask by STATIC structural
     analysis (a code property graph) beats fixed-window and random remasking, with an
     interior-optimum neighbourhood scope (Parent+Leaf). No execution feedback anywhere.
     No repair loop. No confidence baseline.
       |
       v
EGR (this project)
  -> asks: does replacing CDC's STATIC witness with a DYNAMIC one -- the statement an
     ACTUALLY-FAILING TEST implicates -- anchor remasking better, when both are compared to
     each other AND to a confidence baseline, at matched budget, on a REPAIR benchmark?
```

Each arrow is a genuine narrowing of what is still unknown. EGR's job is to answer the
bottom question, honestly, including if the answer turns out to be "no."

## 3. What the anchor paper (DiffuLLaMA) established, and did not

| Established (cite it, don't re-claim it) | Not established (the actual opening) |
|---|---|
| An AR checkpoint can be adapted into a discrete diffusion LM that denoises with bidirectional attention over a partially-masked sequence, at 7B scale (F-1, F-2). | **No whole-function HumanEval pass@1 for the released `diffusionfamily/diffullama` checkpoint.** The only code result in the paper is HumanEval **single-line infilling**, and even that number belongs to `Diffu-CodeLLaMA`, a separately-trained variant (README §2). |
| Any subset of positions can be frozen (`src_mask`) while the rest is regenerated — the mechanical basis for "targeted edit" (docs/01-architecture.md §5.1, F-3). | Whether that mechanism is *useful* for iterative program repair specifically — the paper never runs an iterate-until-pass loop, or any repair benchmark, at all. |
| The model inherits an autoregressive shift (`canvas_position = returned_index + 1`, F-3) and a fixed-length infilling constraint. | Whether these implementation details matter enough to break a repair harness that doesn't account for them — they do (H-1, H-3; see `deploy.md` §6 for what broke and how it was found). |

**What EGR does with this:** takes the paper's *documented* strength — infilling — literally,
and builds the entire method around it. The repair step is defined as "here is a program with
a hole in it, fill the hole," which is exactly the operation the paper measured. This is why
the experiment plan (`docs/02-experiment-plan.md` §2) splits into a low-risk repair track and a
higher-risk generation track, rather than betting everything on the capability the paper never
reported.

## 4. What CDC established, and what it explicitly leaves open

CDC is the closest prior work, found late (three months after posting) and taken seriously: three
of this project's original novelty claims were deleted the day it was found (decision
D-2026-08-28-a). This is not a project that discovered CDC and pretended not to.

**What CDC owns, and EGR does not re-claim (F-12):**
1. "First to use program structure to select remasking positions" — CDC's MDFI operator does
   this: builds a partial code property graph mid-denoising, finds an offending node, lifts its
   AST/dataflow neighbourhood to token spans, remasks under a budget.
2. "First non-contiguous structure-derived remask set" — same mechanism, same prior claim.
3. "Structure beats fixed-window/random remasking" as a *novel finding* — CDC's own Fig. 8(b)
   already measures this, with a genuine interior optimum (Parent+Leaf beats both the looser
   use-def slice and the tighter token window, F-9).

**What CDC does not cover — the actual remaining gaps (F-13), each mapped to what EGR does about it:**

| Gap in CDC | What EGR does |
|---|---|
| No execution feedback anywhere — CDC's witness comes from **static** analysis only. | The witness comes from running the candidate under test and localizing via Ochiai spectrum-based fault localization (`evidence.py`, `localize.py`) — a **dynamic**, execution-grounded witness. |
| No confidence-based remasking baseline, at any budget, anywhere in the literature. | `ConfidencePolicy` (`policy.py`), built correctly on a fresh max-softmax forward pass rather than the model's sampled-token log-prob (H-5) — the comparison this project's own notes call "exists nowhere in the literature" (F-13). |
| Constrained *generation*, not repair — no iterate-until-pass loop, zero repair-benchmark numbers (no HumanEvalFix, DebugBench, QuixBugs, Defects4J). | `loop.py`'s `repair_loop()` — depth-bounded, seed-varies-by-depth, no-progress-escalates — evaluated on a repair benchmark (`humanevalfix.py`, currently a placeholder corpus pending network access; see `deploy.md` §5). |
| CDC's neighbourhood-scope optimum (Parent+Leaf) is measured on **CWEval, a security benchmark**. Whether it transfers to *functional* bugs is unmeasured — CDC routes functional correctness through a different mechanism entirely (F-9). | The scope ladder (`leaf` / `parent_leaf` / `function`) is a first-class, swept parameter (`localize.SCOPES`, `escalate()`), run against functional bugs specifically. This is **open question Q-8**, described in this project's own notes as "Option C's single best remaining research question" — EGR's scope ablation is designed to answer it, not assume an answer. |

## 5. The honest framing, stated once so it can be quoted

> CDC established that a *static* witness node can anchor structural remasking. EGR asks
> whether a *dynamic* witness — the statement implicated by an actually-failing test — anchors
> it better, and runs the matched-budget policy comparison CDC does not.

EGR is a follow-up to CDC and is written as one. It claims none of CDC's ground (§4's first
table). It claims the dynamic anchor, the repair loop, and the two missing baselines (§4's
second table) — nothing else.

## 6. What this project explicitly does NOT claim

Stated once, plainly, so it cannot be misread as understatement elsewhere:

- **Not a speed claim.** `model.py`'s forward pass processes the entire sequence every
  denoising step regardless of how many positions are masked — repairing 5 tokens costs
  exactly what regenerating 500 costs at equal step count (`model.py:139`, F-7, README §7). The
  only claim available is *quality* (and possibly *steps-to-convergence*, measured, not assumed).
- **Not a claim that DiffuLLaMA can write whole functions well.** §3 above is explicit that the
  paper never measured this; the repair track is designed to not need it, and the generation
  track (Track B) is flagged as the higher-risk part of the plan for exactly this reason.
- **Not a claim that CDC's structural mechanism is wrong or inferior "in general."** The claim
  is narrower and comparative: *for functional bugs, at matched budget, does a dynamic witness
  beat a static one, and does either beat a confidence baseline?* If the answer is "no, static
  is just as good," that is a genuine, reportable finding — see `docs/04-build-phases.md`
  Phase 4's acceptance criteria, which require reporting `ours` vs `static` "whatever the
  direction of the result."
- **Not yet an empirical result on the real corpus or the real model.** `deploy.md` records the
  current preliminary localization result (§4.1 there) and exactly what stands between it and
  a result this claim could actually be judged on — a placeholder 15-mutant fixture set, no
  GPU used, no real confidence baseline yet (`project-docs/02-decision-log.md` P-6).

## 7. Where this could still fail, and how the project would know

The three risks in `README.md` §10 are the falsification conditions for this whole document:
the backbone may be too weak to show a curve at all (§3 above is the mitigation, not a
guarantee); the loop can stall deterministically at low temperature (H-4, mitigated but not
eliminated by seed-varying and scope-escalation); and the shift/indentation issues are silent
by nature (H-1, H-2 — closed by tests, but a reason for caution about any *new* silent-failure
class the real model surfaces that a mock cannot). None of these are hedges added after the
fact — they are in the plan because a negative result on any of them is still the answer to
§1's question, and `docs/04-build-phases.md`'s Phase 4 acceptance criteria require reporting it
either way.
