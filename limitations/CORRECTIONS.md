# Corrections and version notes — `limitations/`

Additive record, per decision principle 4 in `project-docs/07-decision-tree.md` ("additive over
destructive"). Nothing in `output/` or `data/` has been edited. Read this before citing any limitation by
its L-number.

**Opened:** 2026-09-21.

---

## C-1 — ⚠️ Two incompatible runs exist, with clashing L-numbers

Two different outputs of this pipeline are in circulation.

| | **v1 — committed** | **v2 — circulated** |
|---|---|---|
| Location | `output/limitations_and_research_problem.md`, commit `a131e13` | **not in the repository**; pasted into an agent session on 2026-09-21 |
| Items | 10 | 12 |
| Provenance style | category tags, e.g. *(Methodological — Inappropriate Method)* | agent-source tags, e.g. *Agent sources: EXT-6, ANA-1, CIT-1* |
| Judge scores | not reported in this form | extractor 84 / analyzer 87 / reviewer 86 / citation 87 |
| RAG audit trail | not reported in this form | 95 chunks (48 cited-in, 47 cited-by) narrowed to 13 after re-ranking; 33 raw limitations consolidated to 12 |

**The numbering is not compatible.** The most dangerous collision: **v1's `L8` is v2's `L6`.**

| v2 (circulated) | v1 (committed) | Topic |
|---|---|---|
| L1 | L1 | annealing dropped at 7B |
| L2 | L2 | shift operation under-scrutinised |
| L3 | L3 | `[MASK]` token confounded with size |
| L5 | L4 | instruction tuning deferred |
| **L6** | **L8** | **answer-selection headroom / hit@3** |
| L7 | L6 | undertrained, no convergence shown |
| **L8** | **L5 + L9 merged** | unfair infilling comparison + code-generation claim vs evaluation |
| **L9** | **L7** | efficiency not compute-normalised |
| L10 | L10 | two AR families only |
| L4, L11, L12 | — | new in v2 (proxy-task validity; statistical rigor; data provenance) |

**Always say which run you mean.** `project-docs/10-inference-time-direction-2026-09-21.md` uses **v2**
numbering throughout.

**Outstanding actions:** commit v2 alongside v1 rather than over it; establish whether v2 was intended to
supersede v1 or is a parallel re-run. Tracked as **P-6** in `project-docs/02-decision-log.md`.

---

## C-2 — ❌ Factual error in v2's L8: the infilling number

**v2 L8 states:**

> "the only code evaluation is HumanEval single-line infilling with prefix and suffix supplied, and that
> number belongs to a separately trained Diffu-CodeLLaMA rather than the released checkpoint."

**The second clause is false.**

From `data/anchor_fulltext.txt` L452–462 — Table 1, "Infilling / Code" column, pass@1 in %:

| Model | Code |
|---|---|
| **DiffuLLaMA 7B** | **15.5** |
| LLaMA2 (prefix only) | 1.7 |
| DiffuGPT-M | 2.9 |
| GPT2-M (prefix only) | 2.6 |
| DiffuGPT-S | 0.3 |

The released checkpoint has its own number. **Diffu-CodeLLaMA's 0.76 is Table 8**, in the appendix — a
separate experiment finetuning CodeLLaMA on 100M tokens of Starcoder, alongside CodeLLaMA FT-SPM 0.80 and
FT-PSM 0.74.

Protocol, Appendix C.3 (`data/anchor_fulltext.txt` L1462): `openai/human-eval-infilling`, **1033 test
cases**, evaluated by pass@1.

**What survives in L8, and is correct:** the self-flagged unfair comparison (the paper withholds suffix
information from AR baselines, notes it "might result in an unfair comparison", and never runs a corrected
version), and the fixed-span-length assumption.

**Note on framing.** The honest version of the criticism is not that the number belongs to another model —
it is that **15.5% is a single-line fill with gold prefix *and* suffix supplied**, which is a much narrower
task than the "code generation" capability claimed in the abstract and introduction. That criticism stands
on its own and does not need the false clause.

---

## C-3 — 🔎 The C-2 error has a shared upstream source that has not been traced

The same false claim was found by `project-docs/09-p1-direction-analysis-2026-09-19.md` §1 row 4 in three
documents on the `plan/execution-grounded-repair` branch — `egr/README.md` §2,
`docs/02-experiment-plan.md` §2, and `novelty.md` §3 — written before this pipeline was run.

This pipeline then generated the same error independently on 2026-09-21.

**Two independent reproductions of the same specific error indicate a shared upstream source**, not two
coincidental slips. Candidates: a contaminated chunk in `data/` (the RAG corpus), an agent prompt carrying
the claim, or a secondary source that both pipelines retrieved.

**The source has not been identified. Until it is, other claims originating from it are suspect.**
Tracked as **Q-16** in `project-docs/04-open-questions.md`.

This is a live instance of decision principle 1 in `project-docs/07-decision-tree.md` ("verify against the
primary source when a claim is load-bearing") and of the pattern in `project-docs/05-mistakes-and-bugs.md`
§A ("five separate errors traced to trusting a summary").

---

## C-4 — ⚠️ v1's L8 (= v2's L6) compares against the wrong baseline

**v1 L8 states:** "hit@3 substantially exceeds single-answer accuracy across every reasoning benchmark
(e.g., 57.7 vs. 23.6 on SATMath)".

`23.6` is **DiffuLLaMA-FS** (few-shot). The paper's own best single-answer method is **DiffuLLaMA-SC**
(self-consistency, majority vote of 3) at **27.7**.

The headroom figures against SC, from `data/anchor_fulltext.txt` L568–580:

| Benchmark | SC | hit@3 | headroom |
|---|---|---|---|
| MAWPS | 33.1 | 40.8 | +7.7 |
| SATMath | 27.7 | 57.7 | **+30.0** |
| TriviaQA | 26.0 | 34.1 | +8.1 |

This is not a factual error — hit@3 does exceed FS — but it is the wrong reference point for any
follow-up work. **Anything claiming to improve on the paper must beat SC, not FS.** Beating 23.6 when the
paper itself reports 27.7 would be a strawman.

Both runs also understate the gap's significance in one respect: the paper **already deployed
self-consistency**, and it recovered only 1.8 / 4.1 / 5.1 points of the available 7.7 / 30.0 / 8.1. On
SATMath the standard selector captures one seventh of the headroom. See
`project-docs/03-established-facts.md` F-26.
