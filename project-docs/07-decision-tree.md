# Decision tree — how we got here

The reasoning chain behind every major fork, so nobody re-walks a path that was already closed.
Diagrams render natively on GitHub.

---

## The whole project, top to bottom

```mermaid
flowchart TD
    A[Course requires: anchor paper<br/>NLP or CV, GenAI, recent] --> B{TCS-provided topics<br/>or independent paper?}
    B -->|independent| C[Reject best-paper winners:<br/>too much follow-up work]
    B -.->|not chosen| B2[7 topics from Manasi Patwardhan's<br/>TCS group — closer mentorship]
    C --> D[Survey 50 papers<br/>15 survive gap-verification]
    D --> E[ANCHOR: DiffuLLaMA<br/>ICLR 2025]
    E --> F{Which enhancement<br/>direction?}

    F --> G[Option C:<br/>Structure-Guided ReMasking]
    F --> H[Option B:<br/>Adaptation audit]
    F --> I[Option E:<br/>Scientific infilling]

    G --> G1{Novelty check}
    G1 -->|CDC preempts mechanism| G2[Mechanism gone.<br/>Reframe on trace-guided post-hoc repair]
    G2 --> G3[Fixed-budget policy study:<br/>does execution evidence<br/>beat simpler selectors?]

    H --> H1{Novelty check}
    H1 -->|verified open| H2[Annealing dropped at 7B,<br/>validated only at 124M/355M,<br/>gain GROWS with scale]
    H2 --> H3[RECOMMENDED]

    I --> I1{Novelty check}
    I1 -->|molecules/proteins crowded| I2[NARROW — rejected]

    style H3 fill:#2d5016,color:#fff
    style G3 fill:#5c4813,color:#fff
    style I2 fill:#5c1616,color:#fff
```

---

## Fork 1 — anchor paper selection

**Question:** independent paper, or one of TTV's seven pre-defined topics?

| Path | Reasoning | Outcome |
|---|---|---|
| TTV's seven topics | All traceable to Manasi Patwardhan's TCS group — one cites her DBRouting paper by name, two mirror her own 2024-25 papers. Likely closer mentorship, possible data access. | **Not chosen** |
| Independent paper | Freedom to pick on technical merit. | **Chosen** → DiffuLLaMA |

**Sub-decision: exclude best-paper-award winners.** Flagship award winners attract the heaviest follow-up
work, so an open gap is unlikely. A 50-paper survey produced 15 gap-verified survivors and 35 rejects, each
reject naming the paper that closed its gap.

> **If revisiting:** the TCS path is still available and would satisfy the course equally. The tradeoff was
> mentorship vs independence, not quality.

---

## Fork 2 — the direction decision (⏳ still open)

```mermaid
flowchart TD
    START{Pick a direction} --> B[Option B<br/>Adaptation audit]
    START --> C[Option C<br/>Structure-Guided ReMasking]
    START --> E[Option E<br/>Scientific infilling]

    B --> B1[Q1: annealing dropped at 7B?]
    B --> B2[Q2: mask token reuse?]
    B --> B3[Q3: any-order vestigial after adaptation?]
    B1 & B2 & B3 --> BV[3 independent shots.<br/>One dying still leaves a paper.]

    C --> C1[Needs: trace-guided AST/def-use spans,<br/>sandbox, offset mapping]
    C1 --> C2[Compare random, window,<br/>confidence, static-only]
    C2 --> CV[Viable only if trace evidence<br/>beats simpler selectors]

    E --> E1[Molecules/proteins:<br/>crowded, purpose-built models]
    E1 --> E2[Scientific text: partly absorbed<br/>by an existing eval paper]
    E2 --> EV[Claim thinner than B's]

    style BV fill:#2d5016,color:#fff
    style CV fill:#5c4813,color:#fff
    style EV fill:#5c1616,color:#fff
```

### Why B is recommended

| Criterion | B | C |
|---|---|---|
| Course fit | **Exact** — *is* reproduce-then-interrogate the anchor | Tangential to the anchor |
| Novelty grounding | A **quotable inconsistency** in the anchor's own text | Follow-up to a preprint |
| Engineering risk | Low — training runs and evaluation | **Very high** — CPG, tracing, sandbox, offset mapping |
| Silent-failure surface | Minimal | **Two known** (off-by-one, `alg="origin"`) |
| Result value | **Interesting either way** | Weak if confirmatory |
| Failure tolerance | **3 independent shots** | 1 claim, already dented |
| Preemption risk | Low, cheap to re-check | **Already realised** (CDC) |

### Why C is not dead, just harder

The execution-grounded seeding gap is **verified genuinely open** — no paper anywhere seeds a structural
remask from a failing test or traceback. And CDC has no confidence baseline at matched budget, so that
comparison exists nowhere.

The honest framing if refined C2 is chosen:

> *"CDC established static-analysis-witness-guided remasking, while CDLM studies confidence-guided revision
> after correction-oriented post-training. We ask whether actual failing-test traces improve fixed-budget,
> post-hoc repair location selection for a frozen diffusion code model."*

Defensible. Visibly a follow-up. Should be written as one.

---

## Fork 3 — how the litreview extension was scoped

```mermaid
flowchart TD
    A[REPORT.md flags keyword scorer<br/>as the #1 limitation] --> B[Replace with real<br/>semantic judgment]
    B --> C{Overwrite the<br/>submitted files?}
    C -->|No| D[Survey cites by POSITION.<br/>Swapping refs desyncs every citation]
    D --> E[Keep in separate<br/>*_llm_judged files]
    E --> F[F1 0.222 to 0.272<br/>zero off-topic papers]
    style F fill:#2d5016,color:#fff
```

**The judgement call:** improving a deliverable is worthless if it silently breaks it. The generated survey
references `[1]`–`[48]` by position against the original `reference_brief.txt`; substituting a better
reference set would leave every citation marker pointing at the wrong paper. Additive, not destructive.

---

## Fork 4 — rejected directions, and why

```mermaid
flowchart LR
    A[Bet 1: multi-GPU decode] --> A1[Premise FALSE:<br/>AR models do split via TP]
    A1 --> A2[dInfer published it<br/>at batch size 1]
    A2 --> AX[REJECTED]

    B[Bet 2: dual-mode serving] --> B1[FLARE states the thesis<br/>near-verbatim, 3 months prior]
    B1 --> BX[REJECTED]

    C[Bet 3: do dLLMs plan?] --> C1[Already done twice,<br/>at 100B scale]
    C1 --> C2[BUT: only from-scratch models<br/>studied, never adapted ones]
    C2 --> CX[NARROWED, folded into B]

    D[Bet 5: cost accounting] --> D1[Premise decayed;<br/>wall-clock is standard now]
    D1 --> DX[REJECTED]

    style AX fill:#5c1616,color:#fff
    style BX fill:#5c1616,color:#fff
    style CX fill:#2d5016,color:#fff
    style DX fill:#5c1616,color:#fff
```

**Note on bet 1:** the hardware objection was **partially retracted** once 2× RTX 6000 Pro Blackwell plus
Sharanga access was confirmed — the experiment is runnable. The rejection stands on the false premise and
on dInfer/Sangam having published the core claim. Two GPUs still cannot support a scaling argument, which
is the actual contribution claimed.

---

## The meta-decision: keep audit work on a throwaway branch

```mermaid
flowchart TD
    A[Large volume of<br/>audit + verification work] --> B{Where does it live?}
    B -->|main| C[Pollutes a<br/>submission-ready repo]
    B -->|throwaway branch| D[audit/opus-review]
    D --> E{But findings must<br/>survive deletion}
    E --> F[Copy durable conclusions<br/>to project-docs/ on main]
    style F fill:#2d5016,color:#fff
```

This is why `project-docs/` exists on `main` and `audit/` does not.

---

## Decision principles this project has converged on

1. **Verify against the primary source when a claim is load-bearing.** Five separate errors traced to
   trusting a summary — see `05-mistakes-and-bugs.md` §A.
2. **Record retractions rather than editing them away.** The bet-1 hardware objection and the Fig. 8(b)
   framing were both corrected in place, with the correction visible.
3. **Label unverified claims as unverified.** Where a search did not complete, that is stated rather than
   presented as coverage.
4. **Additive over destructive.** Improvements go in new files when existing deliverables depend on the
   old ones.
5. **Re-check novelty at every milestone, not once.** CDC was posted three months before it was found and
   nearly invalidated a semester's framing.
