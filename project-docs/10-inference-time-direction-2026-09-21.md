# Inference-time direction analysis — 2026-09-21

> **Status:** written input to decision **P-1**, produced after the team stated a hard constraint that
> narrows it (§1). It does not overturn `09-p1-direction-analysis-2026-09-19.md`; it answers a different
> question, namely *given that training is off the table, which limitation is worth doing*. Produced by an
> AI agent session run by Aaditya on 2026-09-21 against `main` @ `066d8f2`. Confidence labels follow
> `03-established-facts.md`: 🟢 PRIMARY (read the source or ran the code this session), 🟡 SECONDARY
> (second-hand or via a fetch summariser), 🔴 WEAK (arithmetic, recollection, or absence-of-evidence).

## 0. Bottom line

1. **Seven of the twelve limitations require adaptation training and are therefore out** under the stated
   compute constraint: L1, L2, L3, L4, L5, L7, L10 (§3). This is the constraint doing useful work, not a
   compromise — it removes exactly the options the team cannot execute.
2. **Recommend L6 (the answer-selection headroom) as the primary direction.** It is inference-only, it has
   a published reproduction target *and* a published ceiling in the anchor's own Table 2, and it directly
   contests an attribution the authors make about their own model (§4).
3. **Recommend the adaptation-residue audit as the secondary, novelty-carrying axis** (§5). It is
   inference-only on released checkpoints, fits one 96 GB card, and `03-established-facts.md` F-15 already
   verifies that the gap is open.
4. **L8 is viable but the version circulated on 2026-09-21 contains a factual error** that must be
   corrected before it is cited anywhere (§6). It is the same error `09-...md` §1 row 4 caught in the EGR
   documents, which means the error has a shared upstream source rather than being a one-off.
5. **L9 is already-rejected ground** (Bet 5, `07-decision-tree.md` Fork 4). **L11 is half training.**
   **L12 is self-blocking.** (§7)
6. Two sampler defects block every option in §4 and §5 and must be fixed first (§8). Both are GPU-free
   work, which is what the four members without workstation access can take.
7. **No novelty search was run for L6 this session** (§12). Treat §4 as an unaudited hypothesis until one
   is run. This is the largest single gap in this document.

---

## 1. The constraint that drives this analysis

The team has stated that compute is the binding constraint: the lab workstation is shared and contended,
and there is not enough of it for adaptation training or for AR-to-diffusion conversion. Inference on
released checkpoints is what the project can actually execute.

This is consistent with `01-project-brief.md` ("plan for **one** 96 GB card, not two") and with the
circulated limitations document's own scoping note, which states that a faithful reproduction of even the
124M/355M ablation grid is on the order of a thousand GPU-hours, and names the inference-only
adaptation-residue audit as "the cheaper entry point and the one that can produce a result without any
training run."

**Consequence for P-1.** `09-p1-direction-analysis-2026-09-19.md` recommended Option B at about 75%
confidence. Option B is a training study: its §3 design asks for roughly **160 GPU-hours on one card,
three to four weeks of wall-clock at half availability**, with checkpoint-and-resume mandatory. If that
budget is not available, Option B is not merely harder, it is unexecutable, and the recommendation in that
document does not survive the constraint. This document does not overturn its *reasoning* — the annealing
cell is still genuinely empty, and F-1, F-22 and Q-9 stand unchanged — it records that the team cannot
occupy that cell.

That is a legitimate reason to close a direction, and it should be logged as a resource decision rather
than as a disagreement about the science. If compute later becomes available, §11 says what to revisit.

---

## 2. ⚠️ Numbering hazard — two incompatible limitations runs exist

There are **two different outputs** of the `limitations/` multi-agent pipeline in circulation, with
**different item counts and incompatible numbering**. Citing "L8" without saying which run is ambiguous
and will eventually cause someone to argue against the wrong limitation.

| | Committed run (v1) | Circulated run (v2) |
|---|---|---|
| Location | `limitations/output/limitations_and_research_problem.md`, committed in `a131e13` | not in the repo as of this session; pasted into an agent session on 2026-09-21 |
| Item count | 10 | 12 |
| Judge scores reported | not in this form | extractor 84 / analyzer 87 / reviewer 86 / citation 87 |
| RAG audit trail | not in this form | 95 chunks (48 cited-in, 47 cited-by) narrowed to 13 after re-ranking; 33 raw limitations consolidated to 12 |
| Provenance tags | category tags, e.g. *(Methodological — Inappropriate Method)* | agent-source tags, e.g. *Agent sources: EXT-6, ANA-1, CIT-1* |

**Mapping, as far as it can be established by reading both:**

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

**This document uses v2 numbering throughout**, because that is the run the team is working from. Anyone
reading the committed v1 file must apply the mapping above.

Resolving this properly means committing v2 **alongside** v1 rather than overwriting it, per decision
principle 4 in `07-decision-tree.md` ("additive over destructive"). It should also be established whether
v2 was intended to supersede v1 or is a parallel artifact from a re-run — that is not known (§12).

---

## 3. Triage of L1–L12 by compute class

| Item | Topic | Compute class | Verdict under the constraint |
|---|---|---|---|
| L1 | annealing dropped at 7B | **training** — needs a paired adaptation run | out |
| L2 | shift operation | **training** | out |
| L3 | `[MASK]` token confound | **training** (the token-frequency count is CPU-only, but that is a footnote, not a project) | out |
| L4 | proxy task never validated against real adaptation | **training, the most expensive of all** — needs *both* the proxy and the real run | out |
| L5 | instruction tuning deferred | **training** | out |
| **L6** | **answer-selection headroom** | **inference-only** | **primary recommendation (§4)** |
| L7 | unconverged lower bound | **training** | out |
| **L8** | **code / infilling claims** | **inference-only** | viable fallback; contains an error (§6) |
| **L9** | **efficiency not compute-normalised** | **inference-only** | previously rejected (§7) |
| L10 | one route, two families | **training** | out |
| L11 | statistical and reproducibility rigor | **mixed** — seeds/variance on ablations is training; decoding-config sensitivity is inference | thin standalone; fold into L6 (§7) |
| L12 | data provenance and contamination | **CPU-only** | self-blocking (§7) |
| — | **adaptation-residue audit** (v2 Research problem, second linked question) | **inference-only, one GPU** | **secondary recommendation (§5)** |

Seven of twelve are eliminated by the constraint. That is the useful outcome of stating the constraint
plainly rather than treating every limitation as equally available.

---

## 4. Primary recommendation — L6, the answer-selection headroom

### 4.1 The anchor's own numbers 🟢

Read directly from the anchor's extracted full text (`litreview/data/anchor_fulltext.txt` L568–580,
Table 2: "Performance on math/QA benchmarks", exact-match accuracy between gold labels and predictions):

| Setting | MAWPS | SATMath | TriviaQA |
|---|---|---|---|
| LLaMA2 | 63.5 | 24.5 | 45.4 |
| DiffuLLaMA-ZS (zero-shot) | 9.7 | <1 | 18.5 |
| DiffuLLaMA-FS (few-shot) | 31.3 | 23.6 | 20.9 |
| DiffuLLaMA-SC (self-consistency, majority vote of 3) | 33.1 | 27.7 | 26.0 |
| **DiffuLLaMA-@k (hit rate, k = 3)** | **40.8** | **57.7** | **34.1** |
| DiffuLLaMA-CoT | 28.7 | 9.5 | — |

ICL setting: 4-shot on math tasks, 2-shot on TriviaQA. Self-consistency follows Wang et al. (2023);
hit@k with k = 3 "measures whether any of the k predictions include the correct answer, serving as a
reference for the model's upper bound."

**Selection headroom at k = 3, measured against the paper's best single-answer method (SC):**

| Benchmark | SC | hit@3 | headroom | recovered by majority vote (SC − FS) | fraction of headroom captured |
|---|---|---|---|---|---|
| MAWPS | 33.1 | 40.8 | **+7.7** | 1.8 | 23% |
| SATMath | 27.7 | 57.7 | **+30.0** | 4.1 | 14% |
| TriviaQA | 26.0 | 34.1 | **+8.1** | 5.1 | 63% |

⚠️ The committed v1 limitations text quotes this as "57.7 vs. 23.6 on SATMath", i.e. against **FS**, not
against SC. Both are defensible reference points but they are different claims. **Any write-up must beat
SC (27.7), not FS (23.6)** — SC is the strongest single-answer selector the paper itself reports, and
beating the weaker number would be a strawman of exactly the kind F-4 warns about in a different context.

One further observation worth stating in a paper: on SATMath, DiffuLLaMA-SC (27.7) already **exceeds** the
LLaMA2 model it was adapted from (24.5), and hit@3 (57.7) exceeds it by more than a factor of two.

### 4.2 Why this is a real gap and not a solved one

Two points that the limitations write-up understates:

1. **The authors already applied the obvious fix, and it mostly failed.** Majority-vote-of-3 *is*
   self-consistency, and the paper deploys it explicitly. It recovered 1.8 / 4.1 / 5.1 points of the
   7.7 / 30.0 / 8.1 available. On SATMath — the largest gap by a factor of four — the trivial selector
   captures **one seventh** of the headroom. So the position is not "nobody tried selection"; it is "the
   standard selector leaves 26 points on the table on the benchmark where most of the headroom lives."
   That is a stronger gap than "uninvestigated", not a weaker one.
2. **k = 3 is a very small budget and the curve past it is unpublished.** No hit@k for k > 3 is reported
   for any adapted diffusion LM that this session is aware of. The ceiling may keep rising with k; the
   shape of hit@k versus k is itself an unreported measurement, and a cheap one.

### 4.3 The falsifiable claim

The paper writes, verbatim (`anchor_fulltext.txt` L599–601):

> "we report the hit rate results in generated candidate answers, highlighting the model's potential to
> produce the correct answer. This reveals that the current model exhibits high uncertainty about its
> responses, leading to **temporarily suboptimal** performance."

"Temporarily" is an attribution: it implies more training closes the gap. **That attribution is never
tested.** The measurement localises the deficit in *answer selection*, not in *answer knowledge* — the
correct answer is demonstrably present in the candidate set — but no calibration analysis, reranking or
verifier is attempted anywhere in the paper.

> **RQ.** On the frozen released DiffuLLaMA-7B checkpoint, how much of the measured hit@k − accuracy gap
> is recoverable by inference-time selection alone, at matched sample budget?
>
> **H1 (selection).** A training-free selector closes a large share of the gap. The deficit is selection,
> not knowledge, and the authors' attribution of the shortfall to undertraining is wrong.
>
> **H2 (knowledge).** No training-free selector meaningfully beats majority vote. The gap reflects genuine
> answer uncertainty that selection cannot exploit, and the authors' attribution is supported.

Both outcomes are reportable. H2 is a clean negative result that *confirms* an author claim by testing it,
which is a legitimate contribution and is not a project failure. The project should decide now that it
will report H2 if it gets H2.

### 4.4 Course-format fit

This is where L6 separates from every decoding-time proposal considered so far, including EGR. The
objection recorded in `09-...md` §2.2 problem 2 — that an inference-time controller "never touches
adaptation, the reproduction it would perform is a 15.5% infilling number, and 'beat the paper's
baseline' has no referent" — does **not** apply here:

| Course stage | What L6 supplies |
|---|---|
| 2 — reproduce the paper's results | Table 2, six rows, released checkpoint, inference-only. A real reproduction target. |
| 3 — idea enhancement | The selection-versus-knowledge question above, plus the residue axis in §5 |
| 4 — beat the paper's baseline | 33.1 / 27.7 / 26.0 are the anchor's own published numbers, and 40.8 / 57.7 / 34.1 are the published ceiling. Unambiguous, and the maximum winnable amount is known in advance. |
| 5–6 — writing and review | unaffected |

The anchor is also **not interchangeable** here, which was the other half of the EGR objection. The claim
being contested is a claim DiffuLLaMA made about DiffuLLaMA, using DiffuLLaMA's own table.

### 4.5 Design

**Keep it training-free.** A trained verifier or reranker reintroduces the compute the constraint removed
and weakens the "frozen checkpoint" framing. Hold it as a stretch arm, reported separately and labelled as
not training-free if attempted at all.

Candidate selectors, in rough order of cost:

1. Majority vote at k = 3 — **reproduce the paper's SC row first**. If this does not land near
   33.1 / 27.7 / 26.0, nothing downstream is interpretable. This is the gate.
2. Majority vote at k = 8, 16, 32 — establishes the hit@k curve and separates "more samples" from "better
   selection". These are different claims and must not be merged.
3. Confidence-margin selection using max softmax probability (**not** `x0_scores` — see §8).
4. Length and format normalisation of the candidate score.
5. Answer-consistency clustering: cluster semantically equivalent answers before voting, rather than
   exact-match voting. On math benchmarks, exact-match voting fragments over formatting.
6. Per-sample aggregate uncertainty derived from the denoising trajectory — requires §8.1.
7. Step-count ensembling: candidates generated at different `diffusion_steps`, exploiting the sharp
   step-count sensitivity the paper's own Figure 3 shows. This also doubles as the L11 robustness figure
   (§7).

**Budget control.** Every arm must be reported at matched *sample* budget (k) and matched *denoising step*
budget. Per F-5 🟢 the sampler forwards the entire sequence at every step regardless of how many positions
are masked, so k samples cost k times the forwards and there is **no efficiency claim available** — only a
quality claim. Report GPU-seconds per returned answer alongside accuracy so that a quality gain cannot
hide a larger test-time budget.

**Baselines that are mandatory.**

- Majority vote at matched k — the paper's own SC, and the number to beat.
- Oracle hit@k — the published ceiling, reported as an upper bound, never as a result.
- **Random selection from the candidate set** — the floor. Without it, a weak selector can look like a win
  simply because any selection beats none.

### 4.6 Honest risks

- **TriviaQA's headroom is small (8.1) and 63% of it is already captured by majority vote.** The result
  will most likely be carried by SATMath. A paper resting on one benchmark of three is weaker; say so
  rather than averaging across benchmarks to hide it.
- **SATMath's base rate is low and its item count was not checked this session.** A 30-point gap on a
  small test set may carry a wide confidence interval. Verify the item count before quoting any interval.
  🔴
- **The selection gain may be partly an artifact of exact-match scoring.** If hit@3 counts a candidate
  correct on a formatting technicality that no selector could detect, part of the headroom is illusory.
  Inspect candidates by hand before claiming recoverability, and report what fraction of the hit@3 wins
  are formatting artifacts.
- **Reproduction may fail for reasons unrelated to the science** — decoding configuration for Table 2 is
  not fully stated in the paper, which is L11's complaint applied to this project's own critical path.

---

## 5. Secondary recommendation — the adaptation-residue audit

The v2 Research-problem section names two linked questions. The first (the recipe audit) is training and
is out. The second is inference-only:

> "determine whether models adapted from autoregressive checkpoints retain measurable autoregressive
> inductive bias at inference relative to comparable models trained from scratch, since the recipe's
> purpose is precisely to remove that bias, and the base paper never verifies that it did."

### 5.1 The gap is already verified open in this repository

`03-established-facts.md` **F-15** 🟡 records that generation-order falsification is published at scale —
[2601.15593](https://arxiv.org/abs/2601.15593) across 58 benchmarks and 8 MDLMs up to 100B,
[2608.05687](https://arxiv.org/abs/2608.05687) as a causal commitment-order intervention, and
[2601.13228](https://arxiv.org/abs/2601.13228) on AR models rivalling diffusion at any-order generation —
and then states:

> "**But every one studies from-scratch models. None studies an *adapted* model** — that gap is open."

This is the rare case where the methodology is published, so the project does not have to invent metrics
and then defend them, while the population it is applied to is untested. That is a
replication-and-extension paper: a legitimate genre, and a far easier one to defend than a
novel-mechanism claim.

The anchor supplies a directly relevant hypothesis of its own (`anchor_fulltext.txt` L590):

> "We hypothesize that the adapted model retains some of the abilities from the base AR model."

The authors hypothesise residue and never measure it. That is the same shape as L6 — an author claim left
untested — which is what makes the two axes coherent as one paper.

This also revives, in inference-only form, what `07-decision-tree.md` Fork 4 recorded as Bet 3: *"do dLLMs
plan?"* → *"already done twice, at 100B scale"* → *"BUT: only from-scratch models studied, never adapted
ones"* → **NARROWED, folded into B**. Option B is now closed by compute, so the narrowed question needs a
new home. This is that home, and it needs no training.

### 5.2 Design

**Population:** adapted models — DiffuLLaMA-7B, Dream-7B, DiffuCoder-7B — versus from-scratch — LLaDA-8B.
Three-versus-one is thin, but it is close to the entire available population of open checkpoints. State
that as a limitation rather than pretending the design is balanced.

**Two confounds that must be stated, not hidden.** The adapted and from-scratch models differ in backbone,
tokenizer, training corpus and token budget as well as in adaptation. No inference-only design can
separate these. The honest claim is therefore *"adapted checkpoints show property X and the from-scratch
one does not"*, **not** *"adaptation causes X"*. Overclaiming causality here is the single most likely
reviewer objection and the easiest to avoid by writing the weaker sentence.

**Relevant prior facts already in the repository, both of which silently invert results if mishandled:**

- **F-10** 🟢 — Dream ships `"alg": "origin"` (the **random** branch) with `temperature: 0.0` and
  `steps: 512`. Any Dream baseline run without explicitly setting `alg="entropy"` silently runs
  random-order decoding. This is a silent failure mode of exactly the class this project has already been
  bitten by twice (F-3's off-by-one; F-4's confidence signal).
- **F-11** 🟡 — LLaDA unmasks *highest*-confidence-first; the config name `remasking='low_confidence'`
  refers to which tokens *stay* masked. Its default is also semi-autoregressive (`block_length=32`), which
  is itself a partial left-to-right ordering — if left on, the from-scratch control is partially
  autoregressive by configuration and the entire comparison is void.

These two facts are the reason this axis needs careful setup rather than being dismissed as "just
inference".

### 5.3 Compute 🔴

Four checkpoints at bf16: DiffuLLaMA 13.5 GB, LLaDA-8B 16.03 GB, Dream-7B 15.2 GB (all from F-16),
DiffuCoder-7B not verified this session. Run sequentially with one checkpoint resident at a time, each
fits one 96 GB card with large margin. Inference-only: no optimiser state, no gradient memory, no
checkpointing beyond caching outputs. This is the cheapest serious option available to the project.

### 5.4 How it composes with L6

They are independent experiments sharing infrastructure and a thesis: *the anchor makes claims about its
adapted model that it never verifies, and both are checkable at inference.*

If adapted models carry measurable AR residue, that is a **candidate mechanism** for why their answer
distribution is miscalibrated in the L6 sense. Stated as a hypothesis in the discussion this is worth
raising; stated as a conclusion it would be unsupported by either experiment. Keep the two results
separate in the write-up and let the discussion connect them, explicitly flagged as speculation.

---

## 6. L8 — viable, but the circulated version contains a factual error

### 6.1 The error 🟢

The v2 L8 text states:

> "the only code evaluation is HumanEval single-line infilling with prefix and suffix supplied, and that
> number **belongs to a separately trained Diffu-CodeLLaMA rather than the released checkpoint**."

**The second clause is false.** From `litreview/data/anchor_fulltext.txt` L452–462, Table 1 has an
"Infilling / Code" column and the released checkpoint has its own entry: **DiffuLLaMA 7B — Code 15.5**
(pass@1, %). The other entries in that column are LLaMA2 1.7 (prefix only), DiffuGPT-M 2.9, GPT2-M 2.6,
DiffuGPT-S 0.3. Diffu-CodeLLaMA's 0.76 is **Table 8** in the appendix, a separate experiment finetuning
CodeLLaMA on 100M tokens of Starcoder.

The protocol is Appendix C.3: HumanEval single-line infilling via the `openai/human-eval-infilling`
toolkit, **1033 test cases**, evaluated by pass@1 (`anchor_fulltext.txt` L1462).

### 6.2 Why this matters beyond one sentence

This is **the same error** `09-p1-direction-analysis-2026-09-19.md` §1 row 4 identified in three separate
EGR documents (`egr/README.md` §2, `docs/02-experiment-plan.md` §2, and `novelty.md` §3) on
`plan/execution-grounded-repair`. The limitations pipeline reproduced it independently on 2026-09-21.

An error appearing in two independently generated artifacts is not a coincidence — it has a shared
upstream source. Candidates: a contaminated chunk in the limitations RAG corpus, an agent prompt carrying
the claim, or a secondary source that both pipelines retrieved. **This should be traced**, because
anything else from that source is suspect.

It is a live instance of decision principle 1 in `07-decision-tree.md` ("verify against the primary source
when a claim is load-bearing") and of `05-mistakes-and-bugs.md` §A ("five separate errors traced to
trusting a summary"). The correct lesson is not "fix the sentence" but "find the source".

### 6.3 What survives in L8

Both remaining parts are sound, inference-only, and cite the paper correctly:

- The **self-flagged unfair comparison**: the paper withholds suffix information from AR baselines and
  explicitly notes this "might result in an unfair comparison", then never runs a corrected version.
- The **fixed-span-length assumption**: the evaluation supplies the span length, whereas infilling at
  flexible length and position without ground-truth positional data is the harder task. Relevant
  background: DreamOn [2602.01326](https://arxiv.org/abs/2602.01326) (ICLR 2026) addresses variable-length
  infilling for Dream-Coder and DiffuCoder but **not** for DiffuLLaMA, and
  [2609.02108](https://arxiv.org/abs/2609.02108) covers adaptive-length infilling for diffusion LMs. Both
  must be re-read in primary form before any novelty claim rests on them. 🟡

**L8 is the fallback if L6 fails its reproduction gate (§4.5 item 1).** It is ranked below L6 for two
reasons: 15.5% is a much weaker backbone number to build on than Table 2's 27.7–33.1, and the
"beat the baseline" target is correspondingly thinner. It has the advantage of a large test set (1033
cases) where L6's benchmarks are small.

---

## 7. L9, L11, L12 — verdicts

### L9 (efficiency not compute-normalised) — previously rejected, do not reopen

Inference-only and cheap, but `07-decision-tree.md` Fork 4 already rejected this as **Bet 5**:
*"premise decayed; wall-clock is standard now."* Reopening a settled rejection requires new evidence per
the decision-log rules, and this document has none to offer.

What is worth keeping is the **component**, not the project:

- **F-7** 🟢 — `inf_diffullama.py:24` defaults `_attn_implementation="eager"`, and with
  `attn_mask_ratio=1.0` `get_anneal_attn_mask` returns an all-zeros 4-D additive mask that is a
  mathematical no-op while blocking the flash-attention kernel. That is the same kernel the annealing
  removal was justified by, which is a genuinely quotable irony but still not a paper.
- **F-5** 🟢 — every denoising step forwards the whole sequence with no KV cache.

Both belong in L6's cost table (§4.5) as measured facts about what a sample actually costs. Fixing F-7
should also be done simply because it speeds up every experiment in §4 and §5.

### L11 (statistical and reproducibility rigor) — half training, thin standalone

The load-bearing half — seeds, variance estimates and significance tests on the ablations whose 2.1-to-2.5
point differences drive the recipe decisions — requires re-running those ablations, which is training and
therefore out.

The inference half is real but small: decoding configurations are unstated on both sides of key
comparisons, despite the paper's own Figure 3 showing sharp sensitivity to step count. A
decoding-sensitivity sweep on the released checkpoint is one good figure.

**Fold it into L6 as a robustness section.** It is directly relevant there: step-count ensembling is a
candidate selector (§4.5 item 7), and any accuracy claim in §4 needs its decoding configuration pinned or
it inherits the exact defect L11 identifies. Doing this also means the project does not repeat the
paper's own reporting failure, which is worth a sentence in the write-up.

### L12 (data provenance and contamination) — self-blocking

Genuinely CPU-only, and benchmark-contamination auditing is a recognised paper genre. But it is blocked by
its own premise: a contamination check requires the adaptation corpus, and L12's own complaint is that no
data manifest was released. The obstacle here is data access, not GPUs, so the compute constraint does not
rescue it.

A **partial** version is feasible — n-gram overlap between the evaluation benchmarks and a public
reconstruction of the corpus (SlimPajama and Starcoder subsets) — but a negative result then means only
"no contamination found in the part of the corpus we could reconstruct", which is a substantially weaker
statement and must be phrased that way rather than as "no contamination".

Not recommended as a primary direction. Reasonable as an appendix, or as a self-contained task for a
member without GPU access who wants something independent of the critical path.

---

## 8. Engineering prerequisites — two sampler defects block §4 and §5

Both verified by reading `model.py` this session. 🟢

### 8.1 The sampler unmasks uniformly at random, and exposes no confidence

`model.py:132`, inside the denoising loop:

```python
masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
```

with `p_to_x0 = 1/(t+1)` at `model.py:130`. Unmasking order is **uniform random** — not
confidence-ordered, not entropy-ordered. Consequences:

- Any confidence-based or entropy-based ordering baseline **does not exist in this repository** and must
  be built before it can be compared against. Nothing in `model.py` provides one.
- Per **F-4** 🟢, `x0_scores` (`model.py:116`, `model.py:143`) is the log-probability of the *sampled*
  token, not the maximum — the max-based form at `model.py:113` is commented out. It measures sampling
  luck, not model certainty. It is also **never returned**: `model.py:158` returns `x0` only.
- Trajectory-derived uncertainty features (§4.5 items 3, 6 and 7) therefore require changing what the
  sampler returns, not merely reading an existing output.
- Building the confidence baseline on `x0_scores` as-is would produce an unfairly weak comparison and bias
  every result in favour of whatever is proposed against it. **Use max softmax probability.** This is the
  specific trap F-4 was written to prevent.

### 8.2 Once a position commits, it can never be revised

`model.py:134`:

```python
maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)
```

`maskable_mask` is only ever cleared, never set. There is **no remasking path in the released sampler at
all**. F-6 🟢 records that `src_mask` is a genuine freeze primitive (`model.py:102`,
`maskable_mask = ~src_mask`), and F-3 🟢 records the off-by-one that governs any position-targeted
manipulation (`canvas_position = returned_index + 1`); but the *unfreeze* direction is new code in every
case.

This constrains L6 less than it constrained the decoding-time proposals considered earlier — L6 resamples
rather than revises — but it bounds any candidate-diversification scheme built on revision, and it should
be known before anyone designs one.

### 8.3 Who does this, and when

Both items are pure CPU and software work, testable against a 124M checkpoint or a mock backend. Per
`01-project-brief.md`, only Aaditya, Aryan and Arjun have workstation access; **this is exactly the
substantial GPU-free work the other four members can own.** Estimated one week.

It is on the critical path for both §4 and §5, so it should start **before** the direction is formally
ratified — it is useful under any inference-time direction and wasted under none of them.

---

## 9. Compute budget 🔴

Arithmetic, not measurement. Q-7 still applies, and one hour of benchmarking on the real card should
replace all of these numbers before anything is scheduled.

| Item | Estimate | Notes |
|---|---|---|
| L6 reproduction of Table 2 (3 benchmarks × 6 settings, k = 3) | small | MAWPS, SATMath and TriviaQA are small test sets; cost dominated by 7B inference at T denoising steps |
| L6 hit@k curve to k = 32 | roughly 10× the above | the dominant cost; k samples = k× forwards (F-5) |
| L6 selector arms | negligible marginal | selectors run over cached candidates |
| Residue audit, 4 checkpoints | moderate | sequential, one checkpoint resident at a time, ≤16 GB each |

**The single most important engineering decision: cache every generated candidate to disk.** Each selector
in §4.5 is a post-hoc function of the candidate set. If candidates are regenerated per arm, cost
multiplies by the number of arms for no scientific gain; if they are cached, the entire selector study
becomes CPU work that the four members without GPU access can run. Caching also makes the study
resumable, which `01-project-brief.md` makes a hard requirement under workstation contention.

Fixing F-7 (eager attention and the no-op 4-D mask) before the generation runs will reduce all of the
above, by an amount nobody has measured.

---

## 10. Paper viability — honest assessment

**This is an audit and measurement paper, not a method paper.** That should be stated plainly in any
submission rather than dressed up as a new technique.

**In favour:**

- Every claim is falsifiable and both outcomes are reportable (§4.3).
- The reproduction target (33.1 / 27.7 / 26.0) and the ceiling (40.8 / 57.7 / 34.1) are both published in
  the anchor's own table. The maximum winnable amount is known before the first run, which is unusual and
  valuable for planning.
- The §5 gap is verified open by the project's own F-15, with published methodology to reuse rather than
  invent.
- It runs on one contended GPU with no training, which matches the actual constraint.
- Thirty points of measured, unexplained headroom on the anchor's own headline analysis table is a strong
  hook for a workshop paper.

**Against:**

- Novelty is roughly **2 out of 3**. Test-time scaling, self-consistency and verifier-based selection are
  heavily worked in the autoregressive literature. The contribution is the *population* (adapted diffusion
  LMs) and the *specific contested attribution*, not the mechanism. A reviewer will reasonably ask why
  extending k from 3 to 32 on one checkpoint family is a paper — and the answer has to be §5, because the
  residue axis is what makes the work about *adaptation* rather than about sampling.
- The headline result may rest on one benchmark of three (§4.6).
- Neither axis produces a new method, so a venue seeking methodological novelty is a poor fit.
- §12: no novelty search has been run for L6. Until it is, the novelty estimate above is not evidence.

**The unresolved fork that should be settled before committing.** `09-...md` §4 lists, under "what would
change my mind", *"the instructor or industry collaborator stating a preference for a method paper over an
audit paper"*, and §5 lists the same item under what could not be verified. `04-open-questions.md` Q-1
already requires contacting TTV about the venue. **This is the third document to reach this fork, and it
is one email.** The framing recommended here is an asset if an audit paper is acceptable and a liability
if it is not, and the repository cannot settle it.

---

## 11. What would change this recommendation

- **Compute becoming available** at roughly 160 GPU-hours for the semester. Option B's reasoning in
  `09-...md` §3 is untouched by this document and would become live again immediately.
- **Failing the L6 reproduction gate** (§4.5 item 1): if Table 2's SC row cannot be reproduced within a
  reasonable margin on the released checkpoint, the direction is unsound and L8 becomes the fallback.
- **A published hit@k, calibration or test-time-selection study on adapted diffusion LMs.** Not searched
  for this session — see §12.
- **TTV or Manasi Patwardhan preferring a method paper**, per §10.
- **Discovery that the hit@3 headroom is largely a formatting artifact** (§4.6), which would collapse H1
  before any selector is built. This is checkable by hand on a few dozen candidates and should be checked
  early.

---

## 12. What could not be verified this session

- **No novelty search was run for L6.** Whether anyone has published test-time selection, calibration or
  hit@k scaling for *adapted* diffusion LMs is **unknown**, not established as open. This is the single
  largest gap in this document and must be closed before the direction is ratified. The project's own
  history is the argument: `07-decision-tree.md` principle 5 records that CDC was posted three months
  before it was found and nearly invalidated a semester's framing, and `09-...md` §2.2 records the EGR gap
  narrowing four times in four weeks. **Treat §4 as an unaudited hypothesis until a search is run.** 🔴
- **SATMath's item count**, and therefore the confidence interval on a 30-point gap. 🔴
- **DiffuCoder-7B's checkpoint size.** 🔴
- **The upstream source of the L8 error** (§6.2) — identified as shared between two pipelines, not traced.
- **Every compute number in §9** — arithmetic on unmeasured throughput.
- **Whether the v2 limitations run supersedes v1 by intent** or is a parallel artifact from a re-run (§2).
- **Decoding configuration for the anchor's Table 2** — not fully stated in the paper, which is a direct
  risk to the reproduction gate.
- No code was run this session; `model.py` was read, not executed.

---

## 13. Sources read this session 🟢

- `litreview/data/anchor_fulltext.txt` — Table 1 and its Infilling/Code column (L452–462); Table 2 with
  the surrounding §4.4 analysis (L552–605); Appendix C.3 infilling protocol (L1459–1466).
- `model.py` L100–158 — the full sampling loop.
- `project-docs/` files 01, 02, 03, 04, 07, 09, and README.
- `limitations/output/limitations_and_research_problem.md` — the committed v1 run — and the v2 run as
  pasted into the session.
- Repository git state: branch, remotes, log, working-tree status.

arXiv identifiers cited above are carried from `03-established-facts.md` and
`09-p1-direction-analysis-2026-09-19.md` and were **not** re-resolved this session. 🟡 Per decision
principle 2 ("never invent a paper"), re-verify each before it appears in a submission.
