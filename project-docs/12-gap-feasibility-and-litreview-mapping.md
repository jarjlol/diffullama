# Gap feasibility report — L1–L12 against the litreview, scored with arXiv:2601.11578

> **What this is.** A feasibility and relevance assessment of the twelve research gaps in
> `limitations/output/limitations_and_research_problem.md`, answering three questions:
> (1) where does each gap sit in the literature review we already built, (2) how does each score
> under the evaluation rubric of the limitation-generation paper itself
> ([arXiv:2601.11578](https://arxiv.org/abs/2601.11578)), and (3) which of them can actually be
> executed under the binding constraint that **this project does no training and no AR→diffusion
> conversion — inference on released checkpoints only** — with a publishable paper as the end goal.
>
> **Date:** 2026-09-21. **Repo state:** branch `assignment/limitations-multiagent` @ `3e74ef5`.
>
> **Relationship to existing docs.** `10-inference-time-direction-2026-09-21.md` and
> `11-limitations-triage.md` (branch `plan/inference-time-direction`, by Aaditya, same day) already
> triage L1–L12 by compute class and recommend L6. **This document does not repeat that work and
> does not contradict its verdicts.** It adds the three things those documents do not have: the
> litreview mapping (§2), the 2601.11578 rubric applied per gap (§3), and a first-pass novelty
> check that materially changes the ranking (§5). Read §5 before ratifying anything.

---

## 1. Method, and where it deviates

### 1.1 The rubric, from the source paper

arXiv:2601.11578 defines **two** scoring instruments. Both are in its appendix (Fig. 8, Fig. 9) and
both are transcribed in `limitations/scripts/judge.py` and `limitations/data/agent_requests/`.

**Instrument A — Judge Agent, reference-free (Fig. 8).** Four criteria, each 0–10, combined by fixed
weights into a 0–100 total:

| Criterion | Weight | What it asks |
|---|---|---|
| Depth | 0.20 | How critical and insightful is it — does it reveal a significant issue in the study's design, findings or implications? |
| Originality | 0.20 | Generic critique, or novel context-specific insight? |
| Actionability | 0.30 | Can researchers realistically address it in future work? Does it give a clear path? |
| Topic Coverage | 0.30 | How broadly does **the set** cover methodology, scope, peer-review standards, cited-paper gaps? |

Score anchors, taken from the paper verbatim in substance: **0–3 poor** (superficial, generic, not
actionable, narrow), **4–6 fair** (somewhat insightful, partially actionable, incomplete),
**7–8 good** (mostly critical, slightly generic, broadly actionable), **9–10 excellent** (highly
insightful, novel, clearly actionable, comprehensive). The paper sets **8/10 weighted mean** as the
regeneration threshold.

**Instrument B — reference-based validation (Fig. 9).** Three criteria, each **1–5**, scored against
the source paper: **Faithfulness** (does it accurately represent the paper, without introducing
misinformation or contradictions), **Soundness** (detailed, specific, practical, logically coherent),
**Importance** (does it address issues that materially impact the paper's findings and contributions).

The paper's own best configuration (GPT-4o mini, 4 agents, Type-2 evaluation) scores
**Faithfulness 3.36 / Soundness 3.71 / Importance 3.40**. That is the comparison baseline for §3.

### 1.2 Four stated deviations

1. **Topic Coverage is a set-level criterion** — the paper's own wording scopes it to "the set of
   limitations". It cannot be scored per item. It is therefore scored **once for all twelve items**
   (§3.1) and excluded from the per-gap composite, whose weights are renormalised across the three
   remaining criteria (0.286 / 0.286 / 0.429). Stated rather than silently rescaled.
2. **Actionability is scored twice.** The paper means "can researchers realistically address this".
   That is exactly the feasibility question, so it is reported as **Actionability (general)** — any
   well-resourced lab — and **Actionability (this project)** — anchored to inference-only on the
   hardware in §4.1. The gap between the two columns *is* the feasibility story.
3. **Single judge, no independence.** The same model family produced the limitations and scores them
   here. `limitations/REPORT.md` §6.3 already records this bias; it applies to this document too.
   The source paper validates its judge against two human annotators. Nothing like that was done.
4. **The judged text is the working-tree version.** L8 in the tree still contains a verified false
   clause (§3.3). A corrected variant exists in commit `3b9fa15`, which is not currently checked out
   on this branch. Both are scored.

---

## 2. How the gaps map onto the literature review

`litreview/` produced a 48-reference survey of the anchor's domain (F1 = 0.222 against the anchor's
own §5; 0.272 after the LLM-judge rerank). `limitations/` retrieved 95 chunks (48 cited-in from the
survey + 47 cited-by substitute), narrowed to 13. Mapping each gap against both:

| Gap | Survey section | Survey refs | Retained RAG chunks (13) | Grounding |
|---|---|---|---|---|
| **L1** annealing | — | — | PreDiff-LM (10/10), UNIFUSION (9/10) | **RAG only** |
| **L2** shift | — | — | — | **anchor text only** |
| **L3** `[MASK]` token | §3.2–3.3 (adjacent) | — | — | **anchor text only** |
| **L4** proxy recipe | — | — | — | **anchor text only** |
| **L5** instruction tuning | §3.4, §4.3 | [14] TESS 2 | TESS 2 (9/10), Dream 7B (8/10) | **strong** |
| **L6** answer selection | §7 (decoding tension, obliquely) | [16], [2], [3] | — | **weak** |
| **L7** undertrained / budget | §4.1, §7 | [32], [36] | *(re)warm your model* (8/10) | **strong** |
| **L8** code + infilling | §5.4, §6 | [13], [9] | Flexible-length Infilling (9/10), Dream-Coder (8/10) | **strong** |
| **L9** efficiency | §5.1, §7 | [39], [16] | Jacobi Forcing (9/10), dLLM (8/10) | **moderate** |
| **L10** alternative routes | §4.2–4.3 | [14], [9], [21], [28] | Don't Retrain Align (10/10), UNIFUSION (9/10), *Enabling AR Models to Fill In Masked Tokens* (8/10) | **strong (RAG-led)** |
| **L11** statistical rigor | §4.1 | [36] | *(re)warm your model* (8/10) | **weak** |
| **L12** data / contamination | — | — | — | **none** |

### 2.1 The mapping's main finding: relevance and feasibility point in opposite directions

Count the survey keyword hits in `litreview/output/generated_survey.md`:
`anneal` **0**, `shift` **0**, `mask token` **0**, `seed` **0**, `contamin` **0**,
`self-consistency` **0**, `test-time` **0**, `verifier` **0**.

The survey is a *domain* survey — it maps text diffusion, continual pre-training, and
non-autoregressive generation. The sharpest limitations (L1–L4) are *recipe-internal*: they live in
the anchor's Table 3, its appendix, and its token IDs. A domain survey was never going to retrieve
them, and did not. They are grounded in the anchor's own text, which is why their Faithfulness
scores are the highest in §3 and their litreview support is nil.

Conversely, the four gaps with **strong** litreview grounding — L5, L7, L8, L10 — are grounded there
precisely because they touch what the surrounding field publishes about: instruction tuning, continual
pre-training budgets, infilling, adaptation routes. Three of those four (**L5, L7, L10**) are
training-bound and therefore dead for this project.

**The practical consequence:** the litreview is genuinely useful as related work for a paper about
L8, L9 or L10, and nearly useless as related work for L6 — the one direction the compute constraint
leaves standing. A paper built on L6 needs a *new* related-work pass. §5 shows what that pass finds.

### 2.2 A concrete retrieval blind spot, measurable in our own data

`limitations/data/rag_corpus.json` (95 chunks) contains **one** test-time-scaling paper — *Test-Time
Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts* — and it **did not survive into the
top-20**, let alone the retained 13. The corpus contains **zero** occurrences of `self-consistency`
and **zero** of `AR-ness`. Both of those are the exact literatures that decide whether L6 and the
adaptation-residue audit are novel. Neither pipeline was looking there, because neither was asked to.

This is not a pipeline defect — it is a scope mismatch, and it is why the novelty question stayed
open in both documents. It is also why §5 exists.

---

## 3. Rubric scores

### 3.1 Set level — the twelve items scored as one agent output (Instrument A, unmodified)

| Criterion | Score | Justification |
|---|---|---|
| Depth | 7 | L1, L4 and L6 are genuinely incisive; L7, L11 and L12 are bundled or ordinary. "Mostly critical, minor issues." |
| Originality | 6 | L1/L3/L4/L6 are context-specific; L12 sits close to the generic failure mode the source paper names; L9's premise has decayed. |
| Actionability (general) | 6 | Most are addressable by a resourced lab; L4 needs both the proxy and the real run, L12 is blocked by missing data. |
| Actionability (this project) | 3 | Seven of twelve require adaptation training. |
| Topic Coverage | 8 | Methodology, scope, evaluation breadth, statistics, provenance and ethics are all represented. |

**Weighted total: 68.0 / 100 (general) — 59.0 / 100 (this project).**

Neither clears the paper's 80/100 (8/10) threshold. That is consistent with the recalibrated judge
result already recorded in `limitations/REPORT.md` §5.2 (61 / 72 / 68 / 70 per agent). It should not
be read as failure: the source paper's own ablation rates its Extractor-only configuration at
**43.39** coverage, and its own agents score 6.2–8.9 per dimension. **68 on a hostile rubric applied
to a single well-studied paper is an ordinary, honest result.**

### 3.2 Per gap

Composite = Depth/Originality/Actionability under the paper's weights, renormalised (§1.2 item 1).
**Read the composite as a quality score and `Act (proj)` as the gate** — L4 scores well on quality
and is still unexecutable.

| Gap | Depth | Orig | Act (gen) | Act (proj) | Comp (gen) | Comp (proj) | Faith | Sound | Imp |
|---|---|---|---|---|---|---|---|---|---|
| **L1** annealing | 9 | 8 | 7 | 1 | 78.6 | 52.9 | 5 | 4 | 5 |
| **L2** shift | 7 | 6 | 5 | 1 | 58.6 | 41.4 | 5 | 3 | 4 |
| **L3** `[MASK]` token | 7 | 7 | 6 | 2 | 65.7 | 48.6 | 5 | 4 | 4 |
| **L4** proxy recipe | 9 | 8 | 3 | 1 | 61.4 | 52.9 | 5 | 3 | 5 |
| **L5** instruction tuning | 6 | 5 | 7 | 1 | 61.4 | 35.7 | 5 | 4 | 4 |
| **L6** answer selection | 8 | 7 | 9 | 9 | **81.4** | **81.4** | 5 | 5 | 4 |
| **L7** undertrained | 5 | 4 | 4 | 1 | 42.9 | 30.0 | 5 | 3 | 3 |
| **L8** code + infilling | 6 | 6 | 8 | 8 | 68.6 | 68.6 | **2** | 3 | 3 |
| **L9** efficiency | 6 | 4 | 8 | 8 | 62.9 | 62.9 | 5 | 4 | 3 |
| **L10** alt. routes | 7 | 7 | 4 | 1 | 57.1 | 44.3 | 5 | 3 | 4 |
| **L11** stat rigor | 6 | 5 | 6 | 4 | 57.1 | 48.6 | 4 | 4 | 4 |
| **L12** provenance | 5 | 4 | 3 | 2 | 38.6 | 34.3 | 5 | 2 | 2 |
| *(residue audit)* | 8 | 4† | 9 | 8 | 72.9 | 68.6 | 5 | 4 | 4 |
| **mean, L1–L12** | | | | | **61.2** | **50.1** | **4.67** | **3.50** | **3.75** |

† Originality 4, not 8. Scored **after** the novelty check in §5.1. As written, before that check, it
reads as an 8. This single number is the most consequential result in this document.

**Against the source paper's own best run** (Faith 3.36 / Sound 3.71 / Imp 3.40): our Faithfulness is
substantially higher (4.67), Soundness slightly lower (3.50), Importance higher (3.75). The
Faithfulness margin is expected and not a win — our items were hand-checked against
`litreview/data/anchor_fulltext.txt` during authoring; the paper's were not. **Soundness is the
honest signal, and it is the one dimension where we do not beat the paper's baseline.** Soundness
asks whether the critique comes with a practical remedy, and five of our twelve items have "run the
adaptation training again" as their remedy.

### 3.3 Two low scores that need naming

**L8, Faithfulness 2.** The working-tree text says the HumanEval infilling number "belongs to a
separately trained Diffu-CodeLLaMA rather than the released checkpoint." This is false. Table 1 of
the anchor gives DiffuLLaMA 7B its own Code figure of **15.5 pass@1**; Diffu-CodeLLaMA's **0.76** is
Table 8, a separate CodeLLaMA finetune on 100M Starcoder tokens. Fig. 9's anchor for 2 —
inaccuracies that could mislead a reader, key aspects misrepresented — fits exactly. Under
`judge.py`'s faithfulness gate (< 3 caps the total at 50) this item is disqualified as written.
**The criticism survives without the false clause**: a single-line fill with gold prefix *and* suffix
supplied is far narrower than the "code generation" claimed in the abstract. With the clause removed
(commit `3b9fa15`), Faithfulness goes to 5 and the composite is unchanged at 68.6. The error is
traced to agent hallucination in the Reviewer's round-1 output, not retrieval contamination — see
`limitations/CORRECTIONS.md` C-2 on `plan/inference-time-direction`.

**L12, Soundness 2 / Importance 2.** Fig. 9's anchor for 2 on Soundness is "limited practicality";
for Importance, "marginal improvements... fails to address more substantial gaps". Both apply. The
contamination check L12 demands requires the adaptation corpus, and L12's own central complaint is
that no data manifest was released. The item is blocked by the very absence it names.

**Status 2026-09-27.** The L8 error discussed above is corrected in the deliverable (`b0dc799` on `assignment/limitations-multiagent`). The
rubric scores in this section were assigned against the uncorrected text at `3e74ef5` and are **not**
re-scored here; the corrected L8 is the stronger version this section already recommends.

---

## 4. Feasibility under the inference-only constraint

### 4.1 The hardware, verified rather than assumed

| Resource | Reality | Consequence |
|---|---|---|
| This machine | **RTX 3060 Laptop, 6144 MiB total, ~4.9 GB free** (`nvidia-smi`, 2026-09-21). `torch` **not installed**. | DiffuGPT-S (124M) and DiffuGPT-M (355M) fit. **No 7B model fits, in any precision the repo's inference path supports.** |
| Workstation | 2× RTX 6000 Pro Blackwell, 96 GB each — **shared and contended**, and per `01-project-brief.md` accessible to Aaditya, Aryan and Arjun only | All 7B work must be run by someone with workstation access. This is a scheduling dependency, not a compute one. |
| Sharanga cluster | SLURM, Blackwell node admin-reserved, student eligibility unconfirmed (Q-4) | Not plannable. |

Checkpoint footprints (from F-16, already corrected once): DiffuLLaMA **6.74B / 13.5 GB bf16**,
LLaDA-8B **16.03 GB**, Dream **7.6B / 15.2 GB**, DiffuGPT-S **124M**, DiffuGPT-M **355M**.

**The operative constraint is therefore narrower than "no training".** It is: *no training, and every
7B experiment is queued behind shared access held by three of seven members.* Any plan that assumes
on-demand 7B inference should be costed at half availability.

### 4.2 Compute class per gap

Agreeing with `11-limitations-triage.md` — restated here with the rubric's Actionability column as
the gate, and flagging the two places this document reads the evidence differently.

| Gap | Class | Runs where | Verdict |
|---|---|---|---|
| L1, L2, L4, L5, L7, L10 | paired adaptation training | nowhere available | **out** |
| L3 | training (2×2 grid); token-frequency count is CPU | laptop for the footnote only | **out** |
| **L6** | inference + CPU post-processing | workstation for 7B generation; **selector study is CPU once candidates are cached** | **viable, best** |
| **L8** | inference | workstation (1033 HumanEval-infilling cases) | **viable** |
| L9 | inference, cheap | workstation | **rejected as a direction** (settled as Bet 5); keep as cost reporting |
| L11 | splits: training half out, decoding-sensitivity half in | workstation | **fold into L6** |
| L12 | CPU only | laptop | **blocked by data access, not compute** |
| residue audit | inference, 4 checkpoints sequentially | workstation | **see §5.1 before committing** |

**The one structural advantage of L6, which no other item has:** once the candidate generations are
written to disk, the entire selector comparison is CPU work. That decouples the four members without
workstation access from the three who have it, which matters more for a seven-person semester project
than any single compute number in this table.

### 4.3 Shared prerequisite, GPU-free, worth starting now

`model.py:132` unmasks **uniformly at random**, and `x0_scores` (`model.py:116`) is the sampled-token
log-prob, never returned (`model.py:158`). **There is no confidence-ordering baseline in this
repository, and no remasking path at all** (`model.py:134` only clears `maskable_mask`). Any of L6,
L8 or the residue audit needs both built, using max softmax probability rather than the existing
score. Also worth fixing regardless: `inf_diffullama.py:24` defaults to eager attention and passes an
all-zeros 4-D mask that is a mathematical no-op while blocking flash attention (F-7). Fixing it
speeds up every experiment below.

---

## 5. Novelty checks run for this report

Neither prior document had run one; `04-open-questions.md` **Q-13** marks this blocking. What follows
is a **first pass, not a full novelty audit** — it is enough to change the ranking, not enough to
ratify a direction.

### 5.1 The adaptation-residue audit is substantially preempted — by the anchor's own authors

**DiffuCoder** ([arXiv:2506.20639](https://arxiv.org/abs/2506.20639), Gong, Zhang, Zheng, Gu, Jaitly,
Kong, Zhang; Jun 2025) shares **three authors with the anchor paper** — Shansan Gong (first author of
both), Yizhe Zhang, and Lingpeng Kong. Read in primary form (HTML full text, §4.1–4.2):

- It defines **local AR-ness@k** and **global AR-ness@k** — inference-time metrics for how
  autoregressive a diffusion model's decoding order actually is.
- §4.2 states it compares AR-ness across "different dLLMs, including LLaDA trained from scratch and
  Dream or DiffuCoder adapted from AR LLMs". **That is exactly the adapted-vs-from-scratch design the
  residue audit proposes.**
- Its conclusion, verbatim: *"adapted dLLMs tend to exhibit stronger AR-ness than those trained from
  scratch"* — attributed to inheriting left-to-right dependencies from AR training.

`03-established-facts.md` **F-15** currently reads: *"every one studies from-scratch models. None
studies an adapted model — that gap is open."* **F-15 is false as stated and should be corrected.**
The three papers F-15 cites do study from-scratch models; DiffuCoder does not, and it is the one that
matters most because the anchor's own first author wrote it.

**What genuinely remains open**, and it is much thinner than the deliverable's framing:

1. DiffuCoder's AR-ness comparison **does not include DiffuLLaMA**. It covers LLaDA, Dream and
   DiffuCoder. The anchor paper's own model is absent from the analysis its own authors published.
2. The comparison is reported qualitatively on math and code with one configuration
   (low-confidence remasking, 512 steps); it is not a controlled study across model families with the
   backbone/corpus/budget confound stated.
3. DiffuLLaMA is **full-attention, non-block**, and adapted from a *general* AR model rather than a
   code model — a different point in the space from all three models DiffuCoder measures.

That is a real sliver, and it is the difference between "we discovered adapted models retain AR bias"
(preempted, by the authors) and "we measured where DiffuLLaMA sits on a published axis its own
authors did not apply to it" (open, and a much smaller claim). **The residue audit's Originality
score drops from 8 to 4 on this finding**, which is why it cannot carry a paper on its own.

This is at least the third preemption this project has caught; `11-limitations-triage.md` names CDC and
DiffPDE as the earlier two (neither verified here). It is the **first found by checking the anchor
paper's own authors' later work** — the cheapest place to look, and one that should be a standing
first step in any novelty pass this project runs.

### 5.2 L6's mechanism space is crowded; its *target* is not

A search for test-time scaling and selection on diffusion LMs returns a dense 2025–2026 cluster,
every one of them on LLaDA or Dream rather than DiffuLLaMA:

| Work | What it does |
|---|---|
| Prism ([2602.01842](https://arxiv.org/abs/2602.01842)) | hierarchical trajectory search + self-verified feedback; LLaDA-8B GSM8K 67.6 → 85.3 |
| Reward-Guided Stitching ([2602.22871](https://arxiv.org/abs/2602.22871)) | training-free parallel rollouts + stitching, +23.8% avg, 1.8× latency cut |
| RFG ([2509.25604](https://arxiv.org/abs/2509.25604)) | reward-free guidance; beats an equal-compute ensemble baseline |
| Hidden Semi-AR Experts ([2510.05040](https://arxiv.org/abs/2510.05040)) | test-time scaling over decoding schedules |
| Token-level cross-validation ([2510.05090](https://arxiv.org/abs/2510.05090)) | test-time token-level revision |

**Consequence for L6.** "Post-hoc selection recovers headroom in diffusion LMs" is no longer a novel
mechanism claim — it is an established sub-field with strong published baselines. What is *not* done
is the specific thing L6 actually asks: **the anchor paper measured a hit@3-vs-accuracy gap on its own
model, diagnosed it as "temporarily suboptimal" — an undertraining attribution — and never tested
that attribution.** Contesting a specific author claim on the specific model, with the anchor's own
Table 2 as both baseline and ceiling, survives this literature. Claiming a new selection method does
not.

This reframes L6 from a *method* paper to a *measurement* paper, and that is the version that is
defensible. It also means the related-work section must engage the five papers above honestly, and
any selector we build must be compared against at least one of them rather than only against
self-consistency.

**Q-13 is not closed by this.** A proper pass needs: a systematic search (not one query), primary-source
reads of at least Prism and RFG, and a check of whether any of them has already been run on DiffuLLaMA.

---

## 6. What this leaves: the recommendation

### 6.1 Ranked

| Rank | Direction | Rubric (proj) | Paper it would produce | Honest viability |
|---|---|---|---|---|
| **1** | **L6 as a measurement paper** — reproduce Table 2, quantify the selection-vs-knowledge split, test the "temporarily suboptimal" attribution | 81.4 | *"The headroom DiffuLLaMA reported is a selection deficit, not a training deficit"* — a falsification-flavoured empirical paper on the anchor's own claim | **Good, conditional on §5.2.** Has a reproduction target, a baseline (SC 33.1 / 27.7 / 26.0), and a published ceiling (hit@3 40.8 / 57.7 / 34.1). Maximum winnable amount known before the first run. |
| **2** | **L8 corrected-comparison rerun** — run the infilling comparison the authors flagged as unfair and never fixed | 68.6 | a short, clean evaluation-correction paper | **Fair.** 1033 test cases is the largest test set available to us. But 15.5 pass@1 is a weak base to build on, and the result is a correction rather than a finding. Best as a **section** of (1) or a workshop paper. |
| **3** | **Residue audit as an analysis section** — place DiffuLLaMA on DiffuCoder's AR-ness axis | 68.6 | not a paper on its own after §5.1 | **Weak standalone, valuable as a section.** It is what makes the work about *adaptation* rather than about sampling. Must cite DiffuCoder as the method being extended, never as background. |
| — | L11 decoding-sensitivity sweep | 48.6 | one figure | Robustness section inside (1). |
| — | L12 partial contamination check, L3 token-frequency count | 34.3 / 48.6 | one paragraph each | Spare CPU tasks for members without workstation access. |

### 6.2 The paper shape this actually supports

One paper, three results, in this order:

1. **Reproduce Table 2** on the released checkpoint with the decoding configuration stated — which the
   anchor does not state, and which is L11's complaint turned into our own methods section.
2. **Decompose the headroom.** Selection-vs-knowledge: build the confidence/entropy baseline that
   §4.3 says does not exist yet, compare selectors against self-consistency *and* against at least one
   published dLLM test-time-scaling method, and report what fraction of the hit@3 ceiling each
   recovers. The anchor's own numbers say self-consistency captures 23% / 14% / 63% of the available
   headroom on MAWPS / SATMath / TriviaQA — the SATMath figure is the result to chase.
3. **Position the model.** Measure DiffuLLaMA's local and global AR-ness using DiffuCoder's published
   metrics, and report where the anchor's own model sits on the axis its own authors introduced but
   did not apply to it.

The claim is: *"the anchor attributed its accuracy shortfall to undertraining; on its own numbers the
deficit is in answer selection, and here is how much of it is recoverable without training."*
That claim is falsifiable, cheap, and about *this* paper rather than about diffusion LMs in general —
which is the objection that sank the earlier execution-grounded-repair direction.

**Correction 2026-09-27 — the capture fractions in item 2 above are mislabeled.** 23% / 14% / 63% is
*SC gain ÷ headroom remaining above SC* (1.8/7.7, 4.1/30.0, 5.1/8.1), which is not a fraction of anything
majority vote recovered. The fraction of the few-shot → hit@3 headroom that majority vote captures is
*SC gain ÷ (hit@3 − FS)* = 1.8/9.5, 4.1/34.1, 5.1/13.2 = **19% / 12% / 39%**. The TriviaQA figure matters
most: majority vote is far less effective there than 63% suggests. Recomputed from Table 2 as verified
in `litreview/data/anchor_fulltext.txt`.

### 6.3 Kill criteria, set before starting

- **Reproduction gate.** If Table 2's six rows cannot be reproduced within a stated tolerance on the
  released checkpoint, stop and switch to L8. The decoding configuration is unstated in the anchor
  (Q-14), so this gate is a genuine risk, not a formality.
- **Novelty gate (Q-13).** If a proper novelty pass finds any of the five §5.2 papers already run on
  DiffuLLaMA with this framing, the measurement claim collapses to a replication note. Run this
  **before** any GPU time.
- **Margin gate.** If the best selector does not beat self-consistency on at least one benchmark by
  more than run-to-run variance — which requires seeds, which is L11's complaint applied to ourselves
  — there is no result, only a negative one. A negative result is publishable here *only* if the
  reproduction and the AR-ness positioning are both solid.

---

## 7. What this document does not establish

- **No code was run.** `torch` is not installed on this machine; every compute statement is arithmetic
  or a repository read, not measurement.
- **The novelty pass in §5 is a first pass.** Two searches and two primary-source reads (DiffuCoder in
  full text, 2601.11578 in full text). It is sufficient to demote the residue audit and to reframe L6.
  It is **not** sufficient to ratify L6. Q-13 stays open.
- **The scoring is not independent.** Same model family generated, judged and re-judged. The source
  paper validates its judge against human annotators; we have not, and a self-graded 68/100 should be
  read as an ordering signal, not a measurement.
- **Prism, RFG and the other §5.2 papers were read from search results, not in primary form.** Their
  numbers are quoted as retrieved. Per `docs/verification-standards.md` they must be read in full
  before appearing in any write-up.
- **F-15 needs correcting** in `03-established-facts.md`, and `11-limitations-triage.md`'s residue-audit
  section rests on it. That correction is the single most actionable output of this document.
