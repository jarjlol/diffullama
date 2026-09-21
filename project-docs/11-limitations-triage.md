# Limitations triage — which of L1–L12 can this project actually do?

> **Purpose:** a per-limitation walkthrough of the twelve research gaps in the multi-agent limitations
> deliverable, classified by whether they require adaptation training or can be executed at inference on
> released checkpoints. Written because the team's binding constraint is compute, and a limitation the
> project cannot execute is not a candidate direction regardless of how good the gap is.
>
> **Source document:** `limitations/output/limitations_and_research_problem.md` on branch
> **`assignment/limitations-multiagent`** @ `3e74ef5` ("feat(limitations): reimplement limitation
> generation as a faithful multi-agent pipeline"). This is the **v2** run — 12 items. Read against the
> anchor's extracted full text at `limitations/data/anchor_fulltext.txt`.
>
> **Companion documents:** `10-inference-time-direction-2026-09-21.md` argues the recommendation;
> this document explains each item. `limitations/CORRECTIONS.md` records errors found in the
> deliverable itself.
>
> Confidence labels follow `03-established-facts.md`: 🟢 PRIMARY, 🟡 SECONDARY, 🔴 WEAK.

---

## ⚠️ Read this first — which run's numbering

Two runs of the limitations pipeline exist with **incompatible L-numbers**.

| | v1 | **v2 — used throughout this document** |
|---|---|---|
| Branch | `main` (`a131e13`) | **`assignment/limitations-multiagent` (`3e74ef5`)** |
| Items | 10 | **12** |
| Pipeline | earlier implementation | reimplementation: Extractor / Analyzer / Reviewer / Citation workers, Judge, Self-Feedback, Master |

The commit message on `3e74ef5` says **"reimplement"**, so v2 supersedes v1 by intent. This resolves the
question left open in `03-established-facts.md` F-31 and in `10-...md` §12.

**The dangerous collision: v1's `L8` is v2's `L6`.** Full mapping in `limitations/CORRECTIONS.md` C-1.

---

## The table

| | Needs training | Inference-only | Verdict |
|---|---|---|---|
| **L1** annealing dropped at 7B | ✅ paired adaptation run | — | **out** |
| **L2** shift operation | ✅ paired adaptation run | — | **out** |
| **L3** `[MASK]` token confound | ✅ (the token-frequency count is CPU, but that's a footnote, not a project) | — | **out** |
| **L4** proxy-vs-real recipe | ✅ most expensive of all — needs **both** | — | **out** |
| **L5** instruction tuning deferred | ✅ SFT run | — | **out** |
| **L6** answer-selection headroom | — | ✅ **fully** | **best candidate** |
| **L7** unconverged / undertrained | ✅ | — | **out** |
| **L8** code + infilling claims | — | ✅ fully | **viable**, but contains an error (§L8) |
| **L9** efficiency not normalised | — | ✅ fully | already rejected as **Bet 5** |
| **L10** alternative adaptation routes | ✅ | — | **out** |
| **L11** statistical rigor | ✅ seeds on ablations | ⚠️ decoding-config sensitivity only | **thin as standalone** |
| **L12** contamination / provenance | — | ✅ CPU-only | **blocked** — no data manifest (its own point) |
| **adaptation-residue audit** (Research problem, second linked question) | — | ✅ fully, one GPU | **second best** |

Seven of twelve are eliminated by the compute constraint. What follows explains each verdict.

---

# The training-bound limitations

These are not bad gaps. Several are the best gaps in the document. They are out because the project
cannot run adaptation training, not because the science is weak — see `02-decision-log.md`
**D-2026-09-21-a**, which records this as a resource decision so that it can be revisited if compute
appears.

---

## L1 — Attention-mask annealing dropped at the deployment scale

**What the deliverable says.** Annealing is removed entirely for the 7B adaptation, justified as having
"minimal impact" and as simplifying the flash-attention-2 implementation. The supporting ablation
measures the opposite trend: **+2.1** accuracy points at 124M and **+2.5** at 355M. The paper extrapolates
"minimal" roughly twentyfold past its largest tested model, in the direction its own data argues against.

**Verified in this repository** 🟢 — `03-established-facts.md` **F-1** quotes the paper verbatim ("The
mask annealing has minimal impact, so we choose to omit it for 7B adaptation to simplify implementation
using flash-attention 2") and reproduces Table 3: DD-w/o-anneal 43.3 / 47.2 versus DD 45.4 / 49.7.

**Why it needs training.** Answering it means running the adaptation with and without annealing at a
scale above 355M and comparing. There is no inference-time proxy: annealing is a property of the training
attention mask, and the released checkpoint was trained one way only. You cannot recover the
counterfactual from a frozen model.

**What it would cost.** `09-p1-direction-analysis-2026-09-19.md` §3 prices the full design — a scale
ladder at 124M / 355M / 774M / 1.5B / 7B — at roughly **160 GPU-hours on one card**, three to four weeks
of wall-clock at half availability. 🔴 arithmetic, not measurement.

**Verdict: out, and explicitly preserved rather than withdrawn.** This was Option B, recommended at ~75%
confidence on 2026-09-19. The gap is still real: `03-established-facts.md` **F-22** establishes that the
only existing ≥7B annealing result (NBDiff, [2512.06776](https://arxiv.org/abs/2512.06776)) targets
**block** diffusion with a **modified** annealing schedule, so nobody has tested DiffuLLaMA's actual
annealing at 7B for full-attention diffusion. `04-open-questions.md` **Q-9** stays open as a question the
project is *declining to pursue*, not one that has been answered.

---

## L2 — The shift operation is the larger effect and gets the smaller share of scrutiny

**What the deliverable says.** Removing the shift operation costs **11.9 to 15.2** accuracy points —
roughly five times the annealing effect the paper discusses at length. The shift is inherited directly
from autoregressive training dynamics, making it the component most likely to carry AR-specific
assumptions forward, yet its behaviour at scale is never examined the way annealing's is.

**Why it needs training.** Same structure as L1, and same impossibility: "no shift" is a training
configuration. The released checkpoint has the shift baked in.

**A second reason it is weak even with compute.** `09-...md` §2.4 assessed this as an optional arm rather
than a research question: *"'no shift' is a strictly worse variant nobody ships, and Dream, DiffuCoder and
NBDiff all keep it."* Demonstrating that a variant nobody uses is worse is not a finding. The interesting
version of L2 — does the shift carry AR bias forward? — is **not** a training question, and it is exactly
what the adaptation-residue audit tests at inference. See §residue.

**Verdict: out as stated.** Its interesting half survives inside the residue audit.

---

## L3 — The `[MASK]` token is a workaround, applied inconsistently, confounded with size

**What the deliverable says.** The authors state the vocabulary should be expanded but reuse an existing
word for implementation reasons, then apply this inconsistently: DiffuGPT-S and DiffuLLaMA reuse tokens
**10541** and **811** while DiffuGPT-M gets a genuinely new token **50257**. DiffuGPT-M is also the
strongest performer at its scale, so mask-token treatment is perfectly confounded with model size and
never disentangled. The "low frequency" justification is asserted without any corpus measurement.

**Verified in this repository** 🟢 — `03-established-facts.md` **F-2** confirms the token IDs in code
(`DiffuLLaMA-training/train.py:304`; `LLaMA-Factory/src/llamafactory/model/loader.py:145,152`) and records
the same three unexamined points.

**Why it needs training.** Breaking the confound means training a 2×2 — reused token versus new token,
crossed at two scales. Four runs. `09-...md` §3 scoped exactly this as a secondary arm of Option B at
124M/355M.

**The one CPU-only piece, and why it is not a project.** Measuring the actual corpus frequency of tokens
811 and 10541 needs no GPU at all — a tokenizer and a corpus sample. It would settle whether the paper's
"low frequency" assertion is true. But the result is one sentence: either the assertion holds or it does
not. It cannot carry a paper, because knowing the frequency does not tell you what the reuse *cost*, and
that requires the 2×2.

**Verdict: out.** Worth listing the frequency count as a spare CPU task for a member without GPU access;
it is a footnote, not a direction.

---

## L4 — The recipe was selected on a proxy task and never validated on adaptation training

**What the deliverable says.** The authors state direct ablation on adaptation training is too costly and
substitute cheaper finetuning on GSM8K-symbolic to choose the recipe. Every recipe conclusion therefore
rests on the untested assumption that the proxy ranks components the same way the real 30B-to-65B-token
procedure would. No check on that assumption is reported.

**Verified** 🟢 — `09-...md` §1 row 10 quotes the paper: *"Direct ablation on adaptation training is
costly; hence, we conduct preliminary experiments … finetune models on the augmented GSM8K symbolic
dataset."* From GPT-2 weights, not the 100B+-token pre-training.

**Why it is the most expensive item in the document.** Validating a proxy requires running **both** the
proxy **and** the thing it proxies for, then comparing their component rankings. It is strictly more
expensive than L1, which needs only the real run. This is the most compute-intensive of all twelve.

**Why it is nonetheless the sharpest criticism in the document.** L1, L2 and L3 are each an instance of
L4. If the proxy does not rank components faithfully, then *every* recipe decision in the paper is
unsupported, not just the annealing one. It is the root of which the others are branches.

**Verdict: out, decisively.** Worth citing in any write-up as the reason the recipe questions matter, but
unexecutable at any budget this project will see.

---

## L5 — Instruction tuning deferred despite the paper's own evidence it is needed

**What the deliverable says.** Chain-of-thought prompting *reduces* accuracy, which the authors attribute
to the absence of instruction tuning — while citing prior work showing diffusion LMs benefit from it —
and then defer it to future work. Concurrent work doing adaptation plus instruction tuning reports both
the adaptation training and the base-model choice are decisive.

**The paper's own numbers** 🟢 (`limitations/data/anchor_fulltext.txt` L568–580, Table 2): CoT scores
**28.7 / 9.5 / —** against few-shot **31.3 / 23.6 / 20.9**. On SATMath, CoT more than halves accuracy
(23.6 → 9.5). That is a large, self-reported failure.

**Why it needs training.** Instruction tuning is a training run by definition. There is no inference-time
version.

**Verdict: out.** But note the connection to L6, which is the direction being recommended: both L5 and L6
are about the paper's *own diagnosis of its own weakness*, deferred rather than tested. L6 is the one of
the two that can be tested without a GPU-month.

---

## L7 — Every reported number is an unconverged lower bound

**What the deliverable says.** The authors state the models are undertrained and show no sign of
saturation; that DiffuLLaMA falls short of the LLaMA2 it was adapted from, attributing the shortfall to
the data subset without testing that attribution; that the appendix concedes the training budget does not
match GPT2's ~200B tokens while the main tables present the comparison as direct; and that the Table 12
generation samples are largely incoherent beside the fluency numbers reported in the body.

**Why it needs training.** "Is the model undertrained?" is answered by training it more. Nothing about a
frozen checkpoint reveals where its loss curve would have gone.

**Two pieces that are inference-only but small.** Auditing Table 12's samples against the reported
perplexity and diversity numbers needs only generation. So does testing the attribution of the
LLaMA2 shortfall to the data subset — partially, and weakly. Neither is a paper; the first is a figure,
the second is confounded beyond rescue without the training data.

**Verdict: out.** It is also, in a sense, the paper's own honest disclosure rather than a hidden flaw —
criticising a paper for saying it is undertrained is weaker ground than criticising one that hides it.

---

## L10 — One adaptation route, two model families, no comparison against alternatives

**What the deliverable says.** Every result derives from GPT2 or LLaMA2 and none exceeds 7B, so the
central claim is supported only for two architectures. The paper commits to one route — continued
denoising training into masked diffusion — without comparing against alternatives the literature has
since shown: representation alignment instead of retraining, adaptation to uniform-noise diffusion where
every token stays editable, and giving AR models masked infilling directly without diffusion adaptation.

**Why it needs training.** Every alternative route named is a training procedure. Testing "would
representation alignment have been cheaper?" means running representation alignment.

**Additionally assessed as too expensive even under Option B** — `09-...md` §2.4 records L10
(cross-architecture transfer) as needing *"a third AR family and a continual-pre-training budget. Too
expensive."*

**Verdict: out, twice over.**

---

# The inference-only limitations

---

## L6 — A large accuracy headroom is measured, diagnosed, and left uninvestigated ⭐ best candidate

**What the deliverable says.** Hit-rate over generated candidates substantially exceeds single-answer
accuracy across reasoning benchmarks. The authors read this correctly as high answer uncertainty and call
the shortfall "temporarily suboptimal", implying more training resolves it. The measurement localises the
deficit in **answer selection** rather than **answer knowledge**, but no calibration analysis, reranking
or verifier is attempted, and the implication that training alone closes the gap is never tested.

### The numbers 🟢

`limitations/data/anchor_fulltext.txt` L568–580, Table 2 — exact-match accuracy, 4-shot on math, 2-shot on
TriviaQA:

| Setting | MAWPS | SATMath | TriviaQA |
|---|---|---|---|
| LLaMA2 | 63.5 | 24.5 | 45.4 |
| DiffuLLaMA-ZS | 9.7 | <1 | 18.5 |
| DiffuLLaMA-FS | 31.3 | 23.6 | 20.9 |
| DiffuLLaMA-SC (majority vote of 3) | 33.1 | 27.7 | 26.0 |
| **DiffuLLaMA-@k (hit rate, k = 3)** | **40.8** | **57.7** | **34.1** |
| DiffuLLaMA-CoT | 28.7 | 9.5 | — |

**Headroom against the paper's own best selector (SC):**

| Benchmark | SC | hit@3 | headroom | recovered by majority vote | captured |
|---|---|---|---|---|---|
| MAWPS | 33.1 | 40.8 | **+7.7** | 1.8 | 23% |
| SATMath | 27.7 | 57.7 | **+30.0** | 4.1 | 14% |
| TriviaQA | 26.0 | 34.1 | **+8.1** | 5.1 | 63% |

### Why it is inference-only

Everything needed is generation from a frozen checkpoint plus post-hoc selection over the generated
candidates. No weights change. The selectors are functions of the candidate set, so once candidates are
cached to disk the entire selector study becomes CPU work.

### Why it is the best candidate

1. **It has a reproduction target.** Table 2's six rows, on the released checkpoint. Course stage 2 needs
   one and this supplies it. No other inference-only item here has a comparable target.
2. **It has a baseline to beat and a published ceiling.** 33.1 / 27.7 / 26.0 to beat; 40.8 / 57.7 / 34.1
   as the oracle. The maximum winnable amount is known before the first run — unusual and very useful for
   planning a semester.
3. **It contests a specific author claim.** "Temporarily suboptimal" is an attribution to undertraining.
   The measurement localises the deficit in selection. Those are different claims and the paper tests
   neither.
4. **The anchor is not interchangeable.** The objection that sank EGR — that an inference-time controller
   wraps any model and says nothing about *this* paper (`09-...md` §2.2) — does not apply. The claim under
   test is DiffuLLaMA's claim about DiffuLLaMA, from DiffuLLaMA's own table.
5. **The obvious fix was already tried and mostly failed.** The paper deploys self-consistency explicitly.
   On SATMath it captures **one seventh** of the available headroom. The gap is "the standard selector
   leaves 26 points on the table", not "nobody looked".

### ⚠️ Two traps

**Beat SC (27.7), not FS (23.6).** The v1 deliverable quotes "57.7 vs. 23.6", measuring against few-shot
rather than against the paper's own best method. Beating the weaker number would be a strawman. See
`limitations/CORRECTIONS.md` C-4.

**The novelty of this has not been checked.** `04-open-questions.md` **Q-13** is marked blocking: nobody
has searched whether test-time selection, calibration or hit@k scaling has been published for *adapted*
diffusion LMs. Test-time scaling is heavily worked in the autoregressive literature. **This direction must
not be ratified before that search runs** — this project has already been preempted twice (CDC, DiffPDE).

**Verdict: best candidate, gated on Q-13.** Design, selectors, baselines and risks in `10-...md` §4.

---

## L8 — The code and infilling claims are broader than the evaluation ✅ viable, ❌ contains an error

**What the deliverable says.** Code generation is advertised in the abstract and introduction, but the
only code evaluation is HumanEval single-line infilling with prefix and suffix supplied, "and that number
belongs to a separately trained Diffu-CodeLLaMA rather than the released checkpoint". The authors flag
their own infilling comparison as potentially unfair because suffix information is withheld from
baselines, and never run a corrected version. The evaluation also supplies the span length.

### ❌ The error 🟢

**The Diffu-CodeLLaMA clause is false.** `limitations/data/anchor_fulltext.txt` L452–462, Table 1,
"Infilling / Code" column, pass@1 %:

| Model | Code |
|---|---|
| **DiffuLLaMA 7B** | **15.5** |
| LLaMA2 (prefix only) | 1.7 |
| DiffuGPT-M | 2.9 |
| GPT2-M (prefix only) | 2.6 |
| DiffuGPT-S | 0.3 |

The released checkpoint has its own number. Diffu-CodeLLaMA's **0.76** is **Table 8**, a separate
CodeLLaMA finetune on 100M Starcoder tokens.

**The honest version of the criticism does not need the false clause:** 15.5% is a single-line fill with
gold prefix *and* suffix supplied, which is far narrower than the "code generation" claimed in the
abstract. That stands on its own.

### 🔎 Where the error came from — Q-16 substantially answered 🟢

Traced this session through `limitations/data/` on `assignment/limitations-multiagent`:

- The claim originates in **REV-1**, the Reviewer agent's first-round output in `agent_outputs.json`.
- It survives into `agent_outputs_round2.json`, `agent_outputs_final.json`, `master_clusters.json`,
  `master_merged.json` and the deliverable — the Judge scored the reviewer **86/100** and did not catch it.
- **`rag_corpus.json`, `rag_top20.json`, `rag_retained.json`, `ground_truth.json` and
  `zeroshot_baseline.json` contain zero occurrences of "CodeLLaMA".**

**So this is agent hallucination, not retrieval contamination.** The RAG corpus is clean.

That revises the hypothesis in `03-established-facts.md` **F-30**. The same error appeared independently
in three EGR documents, and F-30 inferred a shared upstream source. With the corpus clean, the likelier
explanation is that **the anchor's structure invites this specific misreading**: Table 1 carries a Code
column for the released checkpoint while Table 8 carries a Diffu-CodeLLaMA row, and a reader who finds
Table 8 first naturally concludes the code number belongs to it.

**This is more useful than a contaminated chunk would have been**, because it is reproducible: any future
agent reading this paper may make it again. The correction therefore belongs somewhere agents read at
session start — which is why it is in `limitations/CORRECTIONS.md` C-2 and in F-30.

### What survives, and is inference-only

- **The self-flagged unfair comparison** — the paper withholds suffix information from AR baselines, notes
  it "might result in an unfair comparison", and never runs a corrected version. Running it is pure
  inference.
- **The fixed-span-length assumption** — the evaluation supplies span length; flexible-length, flexible-
  position infilling is the harder task. Background: DreamOn
  [2602.01326](https://arxiv.org/abs/2602.01326) covers variable-length infilling for Dream-Coder and
  DiffuCoder but **not** DiffuLLaMA; [2609.02108](https://arxiv.org/abs/2609.02108) covers adaptive-length
  infilling. 🟡 Re-read both in primary form before any novelty claim.

**Protocol** 🟢 (Appendix C.3, `anchor_fulltext.txt` L1462): `openai/human-eval-infilling`, **1033 test
cases**, pass@1.

**Verdict: viable, and the designated fallback if L6 fails its reproduction gate.** Ranked below L6
because 15.5% is a much weaker base to build on than Table 2's 27.7–33.1, so "beat the baseline" is
thinner. Its advantage is test-set size: 1033 cases against L6's three small benchmarks.

---

## L9 — The efficiency claim is not compute-normalised ❌ already rejected as Bet 5

**What the deliverable says.** Decoding latency is reported at batch size 1 against a single AR
configuration, with no FLOPs- or forward-pass-normalised comparison, despite diffusion requiring full
self-attention over the whole sequence at every step with no KV caching. No optimised AR baseline such as
speculative decoding is included. Later work on adapted models reports limited speedup, attributed to a
pretrain-to-posttrain mismatch.

**Fully inference-only, and cheap.** That is not the problem.

**The problem: this is settled ground.** `07-decision-tree.md` Fork 4 rejected it as **Bet 5** —
*"premise decayed; wall-clock is standard now."* Reopening a settled rejection requires new evidence, and
neither the deliverable nor this triage has any.

**What to keep — the components, not the project.** Two verified repository facts belong in any cost table:

- **F-7** 🟢 — `inf_diffullama.py:24` defaults `_attn_implementation="eager"`, and with
  `attn_mask_ratio=1.0` the annealing mask is **all zeros**: a mathematical no-op that nonetheless blocks
  the flash-attention kernel. That is the same kernel whose convenience justified removing annealing in
  L1. Quotable irony; not a paper.
- **F-5** 🟢 — every denoising step forwards the entire sequence, no KV cache, regardless of how many
  positions remain masked.

Fixing F-7 is worth doing anyway: it speeds up every experiment under L6 and the residue audit.

**Verdict: rejected as a direction, retained as a measurement inside L6's cost reporting.**

---

## L11 — Statistical and reproducibility rigor ⚠️ thin as standalone

**What the deliverable says.** No result carries a seed, variance estimate or significance test —
including the ablations whose 2.1-to-2.5-point differences drive the recipe decisions. The 7B run consumes
65B tokens with no released data manifest or sampling seed. Decoding configurations are unstated on both
sides of key comparisons despite the paper's own figures showing sharp step-count sensitivity. The
released inference path defaults to eager attention with an all-zeros 4-D mask. Continual pre-training is
known to be sensitive to warmup and learning-rate schedule, neither ablated.

**It splits cleanly in two.**

**The training half — out.** Putting seeds and confidence intervals on the ablations means re-running the
ablations. That is L1 and L3 with error bars, and costs more than either.

**The inference half — real but small.** Decoding configurations are unstated; the paper's Figure 3 shows
sharp step-count sensitivity. A decoding-sensitivity sweep on the released checkpoint is **one good
figure**, not a paper.

**Where it goes: fold into L6 as a robustness section.** Two reasons this is the right home rather than a
consolation prize:

1. Step-count ensembling is already a candidate selector in L6's design (`10-...md` §4.5 item 7), so the
   sweep is work L6 needs regardless.
2. L11's complaint lands directly on L6's critical path — `04-open-questions.md` **Q-14** notes the
   reproduction gate may fail because Table 2's decoding configuration is not fully stated. Pinning and
   reporting the decoding configuration means the project does not repeat the failure it is criticising,
   which is worth a sentence in the write-up.

**Verdict: thin as standalone, valuable as L6's robustness section.**

---

## L12 — Data provenance, contamination and release transparency 🚫 blocked by its own premise

**What the deliverable says.** The adaptation corpus mixes web-scraped text with Starcoder source code
carrying heterogeneous licences, with no licensing or data statement. No benchmark decontamination is
reported despite continual pre-training on tens of billions of web and code tokens followed by evaluation
on public benchmarks, leaving contamination unexcluded as a partial explanation of performance. Weights
are released without a model card, intended-use statement or dual-use discussion.

**Fully CPU-only.** No GPU at any point — n-gram overlap between benchmark items and corpus text. Under a
pure compute constraint this is the cheapest item in the entire document.

**Why it is blocked anyway.** A contamination check requires the adaptation corpus, and **L12's own
central complaint is that no data manifest was released.** The obstacle is data access, not compute, so
the constraint that eliminates L1–L10 does not rescue this one.

**The partial version, and why it is weak.** You can reconstruct an approximation from public SlimPajama
and Starcoder subsets and measure overlap against the benchmarks. But a negative result then means only
*"no contamination found in the portion of the corpus we could reconstruct"* — which is a much weaker
claim, and must be written that way rather than as "no contamination". A positive result would be strong;
a negative one, which is the likelier outcome, would be nearly uninformative.

The licensing and model-card half is a documentation critique. Real, and correctly raised, but it is an
observation rather than an experiment — there is nothing to run.

**Verdict: blocked as a direction.** Reasonable as an appendix, or as a self-contained task for a member
without GPU access who wants something off the critical path.

---

## The adaptation-residue audit — second best ⭐

Not one of the twelve. It is the **second linked question** in the deliverable's Research-problem section,
and the deliverable's own scoping note names it as the tractable entry point.

**What it asks**, verbatim from the deliverable:

> "determine whether models adapted from autoregressive checkpoints retain measurable autoregressive
> inductive bias at inference relative to comparable models trained from scratch, since the recipe's
> purpose is precisely to remove that bias, and the base paper never verifies that it did."

**Why it is inference-only.** It compares the behaviour of released checkpoints. Nothing is trained.
Four checkpoints at bf16 — DiffuLLaMA 13.5 GB, LLaDA-8B 16.03 GB, Dream-7B 15.2 GB (F-16), DiffuCoder-7B
unverified 🔴 — run sequentially with one resident at a time, each fitting one 96 GB card with large
margin.

**Why the gap is credibly open.** `03-established-facts.md` **F-15** 🟡 records that generation-order
falsification is published at scale — [2601.15593](https://arxiv.org/abs/2601.15593) across 58 benchmarks
and 8 MDLMs up to 100B, [2608.05687](https://arxiv.org/abs/2608.05687) as a causal commitment-order
intervention, [2601.13228](https://arxiv.org/abs/2601.13228) on AR models rivalling diffusion at any-order
generation — and then states: *"**But every one studies from-scratch models. None studies an adapted
model** — that gap is open."*

The methodology is published, so the project reuses metrics rather than inventing and defending them,
while the population is untested. That is a replication-and-extension paper: a real genre, and far easier
to defend than a novel-mechanism claim.

**The anchor hypothesises the residue and never measures it** 🟢 (`anchor_fulltext.txt` L590): *"We
hypothesize that the adapted model retains some of the abilities from the base AR model."* Same shape as
L6 — an author claim left untested.

**It gives L2 its interesting half back.** L2's real worry is that the shift operation carries AR
assumptions forward. That is a question about the *resulting model*, testable at inference, not a question
about the training run.

**It also rehouses a question the project already narrowed.** `07-decision-tree.md` Fork 4: Bet 3 *"do
dLLMs plan?"* → *"already done twice, at 100B scale"* → *"BUT: only from-scratch models studied, never
adapted ones"* → **NARROWED, folded into B**. Option B is now closed on compute, so the narrowed question
needed a new home. This is it, and it needs no training.

**⚠️ Two configuration traps that silently invert the result:**

- **F-10** 🟢 — Dream ships `"alg": "origin"` (the **random** branch), `temperature: 0.0`, `steps: 512`.
  Run without explicitly setting `alg="entropy"` it silently decodes in random order.
- **F-11** 🟡 — LLaDA unmasks *highest*-confidence-first; `remasking='low_confidence'` names which tokens
  *stay* masked. Its default is semi-autoregressive (`block_length=32`) — left on, the from-scratch
  control is partially autoregressive **by configuration**, and the whole comparison is void.

**⚠️ The confound that must be stated, not hidden.** Adapted and from-scratch models differ in backbone,
tokenizer, corpus and token budget as well as in adaptation. No inference-only design separates these.
The defensible claim is *"adapted checkpoints show property X and the from-scratch one does not"* — **not**
*"adaptation causes X"*. Overclaiming causality is the likeliest reviewer objection and the easiest to
avoid.

**Verdict: second best, and the novelty-carrying axis.** L6 supplies the result and the course-stage fit;
this supplies the reason the work is about *adaptation* rather than about sampling. Keep the two results
separate in any write-up and let the discussion connect them, flagged as speculation.

---

## Summary — what the constraint leaves

| Rank | Direction | Role |
|---|---|---|
| 1 | **L6** answer-selection headroom | reproduction target, baseline to beat, published ceiling, contests a specific author claim |
| 2 | **Adaptation-residue audit** | the novelty axis; makes the work about adaptation |
| 3 | **L8** infilling claims | fallback if L6's reproduction gate fails |
| — | L11 (inference half) | robustness section inside L6 |
| — | L9 components (F-5, F-7) | cost reporting inside L6 |
| — | L3 token-frequency count, L12 partial | spare CPU tasks, off the critical path |

**Prerequisite for 1, 2 and 3 alike** — `10-...md` §8, both verified 🟢 this session:

- **F-28** — `model.py:132` unmasks **uniformly at random**; combined with **F-4** (`x0_scores` is the
  sampled-token log-prob, not the max, and `model.py:158` never returns it), **no confidence or entropy
  ordering baseline exists in this repository.** It must be built, using **max softmax probability**.
- **F-29** — `model.py:134` only ever clears `maskable_mask`. **There is no remasking path at all.**

Both are GPU-free, roughly a week, and useful under every direction above. Per `01-project-brief.md` only
three members have workstation access; this is the substantial work the other four can own, and it should
start before the direction is formally ratified.

**Before ratifying anything: run Q-13.**
