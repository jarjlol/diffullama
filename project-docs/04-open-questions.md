# Open questions

What is genuinely unknown or blocked. **This is where help is most useful.**
When something here gets resolved, move it to `03-established-facts.md` and say what resolved it.

---

## 🔴 Blocking

### Q-1 — Does the target venue still exist?
The project brief names **Stanford Agents4Science**. Verified: the 2025 edition closed submissions
5 Sept 2025 and held its event **22 Oct 2025**. **No 2026 edition found.**

Possibilities: (a) a 2026/2027 edition exists but is unannounced — if it follows the 2025 pattern, its
deadline would be roughly two weeks out and impossible; (b) the brief's reference is aspirational and the
manuscript itself is the deliverable; (c) a different AI-for-science venue is intended.

**Needs:** ask TTV. Changes the deadline and how much polish the artefact needs, not the work itself.

### Q-2 — Which direction? (P-1 in the decision log)
B (adaptation audit) vs C2 (trace-guided post-hoc repair). They need different reproduction targets and
cannot both be the main project. C2 is the refined form of the old Structure-Guided ReMasking option:
frozen dLLM → failed visible test → trace-guided AST/def-use span selection → bounded remask-and-retest
loop. **Everything in stages 2–6 is blocked on this team decision.**

### Q-3 — When is the SOTA assignment due?
Not announced as of 2026-08-29. The work is complete and unsubmitted. **Check Quanta directly** — forum
posts do not reach Gmail, which is why an inbox search found nothing.

---

## 🟡 Needs confirmation

### Q-4 — May BITS Goa students use the Sharanga cluster, and for what?
**Revised 2026-08-29 — earlier framing overstated this considerably.**

The cluster's raw inventory is impressive, but per the team: it is **SLURM-scheduled** (jobs queue, no
interactive whole-node access), **all GPUs on a node cannot practically be requested at once**, and **the
Blackwell node is reserved for admins**. The configuration page documents no access policy at all.

Still unknown: partition names, per-user GPU caps, wall-time limits, and whether BITS Goa students are
eligible at all.

**Not a dependency.** The local 2× Blackwell workstation (192 GB) is sufficient for Option B. Treat
Sharanga as possible batch capacity pending approval — plan around hardware the team controls.

### Q-5 — Has CDC been updated or accepted since May 2026?
[arXiv:2605.16829](https://arxiv.org/abs/2605.16829) is a v1 preprint with no venue. Re-check at kickoff;
an accepted version may have expanded scope.

---

## 🟢 Answerable with a few hours of work

### ~~Q-6 — Does the offset-mapping chain survive the *real* LLaMA tokenizer?~~ ✅ RESOLVED 2026-08-29

**Answer: yes, with a caveat.** Tested via `audit/test_real_tokenizer.py` against the real tokenizer.
All 26 statement nodes map to tokens that cover their source (no corruption), but **85% bleed one
character of leading indentation** because SentencePiece glues the last indent space onto the following
identifier. Whitespace only, never code.

Not a blocker for Option C, but it forces an explicit design choice about indentation handling that the
design doc currently makes silently. See `03-established-facts.md` F-21.

### Q-7 — Is the compute estimate right?
F-17 is FLOPs arithmetic, not measurement. One hour of benchmarking on the actual hardware settles it.
*Needs a working torch environment.*

### Q-8 — Does CDC's neighbourhood-scope optimum transfer to functional bugs?
F-9's result is on **CWEval, a security benchmark**, with metric `func-sec@1`. CDC routes functional
correctness through an entirely different mechanism. Whether the same interior optimum (Parent+Leaf beats
both broader and tighter) holds for *functional* repair is unmeasured.

**This is arguably Option C's single best remaining research question** — it is a real open question rather
than a reframing of a preempted one.

### Q-12 — Does trace-guided span selection beat simpler post-hoc policies?

This is the decisive feasibility and novelty test for proposed Option C2. Under matched token budget `K`,
native sampler, and bounded attempt budget `R`, compare random remasking, a local traceback window,
static AST/def-use selection without execution evidence, model-confidence selection (if exposed fairly),
and trace-guided AST/def-use selection.

The result must be measured on held-out tests, not merely visible repair tests. If trace-guided selection
does not beat static-only and traceback-window policies, the added controller complexity is not justified.

### Q-9 — How much does dropping annealing actually cost at 7B?
**The core of Option B. Strengthened 2026-08-29 after reading NBDiff in full.**

F-1: the annealing gain *grows* with scale (+2.1 at 124M → +2.5 at 355M), yet was called "minimal" and
omitted at 7B — a ~20× extrapolation past the largest tested model.

F-22: the only existing 7B annealing result ([2512.06776](https://arxiv.org/abs/2512.06776)) targets
**block**-diffusion, uses a **modified** annealing restructured for parallel training, runs only 4000
iterations, and critiques annealing on *theoretical* grounds. It never tests full-attention adaptation and
never discusses DiffuLLaMA's omission.

**So the cell is genuinely empty:** nobody has tested DiffuLLaMA's actual annealing at 7B for
full-attention diffusion. Two confounded variables — scale and adaptation target — and this project's
anchor sits exactly in the untested cell.

---

## ⚪ Unresolved, low priority

### Q-10 — Bet 1's sequence-partitioning sliver
Whether anyone has done sequence-dimension partitioning of *denoising positions* for a single dLLM request
across GPUs came back **COULD NOT VERIFY** — neither confirmed nor refuted. Moot unless bet 1 is revived,
which is not recommended.

### Q-11 — Is `mock.py` / the DiffuGPT-small local path viable?
Never exercised. Needs a torch environment.

---

## Practical prerequisite for almost everything above

`torch` and `transformers` are not installed on the development laptop (**RTX 3050, 4 GB** — verified, not
the 6 GB the design doc assumes).

| Task | Needs | Where it runs |
|---|---|---|
| **Q-6** real-tokenizer offset test | `transformers` **only** — `AutoTokenizer` needs neither torch nor a GPU (~50 MB) | laptop, today |
| Q-11 DiffuGPT-S mechanics | `torch` + `transformers` (~2.5 GB); a 124M model fits in 4 GB | laptop |
| Q-7 compute measurement, all real experiments | full CUDA stack | Blackwell workstation |

**Q-6 is the only genuinely unblocked task** — it needs no GPU, no direction decision, and ~50 MB.

---

## 2026-09-21 — questions opened by the inference-only constraint

Context: the team has stated compute cannot support adaptation training, closing Option B on resources
(D-2026-09-21-a). Full analysis in
[`10-inference-time-direction-2026-09-21.md`](10-inference-time-direction-2026-09-21.md).

### Q-13 🔴 BLOCKING — Has anyone published test-time selection or hit@k scaling for *adapted* diffusion LMs?

**Nobody has searched.** The recommended primary direction (L6, the answer-selection headroom, F-26/F-27)
rests on the assumption that this is unexplored, and that assumption has **not been checked even once**.

Test-time scaling, self-consistency and verifier-based selection are heavily worked in the autoregressive
literature. The claimed novelty is the *population* (adapted diffusion LMs) and the *specific contested
attribution*, not the mechanism. If someone has run hit@k or calibration on DiffuLLaMA, Dream or
DiffuCoder, the direction is preempted in the same way Option C was.

**This project has been preempted twice already** — CDC was posted three months before it was found
(`07-decision-tree.md` principle 5), and DiffPDE closed EGR's first gap four weeks after it was written
(`09-...md` §2.2). **Do not ratify the direction before running this search.**

Suggested scope: ReMDM and the remasking-policy line (F-14), diffusion test-time scaling, dLLM
calibration, self-consistency for masked diffusion, and anything citing the anchor's Table 2.

### Q-14 🟡 Does Table 2 reproduce on the released checkpoint?

The reproduction gate for L6. Targets: SC = 33.1 / 27.7 / 26.0 and hit@3 = 40.8 / 57.7 / 34.1 on
MAWPS / SATMath / TriviaQA (F-26).

**Known risk:** the paper does not fully state its decoding configuration for Table 2 — which is L11's own
complaint, landing on this project's critical path. Step count, temperature and top-p all need to be
pinned and reported, and a failure to reproduce may be a configuration difference rather than a real
discrepancy. If the SC row cannot be reproduced within a reasonable margin, the direction is unsound and
L8 becomes the fallback.

### Q-15 🟢 How much of the hit@3 headroom is a formatting artifact?

Cheap and decisive, and it should be done **before** any selector is built. If hit@3 counts candidates
correct on formatting technicalities that no selector could detect, part of the +30.0 SATMath headroom is
illusory and H1 collapses before the experiment starts.

Checkable by hand on a few dozen candidates once generation works. Needs no GPU beyond one generation run.

### Q-16 🔴 Where did the Diffu-CodeLLaMA infilling error come from?

Per F-30, the same false claim has now been generated independently by the EGR documents and by the
`limitations/` pipeline. Two independent reproductions indicate a **shared upstream source** that has not
been identified.

Candidates: a contaminated chunk in the limitations RAG corpus (`limitations/data/`), an agent prompt
carrying the claim, or a secondary source both pipelines retrieved. **Until it is found, anything else
from that source is suspect.** This is the live form of decision principle 1 and of
`05-mistakes-and-bugs.md` §A.

### Q-17 🟡 What is SATMath's item count?

The +30.0 headroom is the largest single number in the L6 case and carries the direction. On a small test
set its confidence interval may be wide enough to weaken the claim substantially. Not checked; needed
before any interval is quoted.

Same class of check as Q-14 — cheap, and load-bearing.

---

### Note on Q-2 and Q-9

**Q-2 (which direction)** is partly resolved: training-based directions are closed on compute
(D-2026-09-21-a). Which *inference-time* direction remains open, now tracked as **P-5** in the decision
log.

**Q-9 (what does dropping annealing cost at 7B)** is **not** resolved and is **not** withdrawn. The cell
is still empty and the question is still good — the team simply cannot afford to answer it. It stays here
as a question the project is declining to pursue, rather than being deleted, so that it can be picked up
if compute later becomes available.
