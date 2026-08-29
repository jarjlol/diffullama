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
B (adaptation audit) vs C (Structure-Guided ReMasking). Different papers, different reproduction targets,
cannot do both. **Everything in stages 2–6 is blocked on this.**

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

### Q-6 — Does the offset-mapping chain survive the *real* LLaMA tokenizer?
`audit/test_shift_offset.py` proves the shift invariant (tokenizer-independent) and validates the
AST→char→token logic against a **stub** tokenizer. The real LLaMA tokenizer is SentencePiece: it glues
leading whitespace into tokens and uses U+2581 markers, and **Python indentation is semantic**.

**To resolve:** re-run test step 5 with
`AutoTokenizer.from_pretrained('diffusionfamily/diffullama', use_fast=True)` and
`return_offsets_mapping=True`, asserting over a corpus of programs that decoded token spans cover each AST
node's source. *Only matters if Option C is chosen.*

### Q-7 — Is the compute estimate right?
F-17 is FLOPs arithmetic, not measurement. One hour of benchmarking on the actual hardware settles it.
*Needs a working torch environment.*

### Q-8 — Does CDC's neighbourhood-scope optimum transfer to functional bugs?
F-9's result is on **CWEval, a security benchmark**, with metric `func-sec@1`. CDC routes functional
correctness through an entirely different mechanism. Whether the same interior optimum (Parent+Leaf beats
both broader and tighter) holds for *functional* repair is unmeasured.

**This is arguably Option C's single best remaining research question** — it is a real open question rather
than a reframing of a preempted one.

### Q-9 — How much does dropping annealing actually cost at 7B?
The core of Option B. F-1 establishes that the gain grows +2.1 → +2.5 from 124M to 355M and was never
tested at 7B for full-attention diffusion. [2512.06776](https://arxiv.org/abs/2512.06776) tested annealing
at 7B but for **block**-diffusion (different target) and found it *hurts* — and its ablation runs only
4,000 iterations. Two confounded variables (scale, adaptation target), neither resolved.

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
