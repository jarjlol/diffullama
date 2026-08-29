# Decisions the team needs to make

> Companion to [`OPUS_Audit.md`](OPUS_Audit.md). That document is the evidence; this one is the set of
> forks in the road, with options laid out. Same scratch branch, same disclaimer — delete once settled.
>
> **Last updated:** 2026-08-29

---

## D-0. THE BIG ONE — which project? (blocks everything else)

**Options B and C are different papers. You cannot write both.** Pick one as primary.

### Option B — "What does AR→diffusion adaptation actually cost?"

Three questions, all forced through DiffuLLaMA specifically:

1. **Annealing at scale.** The anchor paper omitted attention-mask annealing for the 7B model, validating
   that choice only at 124M/355M — where the benefit *grows* with scale (+2.1 → +2.5). They call it
   "minimal impact" and extrapolate ~20× past their largest test. Nobody has tested annealing at 7B for
   full-attention diffusion. ([2512.06776](https://arxiv.org/abs/2512.06776) tested it at 7B but for
   **block**-diffusion, a different target, and found it *hurts* — sharpening the question rather than
   answering it.)
2. **Mask token.** DiffuLLaMA reuses vocabulary token 811 as `[MASK]`; DiffuGPT-S reuses 10541; but
   DiffuGPT-M gets a *proper new* token — and is also the better performer. Confounded with size, never
   disentangled, never ablated.
3. **Vestigial any-order capability.** Every published study of generation order uses *from-scratch* models
   (LLaDA, Dream). None studies an *adapted* one. Does adaptation leave any-order ability vestigial?

**For it:** exact course fit (reproduce-then-interrogate the anchor, stages 2+3 at once); low engineering
risk; interesting whichever way each lands; three independent shots so one dying doesn't kill the paper;
low preemption risk.

**Against it:** less flashy than a method paper; requires actual training runs (though well within the
hardware); "we audited someone's engineering shortcut" is a less exciting elevator pitch than "we invented
a repair method."

### Option C — Structure-Guided ReMasking (Neel's design doc, reframed)

Keep the direction, but with the corrections in `OPUS_Audit.md` §8.3, §3.1, §3.2, §4.1.

**For it:** the team has already invested design work; the execution-grounded seeding gap is verified
genuinely open; the repair setting (iterate-until-pass on dLLMs) doesn't exist anywhere.

**Against it:** CDC ([2605.16829](https://arxiv.org/abs/2605.16829)) preempted the mechanism — the delta is
now one variable (static→dynamic seed) on someone else's pipeline, while still requiring ~80% of their
machinery plus a sandbox and dynamic tracing they didn't need. CDC's own Fig. 8(b) shows backward slicing
(the doc's core signal) *underperforms* a tight AST neighbourhood. Highest engineering risk of any option,
two known silent-failure modes, and everything rides on one already-dented claim.

### Option E — the scientific-infilling pivot — **searched 2026-08-29: NARROW, does not displace B**

Use DiffuLLaMA's distinguishing capability — infilling without prompt reordering — on a **scientific** task.

**Search caveat, stated plainly:** this was *not* the full novelty search this option deserved. Two
background agents died on an API spend limit; what follows comes from two targeted searches run directly.
Treat the verdict as well-founded on the crowded sub-areas and weakly-founded on the "no direct hit" part.

**The obviously-scientific sub-areas are crowded:**
- *Molecular / SMILES:* TGM-DLM (AAAI 2024), GenMol (2025), PepTune, DiffBP, plus dedicated
  diffusion-driven domain adaptation for 3D molecules ([2404.00962](https://arxiv.org/abs/2404.00962)).
- *Protein / peptide / RNA:* guided discrete diffusion for protein design, classifier-guided antibody
  sequence generation, RNADiffFold. Established lines with purpose-built models — a course project would be
  competing with domain labs on their own turf.

**The scientific-text angle is partly taken:** [2606.19475](https://arxiv.org/abs/2606.19475) *"Diffusion
Language Models: An Experimental Analysis"* is a systematic evaluation already spanning "encyclopedic, news,
scientific, and benchmark text, including PubMed and ArXiv" — which absorbs much of the benchmark
contribution that looked open.

**No direct hit** on the exact combination (AR-adapted dLLM + infilling + scientific task). But absence
after two searches is weak evidence, and even if genuinely open, the claim is **thinner than Option B's** —
B rests on a specific inconsistency quotable from the anchor paper; E would rest on "nobody happened to try
this yet."

**Verdict: NARROW.** Does not displace B. Revisit only if B is blocked for an unrelated reason.

### Two papers worth reading regardless of which option wins

- **[DreamOn, 2602.01326](https://arxiv.org/abs/2602.01326)** — *Diffusion Language Models for Code
  Infilling Beyond Fixed-size Canvas.* Directly attacks **length rigidity**, which is Risk 3 (High) in the
  design doc: "masking *K* tokens regenerates exactly *K*; fixes needing more tokens are unreachable." If
  Option C is chosen this is required reading and may partly solve that risk.
- **[2606.19475](https://arxiv.org/abs/2606.19475)** — systematic dLLM evaluation; useful baseline reference
  for any direction.

### Recommendation

**Option B.** E was checked precisely because it might have beaten B on venue fit — it doesn't. It is
crowded where it is clearly scientific, and thin where it might be open. B retains the strongest
combination: a quotable inconsistency in the anchor's own text, three independent shots, exact course fit,
and low engineering risk. C remains viable but is the highest-effort, smallest-claim option.

### ⚠️ Work is on hold pending this

Per the decision to hold: **no direction-specific work has been started.** B's tasks and C's tasks are
different reproduction targets (B needs DiffuGPT *training*; C needs DiffuLLaMA/LLaDA/Dream *inference*),
so there is no shared head start to take. Only direction-independent prep is proceeding.

---

## D-1. Venue — **Agents4Science's status is a problem**

The project brief names the Stanford Agents4Science workshop as the target. Verified:

- **Agents4Science 2025** — submissions closed 5 Sept 2025, virtual event held **22 Oct 2025**. Done, ~10
  months ago.
- **No 2026 edition found.** Searched; nothing announced.
- A *different* conference, AI Agents4Qual 2026 (qualitative research), had a March 2026 deadline — also
  past, and not the same venue.

Its AI-authorship rule is confirmed and matches what the instructor said: *"AI authorship is not only
allowed but required"* — AI as both primary author and reviewer.

**Decision needed:** ask the instructor what the actual target is. Possibilities:
1. A 2026/2027 Agents4Science edition exists but isn't publicly announced yet — if it follows the 2025
   pattern (Sept deadline, Oct event), a Sept 2026 deadline is **~2 weeks away** and impossible for this
   project.
2. The brief's venue reference is aspirational and the real deliverable is the manuscript itself.
3. A different AI-for-science venue is intended.

This changes the *deadline*, not the work — but it changes how much polish the final artefact needs, and
whether "AI as primary author" is a hard formatting requirement or a stylistic goal.

---

## D-2. Sharanga — needs instructor approval, but the specs are excellent

Confirmed from the [official configuration page](https://sharanga.hpc.bits-hyderabad.ac.in/docs/misc_docs/configuration/)
(BITS Hyderabad, 58 nodes total, 8 GPU nodes):

| Node | GPUs | Memory | Interconnect |
|---|---|---|---|
| gpu8 | **6× RTX PRO 6000 Blackwell 96 GB** | 1 TB | HDR |
| gpu7 | **8× H200 NVL 141 GB** | 1 TB | HDR |
| gpu5–6 | 4× H100 SXM5 80 GB each | 1 TB | HDR |
| gpu4 | 8× A100 SXM4 80 GB | 1 TB | HDR |
| gpu1–3 | 1–2× V100 32 GB | 256 GB | EDR |

Partition names and job time limits are **not** documented on that page — ask when requesting access.

**What this changes:**
- Option B's third-scale training run (774M–1.5B) becomes comfortable rather than tight. gpu4 or gpu7 would
  handle it easily.
- It does **not** revive bet 1 (multi-GPU decode). That died on a false premise and on dInfer/Sangam having
  published the core claim — hardware was only the third reason. 8×A100 would make the *experiment*
  runnable, but the contribution would still be preempted.

**Decision needed:** confirm with the instructor whether BITS Goa students may use the Hyderabad cluster,
and under what account/queue. **Do not plan around it until confirmed** — the local 2× RTX 6000 Pro
Blackwell (192 GB) is sufficient for Option B regardless, so this is an accelerator, not a dependency.

---

## D-3. SOTA assignment — one clear addition available

**Deadline still not announced.** The work is done and unsubmitted.

The instructor's email offered **two** papers. Only one was implemented:

| Paper | Role | Status |
|---|---|---|
| SurveyGen / QUAL-SG ([2508.17647](https://arxiv.org/abs/2508.17647)) | **generation** method | ✅ implemented (`litreview/`) |
| LLMs for Automated Literature Review ([2412.13612](https://arxiv.org/abs/2412.13612)) | **evaluation** framework (reference generation, abstract writing, review composition) | ❌ not implemented |

**Option:** implement the second paper's evaluation framework and apply it to the QUAL-SG-generated survey.
This uses both papers the instructor provided and turns a single-method implementation into
generate-then-evaluate. Best value per effort of anything remaining on this assignment.

**Decision needed:** is it worth the effort before the deadline is even announced, or submit as-is? The
assignment is already substantively complete (retrieval + generation + overlap comparison, all 48 references
verified real, plus the LLM-judge upgrade taking F1 0.222 → 0.272).

---

## D-4. Smaller calls

| # | Question | Notes |
|---|---|---|
| a | If Option C: 2 or 3 backbones? | The §4.1 sampler correction changed the cost of this — Dream needs an explicit `alg="entropy"` guard either way |
| b | Who owns which workstream? | Five people; Option B's three sub-questions parallelise cleanly (annealing / mask-token / order-metrics) |
| c | Does the reproduction stage target training or inference? | Depends entirely on D-0; they are different targets |
| d | Delete this branch when? | `git push origin --delete audit/opus-review` once D-0 is settled and any keepers are copied to `main` |

---

## Status of everything so far

| Item | State |
|---|---|
| Part 2 assignment PDF | ✅ submitted |
| SOTA litreview implementation | ✅ done, ❌ unsubmitted (no deadline announced) |
| LLM-judge extension to litreview | ✅ done, pushed to `main` |
| Design doc audit (3 parts) | ✅ done, pushed to this branch |
| Off-by-one invariant test | ✅ written, run, **passing** (`audit/test_shift_offset.py`) |
| Hardware confirmation | ✅ 2× RTX 6000 Pro Blackwell local; Sharanga specs known, access unconfirmed |
| Agents4Science status | ⚠️ 2025 edition is over; no 2026 edition found |
| Direction decision | ⏳ **blocking — this document, D-0** |

Everything above is pushed and visible on GitHub: `main` (litreview work) and `audit/opus-review` (this
audit, the test, and this document).
