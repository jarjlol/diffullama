# P-1 direction analysis — 2026-09-19

> **Status:** written input to the pending team decision **P-1** (which direction the main project takes).
> Not a decision. Produced by an AI agent session run by Aaditya on 2026-09-19 against `main` @ `066d8f2`,
> `plan/execution-grounded-repair` @ `35a6486`, and `audit/opus-review` @ `9038db1`. Confidence labels follow
> `03-established-facts.md`: 🟢 PRIMARY (read the source / ran the code), 🟡 SECONDARY (fetched via a tool
> summary or a second-hand table), 🔴 WEAK (arithmetic, recollection, or absence-of-evidence). Every paper
> cited resolves to a working arXiv link that was opened during this session (§6).

## 0. Bottom line

1. **Recommend Option B**, sharpened into one falsifiable question about attention-mask annealing at scale
   (§3), as the resolution of P-1. Confidence about 75%.
2. **Do not make EGR (Option C) the main project.** Three independent reasons, each verified below:
   - It was preempted again on 31 Aug 2026. **DiffPDE** ([arXiv:2608.30532](https://arxiv.org/abs/2608.30532))
     runs a generate → execute → localize-from-execution-feedback → remask → infill → retest loop on a 7B
     diffusion code model. The sentence in `egr/README.md` §3 ("Execution feedback as the remasking anchor —
     nothing in the literature does this") is no longer true as written.
   - The team's doubt about a post-inference method is well founded on course-fit grounds: a frozen-model
     controller wraps any masked diffusion LM and tests nothing about the anchor paper's actual contribution,
     which is the adaptation recipe.
   - The backbone is weaker than EGR's own documents state. The anchor paper *does* report a HumanEval
     single-line-infilling pass@1 for the released DiffuLLaMA checkpoint: **15.5%** (Table 1). EGR's README
     and `novelty.md` say the only infilling number belongs to Diffu-CodeLLaMA; that is wrong (§1, row 4).
3. The EGR harness is well engineered and its hazard tests pass, but it **does not run on Windows** (§1, row 6),
   and its one reported result (P-6) rests on 6 hand-written fixtures. Keep it as a side artifact.

---

## 1. Independent verification pass

Claims from the task brief (its Part 3) and from the EGR branch, checked against the branch, the anchor's
extracted full text (`litreview/data/anchor_fulltext.txt`), and the CDC PDF.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | EGR is "a complete (but unimplemented — zero code written) design" | **False today** | Branch tip `35a6486` (2026-09-19) contains `egr/*.py` (~3,600 lines incl. tests): `canvas.py`, `verify.py`, `_sandbox_harness.py`, `trace.py`, `evidence.py`, `localize.py`, `policy.py`, `backend.py`, `loop.py`, `cli.py`, benchmarks, 8 test files. Commits `918e1dc` and `35a6486`. The P-5 entry ("No code has been written") is stale; the P-6 entry in the same file records the implementation. 🟢 |
| 2 | P-5 lives in `project-docs/02-decision-log.md` | **Only on the branch** | `main` has neither P-5 nor P-6; both are in the +88-line diff on `plan/execution-grounded-repair`. 🟢 |
| 3 | The paper reports no whole-function HumanEval pass@1 for the released checkpoint | **True** | The only code column in Table 1 is HumanEval *single-line infilling*; no HumanEval/MBPP generation result anywhere in the text. The intro bullet nonetheless claims "code generation" as a capability (L9 in `limitations/`). 🟢 |
| 4 | "Its only code result is HumanEval single-line infilling (Appendix C.3), and even that number belongs to a separately trained Diffu-CodeLLaMA" | **Half wrong** | Table 1, "Infilling / Code" column: **DiffuLLaMA 15.5**, LLaMA2 (1.7, prefix only), DiffuGPT-M 2.9, GPT2-M (2.6), DiffuGPT-S 0.3 — pass@1 in %. Table 8 (Appendix): CodeLLaMA FT-SPM 0.80, FT-PSM 0.74, **Diffu-CodeLLaMA 0.76** — a separate 100M-token Starcoder experiment. So the released checkpoint has its own number, and it is low. `egr/README.md` §2, `docs/02-experiment-plan.md` §2 and `novelty.md` §3 all repeat the error; the README's aside "(with the adjacent GPT2-M 'Code' column sitting at 2.6)" reads GPT2-M's prefix-only infilling score as evidence about whole-function generation. Consequence: the "documented strength" is *relative* to an AR baseline that cannot see the suffix; in absolute terms the backbone fills one masked line correctly 15.5% of the time with gold prefix and suffix. 🟢 |
| 5 | README states the CDC follow-up framing ("static witness … dynamic witness … matched-budget policy comparison CDC does not") | **Present verbatim** | `egr/README.md` §3, `novelty.md` §5. Now stale after DiffPDE (§2.2). 🟢 |
| 6 | "GPU-free first result … reportable in week one … work the four without workstation access can do in full" | **Design says so; not reproducible here** | P-6's table is over **n = 15 mutants from 6 hand-written fixtures** (`egr/benchmarks/mutants.py`), not HumanEval+. I ran the branch on Windows 11 / Python 3.13.2: `pytest egr/tests` → **27 passed, 6 failed** (33 collected); `unittest discover` → 28 run, 6 failures (deploy.md claims 28/28 on Python 3.14; not testable here). All six failures are the sandbox: (a) `egr/types.py` shadows the stdlib `types` module because `_sandbox_harness.py` is launched as a script, so `sys.path[0]` is `egr/` (`ImportError: cannot import name 'MappingProxyType' from 'types'`); (b) `import resource` is unconditional and the module does not exist on Windows. `python -m egr.cli localize` therefore hit `HARNESS_ERROR` on 9 of 15 mutants and scored only the 6 syntax mutants. The H-6 guard worked (harness errors were excluded, not counted), which is to the design's credit, but the claim that Phases 0–2 run on any laptop is false for Windows laptops until both are fixed (a module rename and a guarded import). 🟢 |
| 6b | P-6's numbers | **Two artifacts inside them** | `confidence` top-1 = 0.40 = 6/15 is exactly the six syntax mutants: `MockBackend.confidence()` is a constant, `min()` returns canvas position 1 → line 1, and every syntax mutant's ground truth is line 1 (the colon dropped from `def …:`). `ours` on syntax mutants is tautological: `SyntaxError.lineno` *is* the mutated line by construction. The informative rows are the 9 assertion-kind mutants on 2–6-line functions, which I could not run. P-6 already labels itself preliminary; this makes the label concrete. 🟢 |
| 7 | "~5 GPU-hours to a first repair curve, ~46 GPU-hours total, on 2× RTX 6000 Pro" | **Matches the docs, unmeasured** | `docs/02-experiment-plan.md` §8: ~3 h for Track A HumanEvalFix (the README diagram says ~5 h), ~46 h total, explicitly "arithmetic, not measurement" (44 TFLOP/s effective assumed, T = 32, L = 512, 55% of calls made). Plausible within 2–4× either way; Q-7 remains open. Both that document and `01-project-brief.md` say **plan for one 96 GB card, not two** — the brief in this task said "2×". 🟢 for what the docs say, 🔴 for the number |
| 8 | §10 names three risks: backbone weakness, deterministic stall, two silent failure modes with tests | **Present** | `egr/README.md` §10; `docs/03-hazards.md` H-1/H-2/H-4; `loop.py` varies the seed with depth and escalates scope on a repeated evidence signature. I re-ran `egr/tests/test_real_tokenizer.py` today against the real `diffusionfamily/diffullama` tokenizer (transformers 5.16.1): **26 nodes, 26/26 covered, 22/26 (85%) bleed one leading space, whitespace only** — F-21 re-confirmed. The shift-invariant tests pass on my machine. 🟢 |
| 9 | Closing section says it does not settle P-1 | **True** | `egr/README.md` §11. 🟢 |
| 10 | Option B's factual basis: annealing dropped at 7B, ablation gain +2.1 → +2.5 | **True, and one cost-relevant detail added** | Table 3: DD-w/o-anneal 43.3 / 47.2 vs DD 45.4 / 49.7. Verbatim: "The mask annealing has minimal impact, so we choose to omit it for 7B adaptation to simplify implementation using flash-attention 2." The ablation is a **fine-tuning proxy** — "Direct ablation on adaptation training is costly; hence, we conduct preliminary experiments … finetune models on the augmented GSM8K symbolic dataset" — from GPT-2 weights, not the 100B+-token pre-training. Appendix C.2 Table 5 adds ELBO: GPT2-M DD 0.015 / 49.7, no-anneal 0.019 / 47.2, no-shift 0.028 / 34.5. This is what makes B affordable (§3). 🟢 |
| 11 | CDC differentiation (F-8, F-9, F-13) | **Holds; Q-5 answered for now** | [arXiv:2605.16829](https://arxiv.org/abs/2605.16829) is still **v1 (16 May 2026), no venue** as of today. From the PDF text: GradGuide's evaluator is "an auxiliary surrogate g_φ trained ahead of time with execution-driven labels" (Qwen2.5-1.5B on CodeContests); functional-correctness results are **C++ only** (HumanEval-X C++ 164 tasks, MBPP-C++ 397); MDFI uses static analyzer witnesses on CWEval / LLMSecEval+ (C, C++, Go, JS, Python); backbones DiffuCoder-7B and Dream-Coder-7B. Fig. 8(b) verbatim: "the deployed Parent+Leaf neighborhood (34.3) beats tighter (Token-Window 24.1) and looser (Use–Def Slice 26.9) alternatives" — metric fs@1, CWEval. No test execution at inference, no iterate-until-pass loop, no confidence-remasking *baseline* (entropy and confidence appear only as terms inside GradGuide's saliency, Eq. 24). 🟢 |
| 12 | F-22's unresolved number (48.95 vs 54.94) | **Resolved** | NBDiff v2 Table 2, openPangu-7B averages: Annealed Attention Mask **44.34**; Plain Finetuning **48.95**; +AR loss 52.97; NBDiff 54.94. The 48.95 was the plain (direct target-mask) row; W-2's "−4.6" is 44.34 − 48.95. Their annealed baseline is "the random interpolation between structured attention masks for blocksize=1 and blocksize=32", i.e. not DiffuLLaMA's schedule. 🟡 (HTML v2 through a fetch summariser, two consistent reads) |

---

## 2. Research-gap analysis

### 2.1 What is new since the 2026-08-31 audit

Primary-source facts established this session that the project docs do not yet contain:

- **DiffPDE** ([2608.30532](https://arxiv.org/abs/2608.30532), 31 Aug 2026, Guo et al.). Backbone **Dream-Coder-7B**,
  trained with SFT then "Iterative Debugging GRPO" (350 GPU-hours on 8× A800). Inference: "a generate–execute–debug
  loop … if the solver fails … execution feedback is used to localize the problematic region, re-mask the
  corresponding span, and trigger an infilling-based repair step. This loop continues until an executable
  solver is found or the repair budget is exhausted." Localization (Appendix A): "Parse the execution error
  message and extract the failing line number L … Expand outward from L and terminate at the first natural
  boundary above and below, defined as a blank line, a comment-only line, or a line with indentation strictly
  shallower than that of L." Budget N = 32 candidates, k = 3 debugging attempts. Table 7 (same checkpoint):
  localized infill vs full regeneration, pass@32 0.4938 → 0.5063, latency 64.85 → 0.748 s per attempt.
  Domain: PDEBench only. Cites neither CDC nor CDLM; no Ochiai, no AST scoping, no random/confidence/static
  baselines. Its Limitations section scopes it to failures "where execution feedback can be used to identify
  and revise specific code spans". 🟢
- **Dream 7B** ([2508.15487](https://arxiv.org/abs/2508.15487)) used **no attention-mask annealing**: the term
  does not occur in the paper; it adopts DiffuLLaMA's shift operation and notes "AR initialization also
  experiences a high loss at the beginning due to the transition from causal attention to full attention".
  580B tokens, Qwen2.5-7B configuration. 🟢 — and **LLaDA2.0** ([2512.15745](https://arxiv.org/abs/2512.15745))
  misattributes annealing to Dream ("DiffusionLLaMA and Dream-7B adopt a mask annealing strategy"), a
  reminder not to take adaptation-recipe claims second hand. 🟡
- **DiffuCoder** ([2506.20639](https://arxiv.org/abs/2506.20639), 7B, 65B + 65B tokens): "the adaptation approach
  from Gong et al. (2025)", no recipe detail, no annealing ablation. 🟡
- **LLaDA2.0**: block-size curriculum 1 → 4 → 32 → 64 → 4096 at 16B / 100B, final model block diffusion, no
  ablation against direct training. 🟡
- **PreDiff-LM** ([2607.25157](https://arxiv.org/abs/2607.25157), Jul 2026): a hybrid mask (causal on the prompt,
  bidirectional on the target) beats uniform bidirectional attention at GPT-2 Medium (perplexity 34.1 → 28.7),
  composes with "a DiffuGPT-style objective adaptation" (26.9). 355M only. 🟡
- **FLUID** ([2605.27387](https://arxiv.org/abs/2605.27387), ACL 2026) and **WeDLM** ([2512.22737](https://arxiv.org/abs/2512.22737))
  adapt or decode with strictly causal attention — evidence that the attention-pattern question is live. 🟡
- **2510.04605** (Oct 2025) evaluates a diffusion LM (Mercury 7B, commercial) on Defects4J / Bears / SWE-bench;
  repair by generation, no remasking loop. So "no repair-benchmark numbers for dLLMs" is false in general,
  though still true for open adapted checkpoints on HumanEvalFix. 🟡
- **Detect-Remask-Repair** ([2606.12807](https://arxiv.org/abs/2606.12807)) is the same loop shape for summaries. 🟡
- **HumanEvalFix** ([2308.07124](https://arxiv.org/abs/2308.07124)): 164 problems per language, bugs added by hand,
  "written such that the code still runs but produces an incorrect result leading to at least one unit test
  failing" — i.e. assertion failures without a traceback line inside the function. 🟡

### 2.2 Option C / EGR after DiffPDE

EGR's five claimed gaps (`egr/README.md` §3), re-scored:

| EGR's gap | Status on 2026-09-19 |
|---|---|
| Execution feedback as the remasking anchor — "nothing in the literature does this" | **Closed as written.** DiffPDE does it, with a traceback-line anchor and an indentation-boundary span. |
| An iterate-until-pass repair loop over a diffusion LM | **Closed.** DiffPDE's loop (k = 3); CDLM-style revision exists too. |
| Repair-benchmark numbers for this model family | **Partly closed.** 2510.04605 reports a dLLM on Defects4J/Bears/SWE-bench. What remains is a *number* for open adapted checkpoints, not a *mechanism*. |
| A confidence-remasking baseline at matched budget | Still open. |
| Does CDC's scope optimum transfer to functional bugs (Q-8)? | Still open. CDC's functional results are C++-only and go through a surrogate. |

What survives is real but narrow: a **frozen** model (DiffPDE fine-tunes and runs RL); **assertion-type**
failures with no failing line inside the function (exactly HumanEvalFix's bug class, and exactly what
DiffPDE's traceback anchor cannot see); spectrum-based (Ochiai) plus AST-scoped localization; the
matched-budget policy comparison; Python general-purpose benchmarks. The honest framing would now be:

> DiffPDE showed that a traceback-anchored span is a good remasking target for runtime errors on a
> fine-tuned diffusion coder. We ask whether spectrum-based localization from *failing assertions* can anchor
> remasking on a *frozen* adapted model, and how it compares at matched budget with traceback-window,
> confidence, random and full-resample policies.

Three problems with that surviving claim, in decreasing order of weight:

1. **The surviving components are the ones with the weakest prior.** Ochiai over 3–7 tests on 5–15-line
   HumanEval functions is nearly flat: almost every test executes almost every line, so the ranking collapses
   to execution-count and line-number tie-breaks. The GPU-free Phase-2 study would most likely show `ours` ≈
   `random` on assertion mutants. The `static` baseline (RQ3, "the paper's core claim") is ill-posed for
   functional bugs: a static analyzer has no witness for a flipped comparison operator, and EGR's stand-in is
   "the entry point's most deeply nested statement". A win over it is not a test of CDC's mechanism.
   Meanwhile the traceback-window policy from `08-trace-guided-repair-proposal.md` is now DiffPDE's published
   method and would have to be a first-class baseline.
2. **The post-inference objection is right.** Stage 2 of the course format reproduces the anchor's results;
   stages 3–4 enhance its idea and try to beat its baseline. An inference-time controller does none of this:
   it never touches adaptation, the reproduction it would perform is a 15.5% infilling number, and "beat the
   paper's baseline" has no referent because the paper never ran a repair or generation benchmark. A
   backbone-agnostic controller also makes the anchor interchangeable — the experiment plan already moves the
   headline to Dream-Coder for that reason.
3. **The gap has now been narrowed four times in four weeks** (SGR → C2 → EGR → post-DiffPDE). This is a
   subfield publishing weekly; the project's own principle 5 says re-check at every milestone, and each
   re-check has removed a leg. Two of the three original novelty pillars are gone; the third (confidence
   baseline) is a control, not a contribution.

Add the verified engineering debt: no Windows support; a placeholder corpus; a constant "confidence" signal;
`_budget_trim` does not enforce the budget it names (deploy.md §5 row 5); Track B and the Dream backend do not
exist; and the fixed-length-infilling ceiling (H-3) that MockBackend already hit — which DreamOn
([2602.01326](https://arxiv.org/abs/2602.01326), ICLR 2026) addresses only for Dream-Coder/DiffuCoder, not DiffuLLaMA.

What EGR still has going for it, stated fairly: the harness design is careful (typed verdicts, harness errors
excluded, seed-by-depth, scope escalation, regression tracking); the hazard tests exist and pass where the
sandbox runs; a negative result would still be reportable; four members could contribute without a GPU once
the portability fixes land; and its GPU need is lower than B's. If the team overrides this recommendation and
chooses C, the minimum changes are: fix the two portability defects; load the real EvalPlus/HumanEvalFix data;
give the `confidence` column a real model; add DiffPDE's window policy as a baseline; delete the "nothing in the
literature" sentence and cite DiffPDE, CDLM, 2510.04605 and 2606.12807; and state plainly that the anchor is
one of two backbones rather than the object of study.

### 2.3 Option B after this pass

The precise empty cell is: **a controlled with-versus-without attention-mask-annealing comparison for
full-attention masked-diffusion adaptation at any scale above 355M.** Everything found this session:

| Work | Scale / tokens | Target | Annealing | With/without comparison |
|---|---|---|---|---|
| DiffuLLaMA proxy (Table 3) | 124M, 355M fine-tune | full-sequence | 10K-step random causal → full | Yes: +2.1, +2.5 |
| DiffuLLaMA 7B | 7B, 65B | full-sequence | omitted for flash-attention 2 | No |
| Dream 7B | 7B, 580B | full-sequence | none | No |
| DiffuCoder 7B | 7B, 130B | full-sequence | unstated | No |
| NBDiff | 4B–8B, ~30B | block (32) | modified (block-1/block-32 interpolation) | Yes: 44.34 vs 48.95 plain — hurts |
| LLaDA2.0 | 16B, 100B | block (32) final | block-size curriculum | No |
| PreDiff-LM | 355M | full-sequence | hybrid mask, not annealing | Yes: helps (ppl 34.1 → 28.7) |

Every full-attention adaptation at 7B skipped annealing and none measured what that cost. The only ≥7B
measurement is on a block target with a modified schedule and shows harm; the only sub-1B attention-pattern
ablations (DiffuLLaMA, PreDiff-LM) show help. Two confounds — scale and adaptation target — and the anchor sits
exactly in the untested cell. Q-9's framing is confirmed, and slightly sharpened: Dream is now documented as a
no-annealing 7B run, and PreDiff-LM adds a 2026 small-scale data point on the "helps" side.

A second cell sits inside the first. The proxy the authors used (GSM8K-symbolic, a left-to-right target)
under-exercises the capability annealing exists to create, which is use of right context. The one place
DiffuLLaMA is distinctive in its own Table 1 is infilling (15.5 vs 1.7). **Nobody has measured the annealing
gain on an infilling task at any scale**, and the paper's own infilling protocol (ROCStories 1,000 cases;
HumanEval single-line infilling 1,033 cases; Appendix C.3) makes that cheap to add to every checkpoint.

### 2.4 Other candidates considered

- **L2, the shift ablation at scale** (−11.9 / −15.2 at 124M/355M): five times larger than annealing, but
  "no shift" is a strictly worse variant nobody ships, and Dream, DiffuCoder and NBDiff all keep it. An optional
  arm, not a question.
- **L3 / W-8, the mask-token confound**: cheap and real; kept as a secondary 2×2 at 124M/355M (§3). The
  "low frequency" justification for tokens 811 and 10541 has never been measured (F-2) and is a CPU-only task.
- **W-10, any-order behaviour in adapted vs from-scratch models**: inference-only, machinery published
  ([2601.15593](https://arxiv.org/abs/2601.15593), [2608.05687](https://arxiv.org/abs/2608.05687)); the weakest anchor
  tie of the three B sub-questions. Optional.
- **L8 (uncertainty / verifier) and any decoding-time idea**: same course-fit problem as EGR.
- **L10 (cross-architecture transfer)**: needs a third AR family and a continual-pre-training budget. Too expensive.
- **Option E**: not re-searched; D-2026-08-29-b stands.

None displaces B: B's inconsistency is quotable from the anchor's own text, and the 2026 literature has
widened the cell rather than closed it.

---

## 3. The research problem

> DiffuLLaMA's adaptation recipe was validated by a fine-tuning proxy at 124M and 355M (Table 3), where
> attention-mask annealing was worth +2.1 and +2.5 accuracy points and growing, and was then omitted at 7B on
> the grounds of "minimal impact" and flash-attention convenience — a twentyfold extrapolation in the direction
> the trend argues against. No controlled with/without-annealing comparison exists above 355M for
> full-attention masked diffusion: Dream-7B and DiffuLLaMA-7B both skipped it without measuring it, DiffuCoder
> inherits the recipe without stating it, NBDiff's 7B result uses a modified schedule on a block-diffusion
> target (where it hurts by 4.6 points), and PreDiff-LM ablates a different attention intervention at 355M
> only. **The problem: under the anchor paper's own ablation protocol (discrete-diffusion fine-tuning from AR
> weights on GSM8K-symbolic), does the annealing gain Δ(s) = Acc(DD) − Acc(DD without annealing) grow,
> plateau, or reverse as backbone scale s rises from 124M and 355M (GPT-2 S/M) through 774M and 1.5B (GPT-2
> L/XL) to 7B (LLaMA-2), and is Δ larger on the paper's own infilling tasks — where right-context use is
> load-bearing — than on the left-to-right GSM8K proxy the authors used to dismiss it?** Three pre-registered
> outcomes, each reportable: H1, the authors' extrapolation holds (Δ(7B) within ±1 point of zero); H2, the
> small-scale trend continues (Δ(7B) ≥ Δ(355M) = 2.5, so the released recipe left accuracy on the table); H3,
> the block-diffusion reversal generalises (Δ(7B) < 0, extending NBDiff's critique to full-attention
> adaptation). Secondary, same runs: disentangle the [MASK]-token treatment from scale (reused token vs new
> token, crossed at 124M and 355M — the paper gave only DiffuGPT-M a new token, and it is also the better
> model) and measure the corpus frequency of tokens 811 and 10541, which the paper asserts is low and never
> measures. This resolves **Q-9** directly and **Q-2 / P-1** by choice; it leaves **Q-8** and **Q-12** open as
> future work under Option C.

**Design, scoped to the hardware the team controls (one contended 96 GB card).**

| Element | Choice | Why |
|---|---|---|
| Protocol | Table 3's: DD-loss fine-tune from AR weights on GSM8K-symbolic, DD vs DD-w/o-anneal, matched steps and seeds | Reproduces the anchor's numbers first (stage 2), then extends them (stages 3–4) |
| Scales | 124M, 355M, 774M, 1.5B (GPT-2), 7B (LLaMA-2) | Four points give a trend; 7B is the cell the paper's headline depends on |
| Annealing arm | LLaMA-Factory DDM trainer, `anneal_steps`, `get_anneal_attn_mask` — already in this repo (`LLaMA-Factory/src/llamafactory/train/ddm/trainer.py:154,642`) | The mechanism exists in-repo; it needs eager/SDPA attention with an explicit mask, i.e. no flash-attention 2 — the exact cost the authors avoided, negligible at L ≤ 512 |
| 7B arms | LoRA r = 8 as the anchor did for DiffuLLaMA's GSM8K fine-tune (Appendix C), or full-parameter with an 8-bit optimizer and gradient checkpointing (fits in 96 GB) | Protocol-consistent; full-parameter is the stronger test if time allows |
| Evaluation | GSM8K test accuracy (1,319 problems) plus the paper's infilling protocol on every checkpoint (ROCStories 1,000; HumanEval single-line infilling 1,033) | The infilling axis is the part nobody has measured |
| Primary endpoint | Slope of Δ(s) on log s across ≥ 4 scales, on each task family | More powerful than any single-scale contrast |
| Secondary | Per-scale paired contrasts with Wilson intervals; seeds: 3 at 124M/355M, 2 at 774M/1.5B, 1–2 at 7B | Honest power: per-arm SE ≈ 1.4 points at n = 1,319, so a 2.5-point single-seed contrast at 7B is suggestive, not decisive |
| Secondary arms | Mask-token 2×2 at 124M/355M (4 runs); token-frequency count on a FineWeb/SlimPajama sample (CPU) | Resolves L3/W-8 |

**Compute — arithmetic, not measurement (Q-7 still applies).** Assume the DoT-repo default of 120K steps at
batch 128 (the anchor "follows Ye et al. (2024b)"; its exact step count is unstated), ~150 tokens per example,
so ~2.3B tokens per run, and ~200 TFLOP/s sustained on one RTX 6000 Pro Blackwell (unmeasured). Per run:
124M ≈ 2.5 h, 355M ≈ 7 h, 774M ≈ 15 h, 1.5B ≈ 30 h, 7B ≈ 90–140 h at full length. Plan: full-length paired
runs with seeds at 124M–774M (~40 h), paired at 1.5B (~60 h), and a **reduced 30K-step paired schedule at 7B**
(~25–35 h per arm, ~60 h) — matched between arms, so the contrast is fair even if the absolute number is not
converged. About **160 GPU-hours** on one card, three to four weeks of wall-clock at half availability, with
checkpoint-and-resume mandatory. The stage-2 reproduction (Table 3's four GPT-2 rows) is under a day and the
124M rows fit a 4–6 GB laptop GPU. One hour of measurement on the real card should replace all of these
numbers before scheduling.

**Work split.** GPU-bound runs: Aaditya, Aryan, Arjun. Everyone else: the infilling evaluation harness
(Appendix C.3 protocol), the mask-token frequency count, the 124M reproduction on a laptop, a weekly novelty
re-check (the field moved twice in four weeks), analysis and writing.

---

## 4. Recommendation on P-1

**Option B, sharpened as in §3.** Confidence about 75%.

Why B over the EGR form of C, in one line each: B interrogates the anchor's own recipe and reproduces its
own table; C wraps a frozen model and reproduces a 15.5% number. B's cell has widened since August; C's has
narrowed four times. B's failure mode is a null with wide intervals, mitigated by the trend endpoint and the
infilling axis; C's failure mode is a flat curve on a weak backbone, mitigated only by switching backbones.
B's engineering risk is a custom attention mask; C's is a sandbox, a tracer, an offset mapping and two silent
failure modes, one of which already bit on Windows.

What would change my mind:

- A paper with a controlled with/without-annealing ablation at ≥ 1B for full-attention adaptation. I searched
  for it today (seven web searches, fifteen targeted fetches) and found none; the search was not exhaustive.
- GPU access collapsing below roughly 100 hours for the semester. B's core still survives at 124M–1.5B, but
  the 7B cell is the one that matters.
- The instructor or industry collaborator stating a preference for a method paper over an audit paper. That
  is a judgment the repo cannot settle; ask TTV when confirming the venue (Q-1).

What to do with the EGR branch: keep it, do not merge it into the main-project plan. Fix the two portability
defects if anyone wants to keep running it, and record the P-6 caveats above. Its hazard register and the
rescued `test_shift_offset.py` are worth keeping regardless of direction.

**Proposed log entries (for the team to append, not rewrite):** F-26 DiffPDE closes F-13 gap 1 as written;
F-27 Dream-7B used no annealing, LLaDA2.0 misattributes it; F-28 EGR sandbox fails on Windows (two causes);
F-22 discrepancy resolved (48.95 is the plain-finetuning row); Q-5 re-checked 2026-09-19, CDC still v1;
the Table 1 correction to `egr/README.md` §2; and a dated D-entry once P-1 is decided.

---

## 5. What I could not verify

- **P-6's table.** The sandbox does not run on this machine, so the 15-mutant result is unreproduced. The
  "28/28 passing" claim on Python 3.14 / a POSIX OS is untested here.
- **Every compute number** in §3 and in EGR's plan: FLOPs arithmetic on an unmeasured sustained throughput; the
  anchor's exact GSM8K-symbolic step count (the DoT repo default of 120K steps and batch 128 is the only
  source); the GSM8K-Aug example count (recollection, ~385K; 🔴).
- **NBDiff Table 2** and **2510.04605** details came through a fetch summariser (two consistent reads for NBDiff);
  I did not read those PDFs myself. 🟡
- **DiffuCoder's Stage-1 recipe**: the paper does not say whether annealing was used; its training code was
  not inspected.
- **Dream's "no annealing"** rests on the term's absence from the 17-page PDF text plus the quoted
  loss-transition sentence; if the recipe used annealing without saying so, this is wrong.
- **Novelty coverage** is what one day of searching gives: DiffPDE surfaced on the first query, so a similar
  paper on the annealing question could have been missed. Semantic Scholar and the ACL Anthology were not
  queried programmatically.
- **`SGR_DESIGN.md`** is not in the clone (untracked, never pushed); its content is known only through
  `audit/OPUS_Audit.md`.
- Whether TTV or Manasi Patwardhan prefer an audit-style or a method-style project.

---

## 6. Sources read this session

Anchor: [2410.17891](https://arxiv.org/abs/2410.17891) full text (repo copy). Branches: `plan/execution-grounded-repair`
(all of `egr/`), `audit/opus-review` (`DECISIONS.md`, `OPUS_Audit.md` §2–3, §5, §8, §18–19, §23). Full text
via PDF extraction: CDC [2605.16829](https://arxiv.org/abs/2605.16829); Dream 7B [2508.15487](https://arxiv.org/abs/2508.15487);
DiffPDE [2608.30532](https://arxiv.org/abs/2608.30532); code-diffusion survey [2606.23690](https://arxiv.org/abs/2606.23690).
Abstract or HTML through a fetch tool: NBDiff [2512.06776](https://arxiv.org/abs/2512.06776) (v2), CDLM
[2512.15596](https://arxiv.org/abs/2512.15596), DreamOn [2602.01326](https://arxiv.org/abs/2602.01326),
[2508.11110](https://arxiv.org/abs/2508.11110), OctoPack [2308.07124](https://arxiv.org/abs/2308.07124), DiffuCoder
[2506.20639](https://arxiv.org/abs/2506.20639), LLaDA2.0 [2512.15745](https://arxiv.org/abs/2512.15745), Efficient-DLM
[2512.14067](https://arxiv.org/abs/2512.14067), PreDiff-LM [2607.25157](https://arxiv.org/abs/2607.25157), OPDLM
[2606.06712](https://arxiv.org/abs/2606.06712), WeDLM [2512.22737](https://arxiv.org/abs/2512.22737), FLUID
[2605.27387](https://arxiv.org/abs/2605.27387), [2605.13026](https://arxiv.org/abs/2605.13026), Detect-Remask-Repair
[2606.12807](https://arxiv.org/abs/2606.12807), [2510.04605](https://arxiv.org/abs/2510.04605), DoT repo
(HKUNLP/diffusion-of-thoughts). Ran: `egr/tests` (pytest and unittest), `egr.cli localize`,
`egr/tests/test_real_tokenizer.py`, `_sandbox_harness.py` directly, on Windows 11 / Python 3.13.2.
