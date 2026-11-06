# Borrowing from the reference papers: a second, independent review

Written 2026-11-06 by Aryan's agent, after `BORROWED_IDEAS.md` (Aaditya's analysis). The question:
can we pick and mix ideas from DeepScientist, EvoScientist and AutoSOTA to make our Arbor codification
of idea 1.1 much stronger, and what do we change because of it?

Read `BORROWED_IDEAS.md` first. This file checks it, corrects it where needed, and adds what it
missed. The prompts in `ITERATION_2_PROMPTS.md` were amended to match (each amendment is marked
"Review 2026-11-06").

## Sources and how they were read

| Source | Version read | How |
|---|---|---|
| Arbor, *Toward Generalist Autonomous Research via Hypothesis-Tree Refinement* | arXiv:2606.11926v1 | full text, HTML and PDF, incl. App. A-B |
| DeepScientist, *Advancing Frontier-Pushing Scientific Findings Progressively* | arXiv:2509.26603v1 | full text, HTML and PDF, incl. App. A-C |
| EvoScientist, *Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery* | arXiv:2603.08127v2 | full text, HTML and PDF, incl. App. C-D (prompts) |
| AutoSOTA, *An End-to-End Automated Research System for SOTA AI Model Discovery* | arXiv:2604.05550v2 | full text, HTML and PDF, incl. §4.3 case studies |
| Arbor source code | `RUC-NLPIR/Arbor` tag `v0.1.4` (commit `65ffcc8`, 2026-07-11) | read, not run. v0.1.4 is the version on Blackwell per `HANDOFF.md`; not re-checked on the box |
| Our iteration-1 Arbor run | `results/iter1/arbor_session/` | `EXPERIENCE.md`, `findings.jsonl`, `run_stats.json`, `.coordinator/idea_tree.json` |

Section numbers below are the papers'. File paths in Arbor are relative to its repo root.

## 1. Verdict in short

1. **Aaditya's reading of the three papers is accurate.** I re-read every section `BORROWED_IDEAS.md`
   cites. Its claims and numbers match, with three small corrections (§2).
2. **Arbor already has native versions of several borrowed mechanisms, and in iteration 1 we used
   none of them.** It has cross-run memory (`EXPERIENCE.md` plus recall), a plateau detector that
   forces a "paradigm shift" (close to AutoSOTA's leap rule), runtime hash-checking of protected
   files (AutoSOTA's R2), and plugins with six prompt-injection points. Those points map one to one
   onto AutoSOTA's four layers of red-line enforcement. So we should build the borrowed mechanisms
   **inside Arbor as a project plugin**, not as side files that only reach Arbor as pasted text (§3).
3. **Two safety problems in the current iteration-2 plan:**
   - Arbor's own memory of iteration 1 (`EXPERIENCE.md`) lists cross-model consensus as a
     "[leverage]" finding. That is the direction we rejected twice. Reused as-is, it hands Arbor the
     banned idea as a prior.
   - `eval.py` does not enforce red line R7 (one item at a time). `select()` can keep module-level
     state across items, which is exactly how Arbor built the cross-model cache. (§3, §5)
4. **Two Arbor settings do not do what our files say:**
   - `merge_threshold` is not enforced at the merge gate. Any strict held-out gain merges.
   - Experience recall runs only in the interactive intake. It does not run under `arbor run`,
     which is how we launch. (§3)
5. **Process borrowing alone will not beat majority vote.** Arbor's own iteration-1 analysis
   (`EXPERIENCE.md`) found that every remaining in-item signal over the current features was
   anti-predictive. It put the ceiling for explainable rules at about 0.296-0.298 macro. A real
   gain needs **new information** (DLM self-likelihood, a verifier), **a higher ceiling** (more
   steps or samples at matched compute), or **a fixed pipeline** (the fidelity gap). The borrowed
   mechanisms are most useful when they point the search at those (§4, §6).

## 2. Check of `BORROWED_IDEAS.md` against the papers

| Claim in `BORROWED_IDEAS.md` | Check | Note |
|---|---|---|
| DeepScientist: three-tier Findings Memory; LLM reviewer gives ⟨utility, quality, exploration⟩ 0-100; UCB selects (§3.2, Eq. 2) | ✅ | The paper's own setting is **w_u = w_q = κ = 1**, with retrieval top-K = 15 (App. C). T3's 0.5/0.5/0.3 is our choice. Report it next to the paper's 1/1/1 |
| DeepScientist: >5,000 ideas, ~1,100 run, 21 progress; ~60% of failures were implementation errors (§4.3) | ✅ | Also in §4.3: with 100 ideas picked **at random** per task, the success rate was "effectively zero". With their selection it was 1-3%. That is the direct evidence that the surrogate filter matters |
| DeepScientist: Progress Findings get ablations and new datasets (§3.2 "Analyze & Report") | ✅ | |
| EvoScientist: ideation and experimentation memories, embedding retrieval (§3.3-3.4) | ✅ | Their settings: ideation top-k = 2, experimentation top-k = 1, up to 21 candidate ideas (§4.5) |
| EvoScientist: Elo tournament on novelty / feasibility / relevance / clarity (§3.3, Eq. 3) | ✅ | It keeps the top 3 for direction memory and expands the top 1 into a proposal (Eq. 4-5) |
| EvoScientist: four-stage experiment tree (§3.4) | ✅ | Attempt caps per stage are 20 / 12 / 12 / 18 (§4.5). Stage 2 is **hyperparameter tuning before the proposed method**. See B-15 |
| EvoScientist ablation: −IVE loses feasibility 63.3%, −IDE 50.0%, −all 80.0% novelty / 83.3% feasibility (§5.4) | ✅ | |
| AutoSOTA red lines R1-R7, "six" in the text (§3.9) | ✅ with a correction | The count is inconsistent three ways: "six non-negotiable constraints", seven listed, and "all eight constraints" in the third enforcement layer. We use the seven listed |
| AutoSOTA idea library ≥10 ideas, PARAM / CODE / ALGO, risk, CLEARED / REJECTED (§3.8.1 Phase 2) | ✅ | Ideas are also tagged by **granularity**: micro (one parameter), meso (one function), macro (pipeline stage) |
| AutoSOTA leap rule (3 PARAM in a row) and honeymoon (5 follow-ups) (§3.8.1 Phase 3) | ✅ | |
| AutoSOTA measured baseline (Phase 0) | ✅ | |
| AutoSOTA tree rubric with weighted Pass/Fail leaves (§3.3.1) | ✅ | The paper says its own loop "primarily adopted a results-centric strategy" and used mostly the *Result Match* branch (end of §3.3). Using the rubric on the write-up is our extension, not what they did |
| AutoSOTA retrieval before repair, failure memory (§3.6); monitor (§3.5) | ✅ | |
| `BORROWED_IDEAS.md` §1: red lines "checked during optimisation" | ⚠️ incomplete | AutoSOTA enforces them in **four layers**, and the layering is the point: (1) list paper-specific constraints before ideation, in `code_analysis.md`; (2) audit the idea library; (3) re-check every dynamically created Leap idea; (4) put the rules in the main prompt as "absolute prohibitions, with no exceptions". §3 maps these onto Arbor |

## 3. What Arbor already does (read in the v0.1.4 code)

| Borrowed mechanism | Arbor's native version | Gap for us | What to do |
|---|---|---|---|
| Findings Memory / ideation memory (DeepScientist, EvoScientist) | **In-run:** the hypothesis tree. Each node stores an insight; insights propagate to the root, and pruned nodes become negative constraints (paper §4.2-4.3; `TreeView(format="constraints")`, App. B.3). The ablation (§5.7) shows insight propagation matters *more* than the tree itself: w/o insight 54.5% any-medal, w/o tree 63.6%, full 81.8%. **Across runs:** `RecordFinding` → `findings.jsonl` → `EXPERIENCE.md` at finalize (`src/experience.py`, `src/distill.py`), recalled by `src/recall.py` | (a) Recall only runs in the intake chat (`src/cli/intake/launch_tool.py::_with_experience`). `arbor run "<instruction>" --yes` (our `ops/run_arbor.sh`) never uses it. (b) Recall is scoped to the project folder, and A3 builds a new folder. (c) **Our `EXPERIENCE.md` contains a "[leverage]" note recommending the cross-model cache** (it lifted LLaDA dev 0.745 → 0.765). (d) There are no tiers | T3 writes a **curated** experience file: the banned leverage note moves to pitfalls, and every record gets a tier. A3 injects it through the plugin's `meta_ideate_inject`, so it does not depend on intake. Do not reuse the raw iteration-1 `EXPERIENCE.md` |
| Leap rule (AutoSOTA) | `src/coordinator/convergence.py`: after N consecutive experiments that do not beat the trunk, `warn` (3), then `paradigm_shift` (5), then `stop` (8); "parent exhaustion" after 3 failed children. Active only after 4 completed experiments | It is score-based, not type-based, and with 4 scored nodes in iteration 1 it could not fire | Plugin `convergence:` block: `min_experiments: 2`, `warn_after: 2`, `force_after: 3`, `parent_exhaustion_count: 2`. Keep the type-based leap rule (3 PARAM in a row) as a reviewer rule |
| R2 evaluation integrity (AutoSOTA) | `protected_paths`: SHA-256 manifest and read-only bits per executor worktree. A tampered dev score is **discarded** and the branch made unmergeable (`docs/plugins.md`; `src/coordinator/tools/protected.py`, `executor_run.py`) | **Inactive in iteration 1.** `research_config.yaml` set no `protected_paths`, there was no plugin, and the tree meta has none (`idea_tree.json`). Protection was the README text plus `eval.py`'s static check | Add `protected_paths: [task.py, eval.py, eval.sh, baselines.py, "data/**"]` to the plugin (or to `research_config.yaml`) |
| Red-line layers 1-4 (AutoSOTA §3.9) | Plugin prompt injections (`src/plugins/base.py`): `meta_preamble_inject`, `meta_init_inject`, `meta_ideate_inject`, `meta_decide_inject`, `sub_preamble_inject`, `sub_workflow_inject` | Unused | Layer 4 → both preambles. Layer 1 → `meta_init_inject` ("list R1-R8 for this task before ideating"). Layers 2-3 → `meta_ideate_inject` (audit every proposed child against R1-R8; mark CLEARED / REJECTED in the hypothesis text). Promotion rule and ablation → `meta_decide_inject`. Per-model and compute reporting → `sub_workflow_inject` |
| Measured baseline (AutoSOTA Phase 0) | Arbor measures the seed on dev and test before search (`baseline_cache.json`: 0.148 / 0.157 in iteration 1) | Measures whatever the seed is | Make the seed majority vote (A3 already says so) |
| Independent re-execution (DeepScientist App. C: ~50% of first "completions" were false positives from agent timeouts) | `GitMergeBranch` re-runs `eval_cmd_test` in a fresh detached worktree (App. B.4.3) | None for the gate. Our final split is a third, independent run | Keep `eval_final.sh` as the only number we report |
| Merge threshold | `merge_threshold` appears only in the coordinator prompt ("B_dev exceeds trunk_score by >= ~X%", `src/coordinator/prompts.py`). At the gate, `git_ops.py` merges on **any strict** held-out gain and only logs a warning below the threshold. The default is 5.0, and our config sets 1.0 | `research_config.yaml`'s comment ("needed to merge") and A3's "RELATIVE percent gain" are both wrong. Units are also unclear: Arbor formats scores as %, and our `eval.py` prints 0-1 | Do not rely on the threshold. Enforce "net ≥ 3 questions on the gate split per non-floor model, plus a paired test" in `meta_decide_inject`, and as a reviewer rule at every merge |

### Arbor's own limitations, and which borrowing fills each

Arbor's authors list what their system lacks (App. A). Most of the mix fills exactly those holes,
and that is the argument for the mix in the write-up:

| Arbor App. A says it lacks | Filled by |
|---|---|
| "multi-objective search, explicit constraints, Pareto-style comparison" | AutoSOTA R4 (no trade-offs), plus our R8 (compute, B-20) |
| "cost-aware tree policies, adaptive evaluator allocation" | DeepScientist surrogate + UCB for the GPU queue (T3) |
| "explicit reuse of negative evidence", "revisiting suspended branches" | EvoScientist idea-validation memory (B-16); DeepScientist keeps failed findings and has an exploration term |
| "domain knowledge bases"; agents "fail to identify a genuinely new mechanism" | AutoSOTA's research report (B-13); DeepScientist's limitation-first framing (B-12) |
| §6.3: "identifying that move still depended on prior judgment rather than anything the tree could automatically generate" | AutoSOTA's leap path, with its candidates drawn from the research report and the failure taxonomy |

## 4. Mechanisms `BORROWED_IDEAS.md` does not have

Numbered after its 11 mechanisms. Each row says who does it (the prompt is amended).

| # | From (section) | Mechanism | What it changes for us | Owner |
|---|---|---|---|---|
| B-12 | DeepScientist §2 (last paragraph), §3.2 "Strategize & Hypothesize", §4.3 scaling setup | **Limitation-first search.** Start from an identified limitation of the SOTA method, found by failure attribution. In the scaling study each parallel path got one distinct limitation and shared findings every 5 cycles | Our "SOTA" is majority vote. N4 already classifies where it fails (close-second plurality, arithmetic slip, extraction error, shared wrong step). Make that taxonomy the input to A3: one depth-1 Arbor direction per failure class, with its count. Arbor's dev runs only need the class labels, not gold | N4 → A3 |
| B-13 | AutoSOTA §3.7 (AgentIdeator), §3.5.2 (`research_report.md`) | **External research report.** Before ideation, a research-capable model writes a report of SOTA techniques for the task; it seeds structural ideas | N3's verified SOTA table becomes `codification/memory/RESEARCH_REPORT.md`, injected at ideate. It is the source of Leap ideas, so they are not made up mid-run | N3 → A3 |
| B-14 | AutoSOTA §3.8.1 Phase 0; case studies §4.3.1 (FR-Spec: a reused `tree_draft_ids` bug gave the largest gain) and §4.3.3 (a string-format mismatch limited a data join to 63%) | **Debug the baseline pipeline before optimizing.** In their own cases the biggest wins were bugs in the reproduced baseline, not new ideas | Our first-sample accuracy is far below the published numbers (Dream 41.0 vs 77.2, LLaDA 48.7 vs 70.3; `ITERATION_1.md` §Published numbers). Run A2's fidelity check **first**, and audit `task.extract_answer` on the items where the oracle is right and majority is wrong. A pipeline bug would change every number downstream, so this is the highest-value check we have | A2, N4 |
| B-15 | EvoScientist §3.4 (stage 2: hyperparameter tuning before the proposed method) | **Tune the baseline fairly before testing the method** | The bar is the best of a small, pre-registered set of majority-vote variants: tie-break by earliest index, by fewest characters, or by most explicit calculations; `None` answers counted or ignored. Pick the best on dev, freeze it, then start A3. Without this, a "gain" can just be a better tie-break | N4 → A3 |
| B-16 | EvoScientist App. D, Fig. 11 (idea-validation prompt) | **Failure memory schema:** `FAILED (NoExecutableWithinBudget)`, `FAILED (WorseThanBaseline)` or `NOT_FAILED`, plus trigger conditions, do-not-repeat notes and retrieval tags | Use these as extra fields in `FINDINGS.jsonl`, so a "do not retry" is machine-readable | T3 |
| B-17 | EvoScientist App. C.3 | **Validate the LLM judge against humans.** They report 90.0% agreement (108/120 pairs, 3 annotators) before trusting the LLM judge | T2's Elo is otherwise one more unvalidated LLM score. Have 2-3 teammates judge 10 of the 28 pairs blind, and report agreement per criterion | T2 |
| B-18 | Arbor §5.6 (frozen-harness transfer to HLE and DeepSearchQA) | **Frozen transfer test.** Freeze the merged artifact and evaluate it on unseen tasks with no re-tuning | This makes DeepScientist's "second dataset" promotion step concrete. Run the frozen selector on a second math set (e.g. SVAMP) with candidates generated the same way. No re-tuning, no Arbor | A3 |
| B-19 | Arbor §6.1 (BrowseComp) | **Know the ceiling and the normalization trap.** Verifiers "rarely recover candidates that were never surfaced", and part of a verifier gain came from answer normalization | (a) Selection cannot beat oracle@10, so only more steps or samples (A1) raise the ceiling. (b) Check any merged gain for extraction or normalization effects. (c) The BrowseComp evidence-dossier idea has an analogue here: an answer backed by other candidates' verified calculations. Arbor iteration 1 already checked it: it was **anti-predictive** (dev 4 vs 11 on Dream, 8 vs 15 on LLaDA; `EXPERIENCE.md`). It goes into the idea library as rejected by evidence, not retried | T1, A3 |
| B-20 | AutoSOTA R4 (§3.9) and its own §4.3.6 (MoSES, where inference time grew from 3.7 s to 103 s) | **Compute is a metric (our R8).** A gain bought with more inference compute must say so | Every selector reports the extra forward passes per question. Comparisons against "more steps" or "more samples" use matched compute, which is also idea 1.1's own A_rerank and compute-accounting spec | T1, A1 |
| B-21 | AutoSOTA §3.9, four-layer enforcement | **Turn a human-only red line into an automatic one** | Change `eval.py` before A3 starts: call `select()` on every item in two orders (file order, then reversed) and score 0.0 if any choice differs. That catches the module-level cross-model cache Arbor built in iteration 1. `eval.py` is protected, so the team agrees this first and it never changes mid-run | T1 drafts, A3 applies |
| B-22 | DeepScientist §4.1 ("the exploration assessment within our process avoids such combinations of existing knowledge") | **Define the exploration score to reward testing the core claim, not engineering add-ons** | In T3's GPU queue, a high v_e goes to runs that test idea 1.1's own claim: the step arm (A1), per-question preservation (`audit/task4` 04 §1). A new tuning knob scores low | T3 |
| B-23 | DeepScientist App. A.2 | **What reviewers attacked:** missing baselines, too few ablations, no comparison with closely related work | These become three leaves of T5's rubric | T5 |

## 5. Red lines: two additions to the draft in `BORROWED_IDEAS.md` §4

- **R7, tightened.** `select(item)` must be a pure function of that one item. No state across
  calls: no module-level caches, counters or memo tables that persist between items. It must also
  be order-invariant. Enforced by B-21.
- **R8, compute (new).** Report the added inference compute per question (scorer or verifier
  forward passes, extra samples or steps). A gain that needs more compute is compared at matched
  compute against more samples or more steps. It is never presented as free.

## 6. What actually makes idea 1.1 stronger

Process mechanisms make iteration 2 more valid and cheaper. They do not by themselves add
accuracy. Evidence: Arbor's own audit at the end of iteration 1 (`EXPERIENCE.md`, last pitfall)
found every remaining in-item signal on the current features dead or anti-predictive. That covers
cross-candidate verified-result consistency, question-number derivability, and extraction
commitment, and it put the ceiling for explainable rules at about 0.296-0.298 macro. Majority vote
alone scores 0.294 macro on dev, against an oracle of 0.397 (`results/iter1/numbers/baselines_dev.json`).
So the borrowed machinery should be aimed at the four levers that can move the number, in this
order:

1. **Fix the pipeline (B-14).** Explain the ~30-point gap to the published numbers first. If
   extraction or prompting is at fault, every result changes.
2. **Add information each candidate does not have yet.** The DLM's own likelihood (A2) and a
   verifier or process reward model chosen from N3's verified list. Both score one candidate at a
   time, so they respect R7. If there are several, fuse them like AutoSOTA §4.3.3 did (combining an
   ML predictor with a physics-based one): fit the weights on dev, check on the gate, and freeze
   them before the final split.
3. **Raise the ceiling (B-19, B-20).** Selection can only reach oracle@10. Only more steps or
   samples raise it. That is A1, and it is also what idea 1.1's dependent variable (A_rerank)
   compares.
4. **Restate the claim so it can be tested.** Not "AR-likelihood reranking works" (refuted in
   iteration 1). Instead: "how much a DLM preserves its AR base predicts *which* signal helps, AR
   scorers or DLM-native ones". Test it per question within each model (`audit/task4` 04 §1, fix
   2), not across 4 models.

The borrowed mechanisms then have clear jobs:
- **DeepScientist's surrogate + UCB** ranks the GPU jobs behind levers 1-3.
- **DeepScientist's limitation-first framing and AutoSOTA's research report** set Arbor's opening
  directions.
- **The red lines and the plugin** keep the search valid.
- **EvoScientist's memories** stop repeats across runs and across people.
- **The promotion rule and the frozen transfer test** decide whether anything counts.

## 7. The combined loop, revised

```
 BEFORE A3                                  DURING (Arbor + our plugin)                     AFTER
 ---------                                  ---------------------------                     -----
 fidelity check + extraction audit (B-14)   preamble injections: R1-R8 as absolute rules   final split once (eval_final.sh)
 majority-vote failure taxonomy (B-12)      init: list task constraints (AutoSOTA layer 1)  paired test vs tuned majority
 tuned-majority bar, frozen (B-15)          ideate: curated memory + research report +      ablation of the merged rule
 RESEARCH_REPORT.md from N3 (B-13)            failure classes; audit each child (layer 2/3) frozen transfer to 2nd dataset (B-18)
 curated EXPERIENCE / FINDINGS (T3, B-16)   decide: net >=3 questions + paired test         FINDINGS tier update (DeepScientist)
 GPU queue by surrogate + UCB (T3, B-22)    protected_paths hash guard (R2)                 rubric check of write-up (T5, B-23)
 eval.py order-invariance (B-21)            convergence block = native leap rule
```

## 8. For the meeting

1. Agree the `eval.py` order-invariance change (B-21) before anyone starts A3. It is a change to a
   protected file.
2. Agree that A2's fidelity check runs before A3 (B-14), even though it delays Arbor.
3. Agree which second dataset to use for the frozen transfer test (B-18). Generating candidates on
   it costs GPU hours; T3's queue should rank it.
4. The plugin file (`codification/rerank_task/plugins/rerank_guard.yaml`) is new work, added to T1
   (Aalhad). A3 (Aryan) uses it.

## 9. What I did not verify

- I did not run Arbor. Every Arbor behaviour above comes from reading the v0.1.4 source and docs.
  Before A3, confirm on Blackwell: `~/venvs/arbor/bin/pip show arbor-agent` should report 0.1.4, and
  `arbor config show` should list the plugin once it is added.
- I read the paper text, not the papers' code. DeepScientist's and EvoScientist's public repos do
  not contain these mechanisms, and AutoSOTA's system is not released (`REFERENCE_SYSTEMS.md`).
- The B-21 check catches cross-item state that changes choices under reversal. A pathological
  solution could still leak state that the reversal does not reveal. The human review stays.
