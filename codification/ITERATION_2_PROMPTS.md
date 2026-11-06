# Task 4: prompts for the whole team (written 2026-11-06)

These are prompts to paste into a coding agent (Claude Code, opencode or Codex) started in the repo root
on branch `assignment/codification`. Every member starts with **T0** (it loads the context), then runs
their own prompts. Each prompt is self-contained. The review of iteration 1 that produced them is at
the end of this file. The design that the T-prompts build is `codification/BORROWED_IDEAS.md`: ideas
borrowed from DeepScientist, EvoScientist and AutoSOTA, re-implemented around Arbor. A second,
independent review (`codification/BORROWED_IDEAS_REVIEW.md`, 2026-11-06) read the papers and Arbor's
source again. It added mechanisms B-12 to B-23 and found two safety problems and two misread Arbor
settings. Prompts it changed carry a paragraph marked **Review 2026-11-06**.

## Who runs what (suggested; agree it at the meeting)

| Member | Workstation? | Prompts | Needs |
|---|---|---|---|
| Neel | no | N1 (today), N2, N3, N4 | an LLM endpoint for N2 |
| Aryan | yes | A1, A2, then A3 | Blackwell, both GPUs for hours |
| Arjun | yes (and ada) | A4, T4 | Blackwell or ada; laptop for downloads |
| Aalhad | no | T1 | CPU only |
| Khushi | no | T2 | an LLM endpoint (about 60 calls) |
| Diya | no | T3 | an LLM endpoint (about 20 calls) |
| Aaditya | yes | T5, review of every merge, A3 together with Aryan | — |

Order and dependencies: N1 first (submission). T1, T3 and T4 must land before A3 (Arbor iteration 2
reads their files). A1 and A2 run in parallel, one per GPU. N4 waits for the data A2 shares. T2 and T5
are independent.

Since the review of 2026-11-06, A3 also waits for:
- A2's fidelity check, which runs first in A2;
- N4's failure classes and tuned-majority bar;
- N3's RESEARCH_REPORT.md (A3 may start without it).

The team must also agree the eval.py order-invariance change (T1) before A3.

Branches: each person works on `iter2/<name>`, commits small, and pushes. Merge into
`assignment/codification` after one other person has read the diff. GPU users must follow the
Blackwell rules in `codification/HANDOFF.md` §3.

---

## T0 (everyone, first): load the context

```text
You are helping a member of BITS F471 Group 3 with Task 4 (codification) of the course project.
Read these files in order, fully: AGENTS.md; codification/HANDOFF.md; codification/ITERATION_1.md;
codification/NEXT_ITERATION.md; codification/BORROWED_IDEAS.md; codification/BORROWED_IDEAS_REVIEW.md;
codification/ITERATION_2_PROMPTS.md; codification/REFERENCE_SYSTEMS.md; and the timeline page
project-docs/timeline.html (read it as text). Also read docs/verification-standards.md.

Then tell me, in under 25 lines:
1. what Task 4 asks (the brief is quoted in ITERATION_1.md "What the brief asks for");
2. what iteration 1 did and found (the three findings in ITERATION_1.md §Results);
3. what the four reference papers are, why we built on Arbor, and what we borrow from the other three
   (BORROWED_IDEAS.md and the second review's verdict, BORROWED_IDEAS_REVIEW.md §1);
4. which prompt in ITERATION_2_PROMPTS.md is mine (I will tell you my name), what it needs, and what
   it depends on.

Rules for everything you do after this:
- Every claim and number must come from a file in the repo, a command you ran, or a paper you read
  at the full text. Say "not verified" otherwise.
- Never commit data files (rerank_task/data/*/items.jsonl), model weights, secrets or API keys.
- Work on branch iter2/<my name>; never force-push; ask me before pushing.
- If the task needs the workstation (Blackwell), follow HANDOFF.md §3 exactly: never kill a process
  you did not start, do not touch ~/diffullama, no idle GPU holds, no TLS bypass.
- Record the exact model name, date and settings for every LLM call you make for the project.
```

---

## N1 (Neel, today): make iteration 1 submission-ready

```text
Read AGENTS.md, codification/HANDOFF.md, codification/ITERATION_1.md, codification/NEXT_ITERATION.md,
codification/IDEA_SELECTION.md, and idea 1.1 in ideation/output/research_problems_and_ideas.md
(section "Idea 1.1 — P1-M1", about lines 40-287) plus problem P1 (lines 13-24).
Create branch iter2/neel from assignment/codification.

Goal: ITERATION_1.md must be ready to submit today and must answer every clause of the Task 4 brief.
Do not change any number. Every number must already exist in the repo (ITERATION_1.md or
codification/results/iter1/numbers/). If a number you need is not there, write "not measured".

1. Fix the stale parts:
   - Issue I-11 still says "fill in from the smoke runs" and "LLaDA and DiffuLLaMA pending". Replace it
     with the final first/oracle numbers per model from results/iter1/numbers/baselines_dev.json.
   - The section "Run plan for Thursday" is written as a future plan with paper estimates. Rename it to
     "Run plan (as planned on 2026-11-02)" and add one short paragraph "What actually ran" taken from
     NEXT_ITERATION.md "Process lessons" and the commit log (git log --format='%ad %s' --date=short).
   - In the Arbor table, fill the seed row (dev 0.148 and gate 0.157, from
     results/iter1/arbor_session/.coordinator/baseline_cache.json "baseline_score" and
     "test_baseline_score"), mark nodes 2.1 and 3.1 as "not merged (pruned)" with their dev
     scores, and rewrite the intro paragraph from instructions ("Paste what you told it") into a
     description of what was done.
2. Add a section "Specification vs implementation". One row per element of idea 1.1's method text:
   models (6 named incl. DiffuCoder-7B), tasks (HumanEval, GSM8K, SIQA, WinoGrande), step budgets
   (16/32/64/128), N = 10 via "random seed at the initial noise stage", GPT-2 mismatched scorer,
   TinyLlama matched scorer, BERT non-AR control, LLaMA-2-7B scorer capacity check, CKA corpus (5,000
   stratified segments), CKA floor (random init) and ceiling (identity) normalisation, unified
   normalised metric Q, compute accounting (FLOPs, T_gen/T_bridge/T_score), Reranking Advantage
   A_rerank = dQ_rerank - dQ_steps(16->32), Pearson + Spearman + bootstrap + leave-one-out,
   stratification by family, 8-bit quantisation + activation checkpointing.
   Columns: Spec says | We did | Why (issue ID or reason) | Effect on the conclusion.
   Be exact: e.g. we ran GSM8K only, one step budget (64), 5 models, 2,000 WikiText segments for CKA,
   no BERT control, no compute accounting, no random-init floor. Say plainly that the spec's
   dependent variable (A_rerank, which needs the step arm) was NOT measured, so iteration 1 tested
   "CKA predicts reranking headroom", a reduced form of the hypothesis.
3. Add a column "Feedback given to the model" to the issue log. For each issue, state who/what found
   it and how the fix reached the model: Arbor review reply (quote the row in §Arbor), a coding-agent
   session (the commit hash that applied it), or a direct human edit. Do not invent conversations;
   if the repo has no record, write "human edit, commit <hash>".
4. Add a short "Compliance with the brief" table at the top: clause | where answered | status
   (done / partial / not done) | gap. Be honest: SOTA comparison is partial (no published method
   reproduced), and part (b) of research problem P1 (rerank vs more steps) is not answered.
5. Export ITERATION_1.md to PDF with pandoc (see ideation/REPORT.md "PDF note" for the method used
   before). Check that every table renders.

Verify: git diff shows no changed number; every new number appears in a file under
codification/results/iter1/numbers/ or ITERATION_1.md already. Commit as
"docs(codification): make iteration 1 submission-ready" and push iter2/neel. Report the diff summary
and the PDF path.
```

---

## N2 (Neel): refine idea 1.1 by feedback to the ideation model

The brief asks to "refine the idea by fixing it by providing proper feedback to the model". Iteration
1 gave feedback to Arbor (the codifying model), and humans fixed the spec in code. The idea text
itself was never revised by a model. This prompt closes that gap.

```text
Read ideation/README.md, ideation/REPORT.md, ideation/scripts/templates.py and
ideation/scripts/research_agent.py, then codification/ITERATION_1.md (issue log and §Results) and
idea 1.1 in ideation/output/research_problems_and_ideas.md.

Goal: produce "idea 1.1 v2", a revised specification written by an LLM from our feedback, with the
prompt and the answer recorded verbatim. This is evidence for the brief's "refine the idea by
providing feedback to the model".

1. Use the refinement prompt in templates.py: REFINE_BLOCK (note D4 there: the paper publishes no
   refinement template, so this block is ours), appended to the method-generation prompt the way
   research_agent.py does it. Write a one-off script codification/refine/refine_idea.py that fills
   that template with: the original idea 1.1 text; as "reviews", our feedback built from the issue log
   (I-1 to I-20) and the iteration-1 results (AR reranking loses to majority vote; Arbor 4.1 ties
   majority; CKA n = 4, three floor models; step arm not run). Keep the feedback factual and short:
   one bullet per issue, each with the evidence number.
2. Call one model through the existing pluggable backend in ideation/scripts/llm.py (any
   OpenAI-compatible endpoint you have; record the exact model name, date and temperature). Use a
   different model family from the one that will judge it, if you also run a review.
3. Save request and response verbatim to codification/refine/idea_1_1_v2_request.md and
   idea_1_1_v2_response.md. Do not edit the response.
4. Then review the response yourself against primary facts (model lineages in ideation/REPORT.md §7,
   data availability, inference-only constraint, one shared GPU). Write codification/refine/REVIEW.md:
   what v2 fixed, what it still gets wrong, and a second round of feedback if needed (max 2 rounds).
5. The v2 spec must at least: make majority vote the baseline to beat; keep the step-budget arm
   (it is the spec's dependent variable); replace the across-model correlation as the main test with
   something that has power (e.g. per-question preservation vs per-question rerank success, or more
   models with an AR base); add the DLM's own likelihood as a scorer; name a published comparison.
   If the model does not do this, say so in REVIEW.md and give that feedback in round 2.

Verify: the script runs end to end and the response file is the raw model output. Commit as
"feat(codification): refine idea 1.1 via feedback to the ideation model" on iter2/neel and push.
Report the model used, the number of rounds, and a 5-line summary of what changed.
```

---

## N3 (Neel): state of the art and novelty, at the primary source

```text
Read docs/verification-standards.md first and follow it. Then read
`git show origin/audit/task4-codification:audit/task4/05-novelty-and-sota.md`. Every entry there is
abstract-level only.

Goal: codification/SOTA.md, a verified table of the published methods we must compare against, and
the novelty status of idea 1.1.

1. For each paper in that file (2609.35472, 2609.39560, 2604.06260, 2602.01842, 2602.22871,
   2510.27469) read the full text (arXiv HTML or PDF, not a summary) and record: method in one line;
   checkpoint (Base or Instruct, exact HF id); task and split; shots; N candidates; denoising steps;
   the GSM8K number(s) with table/section; whether it is post-hoc selection (works on fixed candidates)
   or changes generation. Quote the sentence the number comes from.
2. Run a fresh search (arXiv listing/API, Semantic Scholar, the anchor authors' 2025-26 papers) for:
   test-time scaling / best-of-N / self-consistency / reranking / verifiers for diffusion language
   models; and any work linking AR-representation similarity (CKA, probing) of adapted DLMs to
   inference behaviour. Add each relevant paper with the same fields.
3. Mark comparability with our setup (Dream-v0-Base-7B and LLaDA-8B-Base, N = 10, 64 steps, 4-shot,
   GSM8K test, temperature 0.7). Most published numbers use Instruct checkpoints: say so.
4. Pick the best 1-2 methods that are post-hoc selectors (they only read the 10 candidates, maybe
   plus a public reward model) and write an exact spec for Aryan: model id, input format, scoring
   rule, tie-break. That becomes a selector in rerank_task/baselines.py later.
5. Novelty: state whether anyone has already linked AR-representation preservation to selection or
   reranking gain. If yes, idea 1.1's claim is preempted; write that clearly.

6. Review 2026-11-06 (AutoSOTA §3.7 / §3.5.2 "research_report.md"; BORROWED_IDEAS_REVIEW.md B-13):
   also write codification/memory/RESEARCH_REPORT.md, at most 60 lines, for Arbor to read at ideation.
   For each verified method, give one line on the mechanism and the per-candidate signal it uses. Mark
   whether it is admissible under our red lines: it must score one candidate of one item at a time,
   with no other model's answers and no gold. Arbor's Leap ideas come from this list.

Commit as "docs(codification): verified SOTA and novelty for idea 1.1" on iter2/neel and push.
Report the table and the selector spec. Do not write any number you did not read in the paper.
```

---

## N4 (Neel, after Aryan shares the items from A2): CPU analysis

```text
Aryan will share rerank_task/data/{dev,test}/items.jsonl (dev 200, gate 200) and the final-split items
separately. Put them in codification/rerank_task/data/ (git-ignored; never commit them).
Rule: explore on dev only. Use the gate split to check. Touch the final split once, at the end,
for reporting. Write down any rule before you look at the final split.

1. Answer diversity per model: distinct answers per question and the entropy of the answer
   distribution (dev and test). Report it next to headroom; it is a confound for the CKA claim.
2. Where does majority vote fail? On dev, for dream-base and llada-base, take the questions where
   majority is wrong and oracle is right. Classify the failures (wrong plurality with a close second,
   arithmetic slip, extraction error, all candidates share a wrong step). Count each class.
3. Leave-one-out on the CKA-vs-headroom correlation (the spec asks for it): recompute Spearman with
   each model dropped, using codification/pipeline/analyze.py logic. Report all 4 values.
4. If N3 gives a post-hoc selector that needs no GPU, implement it as a selector in a COPY of
   baselines.py (do not edit the protected file) and run it with codification/paired_stats.py on
   dev and gate.

5. Review 2026-11-06 (BORROWED_IDEAS_REVIEW.md B-12, B-14, B-15). Three additions:
   - Failure taxonomy (step 2) is the input to Arbor iteration 2, following DeepScientist's
     "start from a limitation of the SOTA method". End ANALYSIS_ITER2.md with a short block
     "Failure classes for A3": one line per class, with its dev count per model and a one-line
     example. Use counts and labels only, never a gold answer.
   - Extraction audit, following AutoSOTA's lesson that its largest gains were bugs in the baseline
     pipeline. On the dev items where the oracle is right and majority is wrong, read 20 candidates
     by hand. Check that task.extract_answer took the number the candidate meant. Report every
     mismatch.
   - Tuned-majority bar, following EvoScientist's stage 2: tune the baseline before testing the
     method. Fix this set before looking: tie-break by earliest index, by shortest text, or by most
     explicit calculations; with None answers counted or ignored. Run each on dev. Freeze the best as
     "tuned majority" and report it with plain majority on dev and gate. A3 must beat tuned majority.

Write codification/ANALYSIS_ITER2.md with the commands you ran and their real output. Commit on
iter2/neel and push.
```

---

## A1 (Aryan, GPU): the step-budget arm, which the spec's main variable needs

```text
Read AGENTS.md and codification/HANDOFF.md fully before anything else, especially §3 (Blackwell
rules). Hard rules: never kill or signal a process you did not start; do not touch ~/diffullama; do
not keep a GPU server or model loaded while you wait for a person; never disable TLS checks; run with
HF_HUB_OFFLINE=1. Check nvidia-smi and ask in the team chat before you take a GPU. The `antpc`
account is shared with other projects.

Context: idea 1.1's dependent variable is the Reranking Advantage
A_rerank(M) = dQ_rerank(M) - dQ_steps(M, 16->32) (ideation/output/research_problems_and_ideas.md,
"Statistical Correlation Analysis"). Iteration 1 ran one budget (64 steps), so part (b) of research
problem P1 ("is reranking more efficient than more denoising steps?") is unanswered.

1. On Blackwell, in ~/codify/diffullama (sync it to the latest assignment/codification first, see
   HANDOFF.md §3 "git bundle"), create branch iter2/aryan.
2. Add a --steps override to codification/pipeline/generate_candidates.py that also tags the output
   (e.g. model key "dream-base@k16") so 64-step files are never overwritten. Thread the tag through
   score_candidates.py, assemble.py and baselines.py. Run `python3 codification/pipeline/selftest.py`
   on CPU and paste its output.
3. Record wall-clock per phase (T_gen per question, T_score per candidate) in the job logs, because
   the spec compares gains per unit of compute.
4. Smoke: --limit 5 at K = 16 for dream-base and llada-base. Measure s/question and estimate the full
   run before launching. Then run K = 16 and K = 32 on the test split, first 500 questions (same items
   as iteration 1), N = 10, same temperature (0.7). One model per GPU. Jobs must resume after a crash
   (the display watchdog kills kernels; see I-18).
5. Score the new candidates with gpt2, qwen0.5b and the matched 7B base (qwen7b for Dream; LLaDA has
   no base, so qwen7b as the "mismatched" scorer only). Assemble and run baselines.py.
6. Produce a table per model: first / majority / best AR / oracle at K = 16, 32, 64, and the time per
   question at each K. Compute dQ_steps(16->32), dQ_steps(32->64), dQ_rerank at each K, and A_rerank,
   for the AR selectors and for majority vote. Write it to codification/results/iter2/step_arm.md
   with the exact commands.

Stop the GPU jobs when done. Commit code and the small result files (not candidates) on iter2/aryan
and push (GitHub is unreliable from the box: bundle to a laptop and push from there, as in HANDOFF.md).
Report real numbers only.
```

---

## A2 (Aryan, GPU): DLM self-score, a fidelity reference, and data for Neel

```text
Same rules as A1 (read HANDOFF.md §3). Branch iter2/aryan.

1. Data for Neel: copy rerank_task/data/dev/items.jsonl and data/test/items.jsonl from ~/arbor-rerank
   (dev 200, gate 200) and ~/arbor-rerank-final/items.jsonl (final 300) into a zip on the laptop and
   share it with Neel through the team Drive. Do NOT commit them (they are git-ignored on purpose).
2. DLM self-score (the "dlm.selfscore" gap in iteration 1): for dream-base and llada-base, score each
   existing candidate (no new generation) by the model's own likelihood of the answer given the
   prompt. Use the Monte Carlo masked-likelihood estimate (mask a random fraction t of answer tokens,
   sum log-probs of the masked tokens divided by t, average over k = 8 draws, fixed seed). Check how
   LLaDA's official repo estimates likelihood at the pinned commit in config.json
   ("llada_generate_commit") and follow it if it exists. For Dream, check in its model code whether
   logits are shifted by one position before you read them. Write the feature as
   "dlm.self.nll_sum" / "n_tokens" next to the AR features, so baselines.py picks it up as a scorer.
   Test on --limit 5 first and confirm the score differs across the 10 candidates.
3. Fidelity reference: our first-sample accuracy is far below the published numbers (Dream 41.0 vs
   77.2 at 8-shot; LLaDA 48.7 vs 70.3 at 4-shot; see ITERATION_1.md "Published numbers"). On the first
   100 test questions run ONE greedy/low-temperature sample per question with each model's published
   settings (Dream: 8-shot, its recommended temperature and steps; LLaDA: 4-shot, its default
   sampler and steps). Report accuracy next to our 64-step, T 0.7 first sample. This tells us how much
   of the gap is our sampler and how much is a bug. If the gap stays above ~10 points at published
   settings, stop and report: something in prompt format or answer extraction may be wrong.
4. Run baselines.py and codification/paired_stats.py with dlm.self as a selector vs majority vote
   (dev and gate only; do not look at the final split yet).
5. Review 2026-11-06 (BORROWED_IDEAS_REVIEW.md B-14): do step 3, the fidelity reference, FIRST, before
   steps 2 and 4 and before A3. AutoSOTA's biggest gains in its own case studies came from bugs in the
   reproduced baseline (§4.3.1, §4.3.3). Our first-sample accuracy is about 30 points below the
   published numbers. If a prompt-format or extraction bug explains part of that, every later number
   changes. Report the result to the team before you start A3.

Commit on iter2/aryan, push via the laptop. Report real outputs. Free the GPUs when done.
```

---

## A3 (Aryan, GPU, after A1 + A2): Arbor iteration 2 with the right baseline

```text
Same rules as A1. Read HANDOFF.md §3 "Rerun Arbor" and NEXT_ITERATION.md "Process lessons" first.

Iteration 1's merge gate compared against the seed rule (GPT-2 NLL), not majority vote, so Arbor
could merge a rule that only ties majority. Fix that, and wrap Arbor in the pieces borrowed from
the other reference papers (codification/BORROWED_IDEAS.md). Needs T1, T3 and T4 merged first.
Since the review of 2026-11-06 it also needs:
- N4's failure classes and tuned-majority bar;
- N3's RESEARCH_REPORT.md (if it is not ready, start without it and say so);
- A2's fidelity result.

1. Rebuild the Arbor task repo (codification/make_arbor_repo.sh) into ~/arbor-rerank-iter2 with:
   - seed solution.py = majority vote (copy baselines._majority logic);
   - the new features from A1/A2 in items (dlm.self, the 7B scorers; candidates at K = 16/32 only if
     you also give Arbor a compute budget per question, otherwise leave them out);
   - research_config.yaml: interaction_mode review. The macro score averages all 5 models, and the
     3 floor models dilute it. Decide with the team whether eval.py (a protected file, change it only
     before Arbor starts) should print a score over dream-base and llada-base only, so the gate means
     "beats majority vote where there is headroom". One point on a 200-question split is 2
     questions; ask for at least 3 net questions.
     Correction (Review 2026-11-06, read in Arbor v0.1.4 src/coordinator/tools/git_ops.py):
     merge_threshold is NOT enforced at the merge gate. Arbor merges on ANY strict held-out gain
     and only logs a warning below the threshold. The threshold appears only as a soft hint in the
     coordinator prompt. So "at least 3 net questions" must be enforced by you at the merge gate,
     and by the plugin's meta_decide_inject text (step 2b).
2. Write the instruction from codification/arbor_instruction.txt plus three blocks built from the
   memory files (codification/BORROWED_IDEAS.md §3; they come from prompts T1, T3, T4):
   - "Red lines" = codification/memory/RED_LINES.md, verbatim (AutoSOTA's red-line idea).
   - "What earlier runs learned" = the Idea, Implement and Progress findings in
     codification/memory/FINDINGS.md (DeepScientist's Findings Memory): the four directions
     iteration 1 tried with their dev scores, that AR likelihood is anti-predictive on Dream, that
     cross-model consensus is out of scope, that verification weighting tied majority. Tell it not
     to retry these.
   - "Opening directions" = the CLEARED ideas of codification/memory/IDEA_LIBRARY.md, highest
     priority first, with their PARAM/CODE/ALGO type.
   Give your own agent codification/memory/STRATEGIES.md too, and check it before you debug any
   failure (AutoSOTA's "retrieval before repair").
2b. Review 2026-11-06 (codification/BORROWED_IDEAS_REVIEW.md §3-§4). Put the blocks into Arbor
   natively, not only into the instruction text:
   - Plugin. Copy codification/rerank_task/plugins/rerank_guard.yaml (from T1) into the Arbor repo's
     plugins/ folder and set `plugin: rerank_guard` in research_config.yaml. Then check that
     `arbor config show` lists it. The plugin provides:
     - protected_paths, so Arbor hash-checks task.py, eval.py, eval.sh, baselines.py and data/**.
       This was inactive in iteration 1.
     - The red lines in all six injection points, matching AutoSOTA's four enforcement layers.
     - A convergence block that forces a "paradigm shift" after 3 non-improving experiments (native
       leap rule).
   - Memory. Use codification/memory/EXPERIENCE_CURATED.md (T3), never the raw iteration-1
     EXPERIENCE.md. The raw file lists cross-model consensus as a "[leverage]" finding. Also, our
     `arbor run ... --yes` launch never runs Arbor's experience recall (that happens only in the
     intake chat), so the curated memory must go in through the plugin's meta_ideate_inject or the
     instruction.
   - Opening directions. One depth-1 direction per failure class of majority vote, from N4's
     "Failure classes for A3" (DeepScientist's limitation-first search). Add Leap candidates from
     codification/memory/RESEARCH_REPORT.md (N3).
   - Bar and evaluation. The seed is N4's frozen "tuned majority", not plain majority. Before
     launch, apply the eval.py order-invariance check drafted in T1. Only do this if the team
     agreed, and never change it during the run.
3. Start vLLM only when you are ready to answer review prompts; stop it the moment Arbor ends.
   Answer every gate with /approve, /skip, /edit <note> or /answer <text>. Review rules:
   - Reject (/answer with the reason) any proposal that breaks a red line, and name the line.
   - Leap rule (AutoSOTA): if the last three proposals were all PARAM tweaks (weights, thresholds),
     reply with /edit asking for a structural (CODE or ALGO) idea from the idea library.
   - Ask for an ablation of any rule that reaches the gate (EvoScientist's fourth stage): the rule
     without its new part, on dev.
   Paste each reply verbatim into a §Arbor table in codification/ITERATION_2.md.
4. After Arbor ends, score the merged rule (if any) on the final split once with
   codification/eval_final.sh and codification/paired_stats.py against majority vote.
   Promotion rule (DeepScientist's Progress Finding): call it an improvement only if it beats
   majority on the final split with a paired test, its ablation loses, and it also holds on a second
   dataset (for example SVAMP; agree which with the team first). Otherwise record it in
   FINDINGS.jsonl as an Implement Finding with its numbers.
   Review 2026-11-06: run the second dataset as Arbor's own frozen transfer test (Arbor §5.6). Freeze
   the merged solution.py and evaluate it unchanged on the new candidates, with no re-tuning and no
   Arbor. Report the extra compute per question (red line R8). Also check whether the gain comes
   from answer extraction or normalization rather than from the rule (Arbor §6.1 found part of a
   verifier gain was answer normalization).

Copy the session folder to codification/results/iter2/arbor_session/, commit, push via the laptop.
```

---

## A4 (Arjun, optional, needs downloads): models that make the comparison fair

```text
Same rules as A1. Blackwell cannot reach huggingface.co (TLS interception). Download on a laptop,
check sha256, copy with scp into the HF cache (HANDOFF.md §3 and project memory: stage in
~/.cache/huggingface/hub/.incoming/ first).

Add to codification/pipeline/config.json and run at N = 10, 64 steps, test first 500:
- diffusionfamily/diffullama-gsm (GSM8K LoRA on DiffuLLaMA; gives DiffuLLaMA real headroom and a
  within-family CKA pair);
- Dream-org/Dream-v0-Instruct-7B and GSAI-ML/LLaDA-8B-Instruct (published test-time methods report
  on these, so our majority vote and selectors become comparable to them; check N3's table first).
Then scorers, CKA for the models with an AR base, baselines, analyze.py. Each 7B model costs hours of
GPU time: measure on --limit 5 and agree a window with the team first. Check
codification/memory/GPU_QUEUE.md (prompt T3) for which of these the team ranked highest.
```

---

## T1 (Aalhad, CPU): red lines and a typed, audited idea library (from AutoSOTA)

```text
Run T0 first. Then read codification/BORROWED_IDEAS.md §1 "AutoSOTA" and §4, codification/rerank_task/
(task.py, eval.py, baselines.py, solution.py, README.md), codification/arbor_instruction.txt, and
ITERATION_1.md §Arbor and §Results. Branch iter2/aalhad.

Goal: two files that Arbor iteration 2 (prompt A3) and every human reviewer will use.

1. codification/memory/RED_LINES.md. Finish the draft in BORROWED_IDEAS.md §4. For each red line:
   the rule in one sentence; why (AutoSOTA's R-number it follows, and our own evidence: e.g. Arbor
   proposed cross-model consensus twice in iteration 1); how a reviewer checks it (which file or test
   would show a violation). Then check eval.py's existing leakage and correctness gates: list which
   red lines they already enforce automatically and which only a human review catches.
2. codification/memory/IDEA_LIBRARY.md. At least 12 candidate selector ideas for Arbor iteration 2,
   like AutoSOTA's Phase 2 library. Each row: id, one-line idea, type (PARAM / CODE / ALGO), risk
   (LOW / MEDIUM / HIGH), which features in items.jsonl it needs (check what assemble.py writes;
   mark features that do not exist yet, e.g. dlm.self from prompt A2), expected effect and why, and
   its red-line audit: CLEARED or REJECTED with the line it breaks. Mix low-risk tuning with
   structural ideas. Include ideas from ITERATION_1.md issue log and NEXT_ITERATION.md, and do NOT
   repeat the four directions iteration 1 already ran (2.1, 3.1, 4.1, plain majority) unless you
   say what is different. Include at least two REJECTED examples so the audit is visible.
3. Priority order: rank the CLEARED ideas by feasibility and expected gain, with one line of reason.
4. Review 2026-11-06 (codification/BORROWED_IDEAS_REVIEW.md §3-§5). Four additions:
   a. Red lines. Tighten R7: select(item) is a pure function of one item, with no state across calls
      (no module-level caches or counters) and no dependence on item order. Add R8, compute: report
      the extra inference compute per question, and compare at matched compute.
   b. Idea library.
      - Tag each idea micro / meso / macro as well (AutoSOTA §3.8.1 Phase 2).
      - Add the in-item signals Arbor iteration 1 already found anti-predictive. They are in the
        last pitfall of codification/results/iter1/arbor_session/EXPERIENCE.md: cross-candidate
        verified-result consistency, question-number derivability, extraction-commitment weighting.
        Mark them "REJECTED (evidence)" with the numbers, separate from red-line rejections.
   c. Arbor plugin. Write codification/rerank_task/plugins/rerank_guard.yaml in Arbor's plugin format.
      The format is in docs/plugins.md and src/plugins/base.py of RUC-NLPIR/Arbor at tag v0.1.4. It
      needs:
      - name, description, schema_version: 1;
      - eval_contract: metric_direction maximize, eval_cmd "bash {cwd}/eval.sh dev";
      - protected_paths: task.py, eval.py, eval.sh, baselines.py, "data/**".
      - Red lines in the six injections, following AutoSOTA's four enforcement layers:
        - meta_preamble_inject and sub_preamble_inject: R1-R8 as "absolute prohibitions, with no
          exceptions" (layer 4).
        - meta_init_inject: "list the task-specific constraints before ideating" (layer 1).
        - meta_ideate_inject: "audit every proposed child against R1-R8 and write CLEARED or
          REJECTED (rule) into its hypothesis" (layers 2-3).
        - meta_decide_inject: "merge only with >= 3 net questions on the gate split per non-floor
          model, and ask for an ablation first".
        - sub_workflow_inject: "report per-model accuracy and extra compute".
      - A convergence block with min_experiments: 2, warn_after: 2, force_after: 3,
        parent_exhaustion_count: 2. This makes Arbor's paradigm-shift signal fire early enough to
        act as the leap rule.
      Check it parses: python3 -c "import yaml; print(sorted(yaml.safe_load(open(
      'codification/rerank_task/plugins/rerank_guard.yaml', encoding='utf-8'))))".
   d. eval.py patch, a draft only: write it to codification/memory/eval_order_check.patch and do
      not apply it. Call select() on every item in file order, then again in reversed order. Score
      0.0 with "correctness: FAIL (order-dependent select)" if any choice differs. This catches the
      module-level cross-model cache Arbor built in iteration 1. Test it on a copy:
      results/iter1/solution_merged_4_1.py must pass, and a toy solution with a module-level dict
      keyed on item["question"] must fail. Paste both outputs.

Verify: every feature name you cite appears in codification/pipeline/assemble.py or is marked
"needs A2/A1". Commit "docs(codification): red lines and audited idea library for Arbor iteration 2"
on iter2/aalhad and push. Report the counts (CLEARED / REJECTED) and the top 3.
```

---

## T2 (Khushi, LLM API): re-rank P1's ideas with an Elo tournament (from EvoScientist)

```text
Run T0 first. Then read codification/BORROWED_IDEAS.md §1 "EvoScientist", ideation/REPORT.md §7
(why the ideation scores are weak: 78 of 100 ratings were 5), codification/IDEA_SELECTION.md, and
the 8 P1 ideas in ideation/output/research_problems_and_ideas.md (section "Research problem 1").
Branch iter2/khushi.

Goal: check, with a method that does not need calibrated scores, whether idea 1.1 was the right
choice to codify. This supports the brief's "codify the idea which you think is the best".

1. Write codification/memory/elo_tournament.py (stdlib + the existing ideation/scripts/llm.py
   backend). For every pair of P1's 8 ideas (28 pairs), ask a judge model which idea is better on
   EvoScientist's four criteria (novelty, feasibility, relevance, clarity), one call per pair per
   order, so 56 calls; randomise which idea is shown first and run both orders to cancel position
   bias. Give the judge our real constraints (inference only, one shared 96 GB GPU, weeks left) and
   the lineage facts in ideation/REPORT.md §7, so feasibility is judged against reality.
2. Use a judge from a different model family than the ideation generator (Qwen3.8) and reviewer
   (Gemma). Record model, date and temperature. Cache every response (the run must be resumable;
   the free OpenRouter tier allows only 50 requests a day, see the decision log D-2026-10-08-b).
3. Compute Elo ratings (start 1000, K = 32, fixed order of games, then repeat with 20 random orders
   and report the mean and spread of each rating). Report the ranking per criterion and overall.
4. Write codification/memory/ELO_RANKING.md: method, model, the table, and a short verdict: does
   1.1 stay top? If another idea wins clearly, say what that would mean for iteration 2, but do not
   change the project direction; the team decides.
5. Review 2026-11-06 (EvoScientist App. C.3; BORROWED_IDEAS_REVIEW.md B-17): validate the judge before
   trusting it. EvoScientist reports 90.0% LLM-human agreement (108/120 pairs) before relying on its
   judge. Pick 10 of the 28 pairs at random (fixed seed). Ask 2-3 teammates to judge them blind on
   the same four criteria, then report agreement with the LLM judge per criterion. If agreement on
   a criterion is under ~70%, say that this criterion's Elo is not reliable.

Commit the script, the cached responses (no API keys), and ELO_RANKING.md on iter2/khushi; push.
```

---

## T3 (Diya, CPU + LLM API): Findings Memory and a surrogate-ranked GPU queue (from DeepScientist)

```text
Run T0 first. Then read codification/BORROWED_IDEAS.md §1 "DeepScientist", ITERATION_1.md (issue log,
§Arbor, §Results), NEXT_ITERATION.md, and codification/results/iter1/arbor_session/ (REPORT.md,
findings.jsonl, EXPERIENCE.md, experiments/*/report.md). Branch iter2/diya.

Goal: a persistent Findings Memory that Arbor iteration 2 and every teammate's agent reads, and a
ranked list of the GPU experiments we could run next.

1. codification/memory/FINDINGS.jsonl: one JSON record per finding, fields: id, date, source (Arbor
   node id, issue id, or commit), hypothesis, tier ("idea" / "implement" / "progress", as in
   DeepScientist §3.2), result (numbers with split and n), verdict, evidence (file path). Seed it
   from iteration 1: every Arbor node (incl. the rejected cross-model idea and the skipped node-1
   child), every selector in results/iter1/numbers/baselines_*.json, and issues I-16 to I-20.
   Use only numbers from those files. Also write FINDINGS.md, a readable view grouped by tier.
2. codification/memory/GPU_QUEUE.md: list every GPU experiment proposed so far (A1 step arm, A2
   self-score and fidelity, A4 models, NEXT_ITERATION.md ideas 2-7, the per-question preservation
   test in audit/task4 04-science-audit.md). Ask one surrogate LLM reviewer, with FINDINGS.md and our
   constraints in its context, to score each with DeepScientist's vector <utility, quality,
   exploration> (integers 0-100) and a short reason. Rank by UCB = 0.5*u + 0.5*q + 0.3*e (our chosen
   weights; record them). Add your estimate of GPU-hours from the timings in ITERATION_1.md and the
   prompts. Record model, date, temperature.
3. Write a 5-line rule in FINDINGS.md for how a finding moves up a tier, using BORROWED_IDEAS.md §3
   (promotion needs the final split, an ablation and a second dataset).
4. Review 2026-11-06 (codification/BORROWED_IDEAS_REVIEW.md §3, B-16, B-22):
   a. Fields. Add EvoScientist's idea-validation fields to every FINDINGS.jsonl record (App. D,
      Fig. 11):
      - outcome: FAILED (NoExecutableWithinBudget), FAILED (WorseThanBaseline) or NOT_FAILED;
      - trigger_conditions;
      - do_not_repeat;
      - retrieval_tags.
   b. Curated Arbor memory. Write codification/memory/EXPERIENCE_CURATED.md in the same format as
      results/iter1/arbor_session/EXPERIENCE.md (front matter, then "- **[kind] about** — note").
      Copy the raw file's findings, with one change: the "[leverage] ... cross-model consensus"
      item must become a "[pitfall]" that says it breaks red line R7 and was rejected twice. That
      raw item recommends the banned idea, and Arbor would read it as a prior. Add the tier of each
      finding. A3 injects this file into Arbor.
   c. UCB. The paper's own weights are w_u = w_q = κ = 1 (DeepScientist App. C). Rank by those, and
      by your 0.5/0.5/0.3 as a sensitivity check. Report both orders.
   d. Exploration score. Tell the surrogate that v_e should be high for runs that test idea 1.1's
      own claim (the A1 step arm; the per-question preservation test) and low for engineering
      add-ons. DeepScientist's exploration assessment avoids "combinations of existing knowledge"
      (§4.1).

Verify: every number in FINDINGS.jsonl can be found in the cited file (write a tiny check script
that loads the JSON and prints each evidence path; open a sample of 5 by hand). Commit on
iter2/diya and push.
```

---

## T4 (Arjun): experimentation memory, from what broke and how it was fixed (from EvoScientist and AutoSOTA)

```text
Run T0 first. Then read codification/BORROWED_IDEAS.md (EvoScientist experimentation memory; AutoSOTA
"retrieval before repair"), codification/HANDOFF.md §3, NEXT_ITERATION.md "Process lessons",
ITERATION_1.md issues I-7 to I-20, codification/results/iter1/ops/, ideation/SERVER_RUNBOOK.md and
ideation/WORKSTATION_RUN.md, and `git log --format='%h %ad %s' --date=short` for fix(...) commits.
Branch iter2/arjun.

Goal: codification/memory/STRATEGIES.md, a short, searchable list of known failures and their fixes,
so any agent checks it before debugging on the workstation.

1. One entry per failure: symptom (the exact error string where the repo has it), cause, fix, the
   command or flag, and evidence (commit hash or file). Cover at least: vLLM context too small for
   Arbor (131072), tensor-parallel hang (use data-parallel), Qwen3 XML tool calls, Arbor plain-text
   replies going to the helper LLM, HF unreachable from Blackwell (offline cache, laptop download,
   sha256), datasets.load_dataset ignoring the hub cache offline, display watchdog killing kernels
   (resume up to 8 tries), DiffuGPT-S 512 context, Dream temperature 0.2 giving identical
   candidates, causal attention in DiffuGPT CKA, scorer names with dots, GitHub/LFS push timeouts
   (bundle sync), Windows selftest needing Git Bash, idle GPU hold.
2. Group entries by area (vLLM/Arbor, models and sampling, data and cache, GPU and machine, git).
3. Add any failures you met yourself in the ideation runs (OpenRouter limits, retired model slugs)
   from ideation/REPORT.md, with evidence.

Commit on iter2/arjun and push. Then do A4 if the team agrees on a GPU window.
```

---

## T5 (Aaditya or anyone, CPU + optional LLM): a Pass/Fail rubric for the write-up (from AutoSOTA)

```text
Run T0 first. Then read codification/BORROWED_IDEAS.md (AutoSOTA AgentObjective), the Task 4 brief
as quoted at the top of ITERATION_1.md, idea 1.1's spec (ideation/output/research_problems_and_ideas.md,
"Idea 1.1"), and the review at the end of codification/ITERATION_2_PROMPTS.md. Branch iter2/<name>.

Goal: codification/memory/RUBRIC.md, a weighted tree of Pass/Fail checks that any agent can apply
to ITERATION_1.md or ITERATION_2.md before submission.

1. Root = "Task 4 iteration is complete and honest" with weight 100. Children = the brief's clauses
   (choose best idea; codify with LLMs; spec gaps before and after trials; refinement by feedback to
   the model; final results on metric + dataset; SOTA / baseline comparison; use of the reference
   papers), plus "fidelity of our setup" and "statistics". Split each weight among concrete Pass/Fail
   leaves (e.g. "majority vote is in the results table", "every spec element has a row in the spec
   vs implementation table", "every Arbor reply is quoted verbatim", "every number has a file
   source"). Stop splitting when a leaf weighs less than 3.
2. Apply it to the current ITERATION_1.md yourself and give the score with each failed leaf.
3. Optional: ask one LLM judge to apply the same rubric and compare its verdicts with yours; record
   disagreements. Record model, date, temperature.
4. Review 2026-11-06 (DeepScientist App. A.2; BORROWED_IDEAS_REVIEW.md B-23, B-20). Add these leaves.
   The first three are what human reviewers attacked in DeepScientist's own papers:
   - "every baseline a reviewer would expect is in the table (incl. tuned majority and a published
     DLM selector)";
   - "every claimed gain has an ablation";
   - "closely related work is compared, not only cited";
   - "every selector reports its extra compute per question (R8)".

Commit on iter2/<name> and push. Report the score and the failed leaves.
```

---

## Review summary (why these prompts)

### Did iteration 1 meet the brief?

| Clause | Verdict | Why |
|---|---|---|
| Pick the best of the top 5 | done | `IDEA_SELECTION.md`. Team confirmation not recorded; the selection missed a lineage error (I-13) |
| Codify it using LLMs | partial | Arbor rewrote only `solution.py`, the selection rule. The pipeline was written by people with a coding agent, and that role is not documented as method. The CKA half of the idea never went through the LLM loop |
| Spec gaps, before and after trials | done, strong | 20 issues with evidence and fixes; several were real bugs caught before results |
| Refine by feedback to the model | partial | Every Arbor reply is quoted. The idea itself was never revised by a model; most fixes were human edits; no "feedback given" column |
| Final results on a metric and dataset | partial | GSM8K, 300 held-out questions, paired tests. One task of the four the spec names, one step budget, three floor models |
| SOTA or baselines | partial | Strong baselines. No published method reproduced; first-sample accuracy far below published numbers, not measured |
| Use of the reference papers | partial | All four assessed, Arbor used. No ideas borrowed from the other three (fixed by `BORROWED_IDEAS.md` and T1-T5) |

### Where it could have been better

1. **The main claim of idea 1.1 was not tested.** The spec's dependent variable is the Reranking
   Advantage A_rerank = dQ_rerank - dQ_steps(16->32), and research problem P1 asks whether
   reranking beats more denoising steps. Iteration 1 ran one budget (64), so it tested a reduced
   claim ("CKA predicts reranking headroom"). "Idea 1.1 is refuted" holds for AR-likelihood
   reranking, not for the full hypothesis.
2. **Spec elements were cut without all being logged:** SIQA, WinoGrande, HumanEval; DiffuCoder; the
   BERT non-AR control; compute accounting; the random-init CKA floor and normalised score; the
   5,000-segment stratified CKA corpus.
3. **Arbor's gate compared against the seed, not majority vote,** and the macro score was diluted by
   three floor models.
4. **Process:** deadline (5 Nov) possibly missed; stale write-up; Arbor ran 7 of 12 cycles; one node
   skipped by mistake; GPU idle for 8 h. The 2 Nov audit (`audit/task4-codification`) had flagged
   most of these gaps (n = 4, SOTA, step arm, feedback column); only some were acted on.

### Which prompt fixes which gap

| Gap | Prompt |
|---|---|
| The spec's dependent variable (A_rerank, rerank vs more steps) and half of problem P1 were not measured | A1, N1 |
| The idea was never revised *by a model* from our feedback; most fixes were human code edits | N2, N1 (feedback column) |
| No published method was reproduced or compared; our first-sample accuracy is far below published numbers and not explained by a measurement | N3, A2 (fidelity), A4 |
| Arbor's gate compared against the seed, not majority vote; Arbor only edited the selector, not the preservation half | A3, T1 |
| Known open items: dlm.selfscore, novelty check, n = 4 for CKA | A2, N3, A4 |
| Nothing borrowed from DeepScientist, EvoScientist or AutoSOTA | T1 (red lines, idea library), T2 (Elo), T3 (Findings Memory, surrogate + UCB), T4 (experimentation memory), T5 (rubric), A3 (leap rule, ablation stage, promotion rule) |
| The top-5 order rests on saturated scores | T2 |
| Lessons live in notes, not where agents read them | T3, T4 |
| (Review 2026-11-06) Arbor's native guards were off: no protected_paths, no plugin, convergence never fired | T1 (plugin), A3 |
| (Review 2026-11-06) Arbor's own iteration-1 memory recommends the banned cross-model cache | T3 (curated memory), A3 |
| (Review 2026-11-06) eval.py lets select() keep state across items (R7 not enforced) | T1 (patch draft), A3 |
| (Review 2026-11-06) merge_threshold is not enforced by Arbor's merge gate | A3 (reviewer rule), T1 (meta_decide_inject) |
| (Review 2026-11-06) In-item rules over current features are exhausted; gains need new information, a higher ceiling, or a fixed pipeline | A2 (fidelity first), N3 (research report), N4 (failure classes, tuned bar), A1 |

Overall: iteration 1 meets every clause at least in part. It is strongest on finding spec gaps and on
honest, held-out results with paired statistics. Self-grade B now (`NEXT_ITERATION.md`); A- is
reachable if A1, N1-N3 and A3 land.
