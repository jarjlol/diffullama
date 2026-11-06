# Task 4: prompts for Neel and Aryan (written 2026-11-06)

These are prompts to paste into a coding agent (Claude Code, opencode or Codex) started in the repo root
on branch `assignment/codification`. Each prompt is self-contained. Run them in the order given. The
review that produced them is summarised at the end of this file.

Who runs what:
- **Neel** (no workstation access): N1 to N4. Everything is CPU or API only.
- **Aryan** (workstation access): A1 to A4 on Blackwell. A1 and A2 can run in parallel, one per GPU.
  A3 needs the output of A1 and A2. A4 is optional and needs model downloads.

Each person works on their own branch (`iter2/neel`, `iter2/aryan`), commits small, and pushes. Merge
into `assignment/codification` after the other has read the diff.

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

Commit on iter2/aryan, push via the laptop. Report real outputs. Free the GPUs when done.
```

---

## A3 (Aryan, GPU, after A1 + A2): Arbor iteration 2 with the right baseline

```text
Same rules as A1. Read HANDOFF.md §3 "Rerun Arbor" and NEXT_ITERATION.md "Process lessons" first.

Iteration 1's merge gate compared against the seed rule (GPT-2 NLL), not majority vote, so Arbor
could merge a rule that only ties majority. Fix that, and carry iteration 1's findings into the run
(the "findings memory" idea that DeepScientist and EvoScientist build their loops on; see
codification/REFERENCE_SYSTEMS.md).

1. Rebuild the Arbor task repo (codification/make_arbor_repo.sh) into ~/arbor-rerank-iter2 with:
   - seed solution.py = majority vote (copy baselines._majority logic);
   - the new features from A1/A2 in items (dlm.self, the 7B scorers; candidates at K = 16/32 only if
     you also give Arbor a compute budget per question, otherwise leave them out);
   - research_config.yaml: interaction_mode review. Note that merge_threshold is a RELATIVE percent
     gain on the macro score over all 5 models, and the 3 floor models dilute it. Decide with the
     team whether eval.py (a protected file, change it only before Arbor starts) should print a
     score over dream-base and llada-base only, so the gate means "beats majority vote where there
     is headroom". One point on a 200-question split is 2 questions; ask for at least 3 net questions.
2. Write the instruction from codification/arbor_instruction.txt plus a "What iteration 1 learned"
   block: the four directions it tried and their dev scores (results/iter1/arbor_session/REPORT.md),
   that AR likelihood is anti-predictive on Dream, that cross-model consensus is out of scope, that
   verification weighting tied majority. Tell it not to retry these.
3. Start vLLM only when you are ready to answer review prompts; stop it the moment Arbor ends.
   Answer every gate with /approve, /skip, /edit <note> or /answer <text>, and paste each reply
   verbatim into a new §Arbor iteration 2 table in ITERATION_1.md's successor (codification/ITERATION_2.md).
4. After Arbor ends, score the merged rule (if any) on the final split once with
   codification/eval_final.sh and codification/paired_stats.py against majority vote.

Copy the session folder to codification/results/iter2/arbor_session/, commit, push via the laptop.
```

---

## A4 (Aryan, optional, needs downloads): models that make the comparison fair

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
GPU time: measure on --limit 5 and agree a window with the team first.
```

---

## Review summary (why these prompts)

Iteration 1 meets every clause of the brief at least in part. It is strongest on finding spec
gaps (20 issues, before and after trials, with evidence) and on honest results with paired
statistics on a held-out split. It is weakest in four places, which the prompts target:

| Gap | Prompt |
|---|---|
| The spec's dependent variable (A_rerank, rerank vs more steps) and half of problem P1 were not measured | A1, N1 |
| The idea was never revised *by a model* from our feedback; most fixes were human code edits | N2, N1 (feedback column) |
| No published method was reproduced or compared; our first-sample accuracy is far below published numbers and not explained by a measurement | N3, A2 (fidelity), A4 |
| Arbor's gate compared against the seed, not majority vote; Arbor only edited the selector, not the preservation half | A3 |
| Known open items: dlm.selfscore, novelty check, n = 4 for CKA | A2, N3, A4 |
