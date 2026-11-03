# Codification — iteration 1 (due Thursday 2026-11-05)

What the first submission has to show, mapped to the brief, plus the run plan that gets
there on one shared GPU in three days. The full runbook stays in `README.md`; this file is
the iteration-1 slice of it and the running log of issues found.

## What the brief asks for, and where each part lives

| Brief | Iteration 1 deliverable | Where |
|---|---|---|
| Codify the best of the top-5 ideas, using LLMs | Idea 1.1 chosen; codified through Arbor (LLM edits `solution.py`, merges only on held-out gain) | `IDEA_SELECTION.md`, `REFERENCE_SYSTEMS.md`, `rerank_task/` |
| Identify spec gaps / infeasibilities, before or after initial trials | Issue log below: I-1–I-6 before codification, I-7–I-10 and I-12–I-15 found while preparing the run, I-11 and I-16+ from the first GPU trials | this file |
| Refine the idea by giving feedback to the model | The feedback given to Arbor, verbatim, plus its session folder | §Arbor, `.arbor/sessions/<run>/` |
| Final results on a relevant metric and dataset | GSM8K accuracy per model: first / majority / AR-rerank / Arbor selector / oracle, plus CKA vs headroom | §Results |
| Compare with SOTA / baseline ideas | Majority vote (self-consistency, the anchor's own SC baseline) is the bar; published base-model GSM8K numbers check generation fidelity | §Published numbers |

Iteration 1 is allowed to be partial. It must be honest about what ran, on what subset.

## Issue log

Before codification (from reading the spec, `IDEA_SELECTION.md` §Known issues):

| # | Issue | Fix |
|---|---|---|
| I-1 | No self-consistency or published baselines | `baselines.py`: first, random, majority, AR sum/mean, oracle |
| I-2 | Summed NLL favours short answers | Per-token NLL reported alongside. The seed keeps the summed version so Arbor has something to fix |
| I-3 | CKA-vs-gain correlation across ~4 models has no power | Exact permutation p, labelled descriptive; per-model accuracy carries the result |
| I-4 | No from-scratch control | `llada-base` (no AR base) |
| I-5 | GSM8K answer extraction unspecified; HumanEval harness unspecified | `task.py` extraction; GSM8K only |
| I-6 | Novelty unchecked | Open, see §Still open |

Found while preparing the run (2026-11-02, before any GPU trial):

| # | Issue | Evidence | Fix |
|---|---|---|---|
| I-7 | **DiffuGPT-S cannot take a 4-shot GSM8K prompt.** It was adapted at context 512 (anchor App. B training table: DiffuGPT-S 512, DiffuGPT-M 1024; HF config `max_length` 512). The 4-shot prompt alone is 581 GPT-2 tokens at the median, so `generate_candidates.py` would have raised on the very first question. | GPT-2 tokenizer over the real prompts: 4-shot + 256 > 512 for 1319/1319 test questions; 1-shot + 192 for 0/1319 | Per-model overrides in `config.json`: `diffugpt-s` runs **1-shot, 192 new tokens**. 192 covers GSM8K gold solutions to p99 (181 tokens; max 237). Stated confound: shot count differs for this one model |
| I-8 | DiffuLLaMA's CKA base `meta-llama/Llama-2-7b-hf` is gated (manual approval) | HF API | Use `NousResearch/Llama-2-7b-hf`. Both safetensors shards and `tokenizer.model` have identical SHA-256 to Meta's (HF API `?blobs=true`) |
| I-9 | **No published few-shot GSM8K number exists for DiffuGPT or DiffuLLaMA.** The anchor's GSM8K column (Table 1, ∗) is after fine-tuning on augmented GSM8K-symbolic data; its few-shot/SC/hit@k study (Table 2) is on MAWPS, SATMath and TriviaQA. LLaMA2-7B itself scores 13.1 4-shot (LLaDA Table 1), so DiffuLLaMA and especially DiffuGPT-S/M may sit near the floor, which leaves little headroom to rerank | anchor full text, `litreview/data/anchor_fulltext.txt` | Expect it, measure it on the smoke run (I-11), report it. `diffusionfamily/diffullama-gsm` (the released GSM8K LoRA adapter) is a possible iteration-2 addition with real headroom |
| I-10 | `selftest.py` fails on Windows: `bash` resolves to WSL's `System32\bash.exe` | `execvpe(/bin/bash) failed` | Not a pipeline bug (the workstation is Linux). On Windows, run with Git Bash first in the resolution order |
| I-12 | **Dream and LLaDA at 256 denoising steps do not fit the deadline.** The diffusion samplers have no KV cache, so every step reprocesses 10 candidates × ~900 tokens. Estimated ~45 GPU-hours each on the 700-question subset (table in §Run plan) | FLOP estimate: 2 × params × tokens × steps, ~150 TFLOPS effective | Both cut to **64 steps** (4 tokens per step), the same budget as the anchor models. This costs accuracy against their published numbers, which used more steps. Stated confound |
| I-13 | **Idea 1.1 has a model-lineage error** the selection missed: its matched scorer is "TinyLlama for LLaMA-based DLMs (DiffuLLaMA, Dream-7B, LLaDA-8B, DiffuCoder-7B)". Dream and DiffuCoder are Qwen2.5-based, and LLaDA is trained from scratch | `ideation/output/research_problems_and_ideas.md` line 67; `ideation/REPORT.md` §7 lineages | A Qwen2.5-0.5B scorer is added next to TinyLlama, so Dream gets a same-family scorer. LLaDA has no base and is the control |
| I-14 | **DiffuGPT's CKA would have been computed with causal attention.** `extract_hidden.py` passed `attention_mask=None`. DiffuGPT loads with SDPA attention, where the patched GPT-2 forward turns `None` into `is_causal=True`, and `model.py`'s `attn.bias.fill_(True)` only affects eager attention. That inflates exactly the preservation score being measured | CPU probe on Blackwell: when the last token changes, earlier hidden states move by 0 with `None` and by 4.36 with the all-zero 4D mask | Always pass the all-zero 4D mask that `generate_samples` uses. `check_attention()` aborts unless a DLM is bidirectional and an AR base is causal |
| I-15 | **Plain linear CKA is unreliable here in two ways.** (a) It is biased upward when n = 2000 is not much larger than d, by different amounts for widths 768–4096. (b) It is dominated by outlier directions such as the attention-sink activation causal LMs put on position 0 | synthetic check on Blackwell: one outlier direction moves CKA from 0.537 to 0.0007; debiasing moves 0.537 to 0.360 | `extract_hidden.py` also saves pooling over positions 1..end. `cka.py` adds debiased CKA (unbiased HSIC). `measure_preservation.py` records all four variants, and `analyze.py` reports the correlation under each. The specified score stays primary; if the ranking changes under the variants, it is not reported as preservation |

After initial trials (fill in from the smoke runs):

| # | Issue | Evidence | Fix / feedback given |
|---|---|---|---|
| I-11 | first-sample and oracle accuracy per model on `--limit 20` | campaign gates (Blackwell, 2026-11-02): diffugpt-s first 0% / oracle 15%, 4.1 s/question; diffugpt-m 0% / 20%, 15.9 s/question; dream-base (T 0.2) 60% / 60%, 36 s/question | DiffuGPT is at the floor, as I-9 predicted: reranking can only recover what the oracle shows. LLaDA and DiffuLLaMA pending |
| I-16 | **Dream at its recommended temperature 0.2 gives no candidate diversity.** Reranking N = 10 identical answers recovers nothing, so Dream would add no information to the CKA-vs-recovery correlation, and the near-zero result would reflect the temperature, not preservation | Dream smoke: 2.4 distinct answers per question, first = oracle = 60%; most questions have all 10 candidates identical | Dream raised to temperature 0.7, the same as LLaDA, so the two 7B diffusion models sample alike. The 20 smoke rows at 0.2 were set aside (`dev.jsonl.t02`) and the smoke gate re-run. Stated confound: first-sample accuracy is no longer Dream's best-case number |
| I-17 | **CKA corpus too small.** WikiText-103 validation yields only 1854 segments of 128 GPT-2 tokens; the config asks for 2000 | `hidden:diffugpt-s:dlm` failed: `corpus too small: 1854 < 2000` | Corpus is now validation, then test (both held out from training text). Also: offline, `datasets.load_dataset` ignores the HF hub cache, so the parquet files are read directly |
| I-19 | **Every `qwen0.5b` reranking result read 0.000.** Feature keys are `ar.<scorer>.nll_sum`, and `baselines.py` / `analyze.py` took the scorer name as `key.split(".")[1]`, so `qwen0.5b` became `qwen0`, whose features do not exist: no candidate had a score | test: `ar.qwen0.sum` 0.000 for both DiffuGPT models while random gave 1.0-1.4% | Parse the name as everything between `ar.` and the last field. After the fix: 1.0-2.2% on DiffuGPT, in line with the other scorers |
| I-20 | **Reranking by a small AR scorer is worse than taking the first sample wherever there is headroom.** On Dream (first 40.4%, oracle 87.8%, test n=500) every small-scorer selector loses: gpt2 per-token 34.6%, the others 26.8-31.8%; majority vote reaches 72.4%. Summed NLL is worst (it prefers short answers). On the three floor models every selector sits between 0.6% and 2.2% | `baselines.py` dev and test, 2026-11-03 | The idea as specified does not beat the baseline. Feedback, in two parts: (1) score with the AR bases themselves, Qwen2.5-7B for Dream and Llama-2-7B for DiffuLLaMA (`qwen7b`, `llama2-7b`), the strongest form of the matched scorer; (2) give Arbor the target "beat majority vote" and the opening direction recorded in `arbor_instruction.txt` |
| I-18 | **The display server (Xorg) is attached to both Blackwell GPUs**, and the display watchdog sometimes kills a CUDA kernel | `CUDA error: the launch timed out and was terminated`: `diffugpt-m` dev after 49 min (GPU 0), `diffullama` dev after 65 min (GPU 1), `diffullama` test after 190 min (GPU 0), `dream-base` test after 178 min (GPU 1) | Generation resumes by question uid, so each crash loses one question; the campaign allows 8 tries per generation job. Both cards run at their 600 W limit, so a second job per GPU gains nothing |

## Published numbers (verified 2026-11-02 from arXiv HTML full text)

| Model | GSM8K | Shots | Source |
|---|---|---|---|
| LLaDA 8B Base | 70.3 | 4 | LLaDA, arXiv:2502.09992, Table 1 |
| LLaMA2 7B (DiffuLLaMA's base) | 13.1 | 4 | same table |
| Dream 7B Base | 77.2 | 8 | Dream 7B, arXiv:2508.15487v1, Table 1 |
| LLaDA 8B Base (Dream's protocol) | 70.9 | 8 | same table |
| Qwen2.5 7B (Dream's base) | 78.9 | 8 | same table |
| DiffuLLaMA, DiffuGPT-S/M | none few-shot | — | anchor Table 1 is fine-tuned (63.1 / 50.2 / 61.8), not comparable |

Our first-sample accuracy is not expected to match these exactly. Our samplers run at
temperature > 0 to get 10 distinct candidates, LLaDA runs fully non-autoregressive
(block length = generation length), Dream is 4-shot here, not 8, and both run 64 steps
for 256 tokens (I-12). Report the gap as a
fidelity check, not as a reproduction.

## Run plan for Thursday

Subset for iteration 1: **dev 200, test 500** (first 200 of the shuffled train-derived dev,
first 500 of the official test). `--limit` restricts generation only; `assemble.py`,
`eval.py` and `baselines.py` then score exactly the generated questions. Every step
resumes, so iteration 2 re-runs the same commands with a larger `--limit`.

Order is cheapest first, so problems surface before the 7B runs:

1. **Venvs and smoke test** (README §B). The four venvs on Blackwell with a current torch
   build are untested. `--limit 5 --split dev` on each backend. Check that `generate_candidates.py`
   prints the shot count and new-token count you expect (`diffugpt-s`: 1-shot, 192).
2. **Smoke accuracy (I-11).** `--limit 20` on every model, score with `gpt2`, assemble, run
   `baselines.py --split dev`. A model whose oracle is ≈ 0 has no headroom to study. Record it
   and decide whether to keep it before spending GPU hours on it.
3. **Timing gate.** `generate_candidates.py` prints s/question every 10 questions. Multiply
   by 700 (200 dev + 500 test) per model and replace the estimate below before launching.
   If the 7B total does not fit before Wednesday night, cut `test` to 300, never `dev` below 200.

   Paper estimate (2026-11-02, not measured; 2 × params × tokens × steps at ~150 TFLOPS
   effective, so trust it to about 2× either way):

   | Model | Steps | s/question | 700 questions |
   |---|---|---|---|
   | diffugpt-s | 64 | ~1 | ~15 min |
   | diffugpt-m | 64 | ~3 | ~35–60 min |
   | diffullama | 64 | ~50 | ~10 h |
   | dream-base | 64 (was 256: ~43 h) | ~55 | ~11 h |
   | llada-base | 64 (was 256: ~47 h) | ~60 | ~12 h |
   | scorers + CKA | — | — | < 1 h total |

   About 34 GPU-hours in all, about 23 h with `test` cut to 300 for the three 7B models.
   With both cards free, run two 7B models side by side, one per card.
4. **Full subset runs**: diffugpt-m, diffugpt-s, diffullama, dream-base, llada-base.
5. **Scoring**: all three scorers (`gpt2`, `qwen0.5b`, `tinyllama`) on both splits. These are small and fast.
6. **CKA** for the four models with a base (README §B step 5).
7. **Arbor** (README §C), then `baselines.py --split test --json` and `analyze.py --split test`.

Check `nvidia-smi` before every launch and never kill another user's job.

```bash
cd codification/pipeline
export HF_HOME=/path/to/hf-cache CODIFY_WORK_DIR=/fast/disk/codify-work
python3 prepare_data.py
# smoke
~/venvs/anchor/bin/python generate_candidates.py --model diffugpt-s --split dev --limit 5
# subset (per model, in its venv)
~/venvs/anchor/bin/python generate_candidates.py --model diffullama --split dev  --limit 200
~/venvs/anchor/bin/python generate_candidates.py --model diffullama --split test --limit 500
```

## Arbor: the feedback loop

Run with `ui.interaction_mode: review` (edit `rerank_task/research_config.yaml`) so every
proposed hypothesis passes through a person. Paste what you told it, verbatim, into the
table. This table is the brief's "refine the idea by providing feedback to the model".

Opening direction to give it: (1) summed NLL is length-biased, try per-token; (2) the
candidates' extracted answers agree or disagree, and agreement is a signal (majority vote
already beats most single-sample selectors); (3) the three scorers can be combined.

| Cycle | Arbor proposed | Feedback given | Dev score | Test score | Merged? |
|---|---|---|---|---|---|
| seed | lowest summed GPT-2 NLL | — | | | — |
| 0 (node 1) | plain majority vote | `approve` | | | |
| 0 (node 2) | cross-model consensus: defer to another strong model's cached answer for the same question | Replaced with: "Mechanism: majority vote with an AR tie-break, using only this item's own 10 candidates. Rejected the cross-model cache: reading other models' answers to the same question is cross-model ensembling, which changes the task (one model's N samples), is unfair to every per-model baseline, and makes the score depend on evaluation order. Rule: count votes over extracted answers; among answers tied for the top count, pick the answer whose best candidate has the lowest per-token NLL (nll_sum / n_tokens) under qwen7b; earliest index as the last resort; all-None -> 0. Hypothesis: ties are where majority vote loses most, and per-token NLL from the strongest scorer may separate tied answers even though it is anti-predictive across all candidates. Observable: dream-base and llada-base dev >= majority (0.700 / 0.745); if not, report that AR likelihood adds nothing on top of the vote." | | | |
| 0 (node 3) | likelihood-weighted voting (sum of exp(-per-token NLL) per answer), as a falsification test; claimed llada's AR base is llama2-7b | Replaced with the same idea plus: "Correction to the proposal: LLaDA has no AR base model; it is the control, so do not call llama2-7b its base." Scorers: dream-base and others qwen7b, diffullama llama2-7b. Report per-model dev scores | | | |
| 0 (node 1 child) | plain majority again, as a depth-2 experiment | `skip` (read as a duplicate; it was the concrete run of node 1, see `NEXT_ITERATION.md`) | | | — |
| 0 (node 2.1) | majority + qwen7b per-token tie-break, concrete | `approve` | 0.290 (14 turns, 460 s) | | |
| 0 (node 3.1) | weighted vote, concrete, scorer mapping corrected | `approve` | 0.282 (12 turns, 166 s) | | |
| 1 (exec 2.1, 3.1) | pre-execution review of both | `approve`, `approve`; both executors started in parallel | | | |
| 2 (ROOT) | cross-model consensus again, as a depth-1 direction | Replaced with: "Direction: within-item arithmetic verification as a vote weight. Rejected again: cross-model consensus is out of scope for this task (one model's N samples per item; reading other models' answers is ensembling, unfair to the per-model baselines and order-dependent). Do not propose it again. Instead use the candidate text itself: parse every explicit calculation in a candidate's reasoning (patterns like "a op b = c", including GSM8K "<<a*b=c>>" annotations), re-evaluate it exactly, and record the fraction of correct calculations and whether the final answer equals the result of the last calculation. Weight each candidate's vote by a simple function of that (e.g. 1 if all checks pass, 0.5 otherwise), then majority over weighted votes; ties -> earliest index; all-None -> 0. Hypothesis: when majority is wrong, the wrong plurality often comes from samples with an arithmetic slip, which a deterministic checker can see and AR likelihood cannot. Observable: dream-base and llada-base dev >= majority (0.700 / 0.745); report per-model counts of majority-wrong items that the weighting fixes and breaks." | | | |
| 2 (node 4 child) | verification-weighted voting: weight 1.0 if every parsed calculation re-checks, else 0.5; predicts dream +3 items, llada -1 | `/approve` | | | |
| 2 (exec 4.1) | pre-execution review | `/edit Keep the weight at 0.5 as proposed; do not sweep it on dev (the gain is a few items and a sweep would overfit). Report per model the majority-wrong items fixed and the majority-right items broken, and the share of candidates with at least one checked calculation.` | 0.296 (25 turns, 390 s) | 0.296 (gate) | yes, against the seed's 0.157 gate score; majority vote alone was never merged, so this merge does not show 4.1 beats majority (checked on the final split) |

## Results

Arbor run `iter1b` (2026-11-04, 68 min, 7 of 12 cycles, stopped on its own): 4 ideas
executed, 1 merged (4.1, verification-weighted voting), AR-likelihood ideas pruned as
verified negatives. Session: `~/arbor-rerank/.arbor/sessions/iter1b/` on Blackwell.

Final split: the 300 GSM8K test questions Arbor never saw (`split_gate_final.py`), scored
with `eval_final.sh` and `baselines.py`. Accuracy per model:

| Model | Shots | First | Majority | AR best (scorer) | Arbor selector (4.1) | Oracle | Headroom closed by 4.1 |
|---|---|---|---|---|---|---|---|
| diffugpt-s | 1 | 1.7 | 2.3 | 2.0 (llama2-7b mean) | 2.3 | 14.7 | 5% |
| diffugpt-m | 4 | 0.7 | 1.3 | 2.3 (tinyllama mean) | 1.3 | 8.7 | 8% |
| diffullama | 4 | 0.3 | 2.0 | 2.3 (gpt2 mean) | 2.0 | 9.7 | 18% |
| dream-base | 4 | 41.0 | **73.3** | 32.0 (qwen7b sum) | 73.0 | 88.0 | 68% |
| llada-base | 4 | 48.7 | **73.0** | 52.0 (qwen7b sum) | 73.0 | 88.3 | 61% |
| macro | | | | | 30.3 | | |

Paired comparison, `paired_stats.py` (95% bootstrap CI over questions, exact McNemar on
discordant pairs; "wins/losses" = questions only 4.1 / only majority gets right):

| Split | Model | 4.1 − majority | 95% CI | wins/losses | McNemar p | 4.1 − first | p |
|---|---|---|---|---|---|---|---|
| final (n=300) | dream-base | −0.3 | [−2.0, +1.3] | 3/4 | 1.00 | +32.0 | 8e-27 |
| final (n=300) | llada-base | 0.0 | [−1.7, +1.7] | 3/3 | 1.00 | +24.3 | 2.5e-16 |
| gate (n=200) | dream-base | +1.5 | [−0.5, +4.0] | 4/1 | 0.38 | +33.0 | 3e-17 |
| gate (n=200) | llada-base | −1.5 | [−3.5, 0.0] | 0/3 | 0.25 | +23.5 | 1e-12 |
| dev (n=200) | dream-base | +1.5 | [0.0, +3.5] | 3/0 | 0.25 | +34.0 | 1e-19 |
| dev (n=200) | llada-base | −0.5 | [−2.5, +1.0] | 1/2 | 1.00 | +26.5 | 1e-13 |

On the three floor models 4.1 and majority pick the same answer on every question.

What this says:
1. **Idea 1.1 as specified is refuted.** Picking the candidate with the lowest AR-scorer NLL
   is worse than the first sample on Dream (best 32.0 vs 41.0) and no better on LLaDA
   (52.0 vs 48.7), even with each model's own 7B AR base as scorer. Majority vote beats every
   AR selector by 21+ points on both. Using AR likelihood only to break vote ties (Arbor 2.1)
   or to weight votes (3.1) also lost to plain majority on dev.
2. **The best selector found is equivalent to self-consistency.** Arbor's merged rule
   (arithmetic-verification-weighted voting) is indistinguishable from majority vote on
   every split; its dev gain on Dream (+1.5, 3/0) did not hold on the final split (3/4).
   The merge gate compared against the seed (GPT-2 NLL, 0.157), not against majority.
3. **Majority vote closes 61-69% of the first-to-oracle gap** on the two non-floor models;
   the remaining 15 points of oracle headroom are not reachable with any in-item signal
   Arbor tried.

CKA (mean over layers) vs headroom, `analyze.py --split test`: Spearman ρ and the exact
permutation p, n ≤ 4, descriptive only.

## Still open after iteration 1

- Novelty check (I-6). The defensible claim is the preservation → gain link, not reranking itself.
- `dlm.selfscore` (the DLM's own-likelihood reranker) is not computed.
- No published test-time selection method for diffusion LMs has been compared yet.
- The step-budget arm of idea 1.1 (16/32/64/128 steps) is out of scope. One step budget per model.
