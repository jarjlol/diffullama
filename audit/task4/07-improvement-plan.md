# 07 — Improvement plan

Ranked by value per hour. **P0** = before the Thursday submission, cheap, no extra GPU.
**P1** = high value, worth GPU time in iteration 2 (or iteration 1 if the cards free early).
**P2** = good, later. References point to the audit file with the evidence.

## A. Task 4, before Thursday (P0)

| # | Action | Why | Cost | Who |
|---|---|---|---|---|
| P0-1 🟡 issues logged a5f2c02, feedback column still to add | Log I-13 (idea 1.1's lineage error) and the four audit fixes (C-4, C-5, C-9, C-10) in `ITERATION_1.md`, and add a "feedback given to the model" column to the issue log | the brief grades the refinement loop; we have the evidence but have not written it down (`02`) | 30 min | anyone |
| P0-2 | Three-way split for Arbor: iterate on dev-A, gate on dev-B, report test once; merge threshold about 2 SE | otherwise the reported test score is selected on test (`03` C-7, `04` §4) | 1 h, edits to protected files before Arbor first runs | Aaditya / Neel |
| P0-3 | `interaction_mode: review` and the feedback table | brief compliance (`02`) | 5 min | whoever runs Arbor |
| P0-4 ✅ done 97ebc8d | CKA robustness: also pool positions 1…127, and add debiased CKA | protects the one novel claim from two known artefacts (`03` C-9, C-10) | 1 h of code; **must land before the hidden-state jobs launch**, which are last in the queue | Aaditya |
| P0-5 | Paired statistics: per-model bootstrap CI, and McNemar selector vs majority, in `baselines.py --json` or a new `stats.py` | "beats majority vote" is not a claim without them (`04` §4) | 1 h, CPU only | anyone |
| P0-6 | Commit the pre-registered predictions (`04` §7) before the first test table | turns iteration 1 into a test, not a search | 15 min | team |
| P0-7 | Ask atharva for a GPU window; agree a claiming convention | the campaign cannot start otherwise (`06`) | a message | Aaditya / Arjun / Aryan |
| P0-8 | Get the 997 uncommitted lines in Blackwell's `~/diffullama` committed by their owner | someone's work exists only on that disk (`01`) | a message | whoever edited ideation on 2026-10-21 |

## B. Task 4, iteration 2 (P1, GPU)

| # | Action | Why | Cost |
|---|---|---|---|
| P1-1 | **Add three AR-based diffusion models:** `diffullama-gsm`, `Dream-v0-Instruct-7B`, `DiffuCoder-7B-Base` | n = 4 → 7 for the headline correlation; two within-family pairs; real headroom on GSM8K (`04` §1) | ~11 h each on the subset |
| P1-2 | **Add `LLaDA-8B-Instruct`** (control) **and Dream-Instruct** from P1-1 | makes our numbers comparable with 2609.39560 and 2609.35472 (`05`) | one more 7B run |
| P1-3 | **Step-budget arm**: K = 16 on the test subset for every model | answers P1(b), the half of the research question currently dropped (`04` §3) | ~+25% of the 64-step time; needs a `--steps` override and a separate output tag |
| P1-4 | **Per-question preservation**: DLM vs base similarity on each prompt, then correlate with per-question rerank success | makes the novel claim testable at n = 500 per model instead of n ≈ 4 (`04` §1) | one forward pass per question per model + base; small |
| P1-5 | Greedy / low-temperature single-answer reference per question | "first" is a temperature sample, which inflates headroom (`04` §5) | +10% generation |
| P1-6 | Temperature control: report answer diversity per model; calibrate or run two temperatures on dev | headroom is confounded by sampler temperature (`04` §2) | one dev run per model |

## C. Project level

1. **Ratify the main-project direction as idea 1.1.** P-5 has been pending since 2026-09-21.
   `project-docs/10` recommended L6 (answer-selection headroom) as primary and the adaptation-residue
   audit as secondary. P1 / idea 1.1 is exactly L6 plus residue (CKA). Making it the main project:
   - turns this assignment's runs into stage 2 (reproduction: our first-sample and published numbers)
     and stage 4 (beat the baseline: selector vs majority vote);
   - ends a split where the course assignments and the "main project" were drifting apart.
2. **Merge to `main`.** `main` is at 2026-09-28 and has none of ideation or codification. After
   Thursday, merge `assignment/codification`, which already contains research-ideation.
3. **Update `project-docs`**: `01-project-brief.md` status rows, the decision log (2026-11-02
   decisions: subset, 64 steps, overrides, campaign), established facts (C-4 causal-SDPA trap, DiffuGPT-S
   512 context, the Llama-2 mirror hash, published GSM8K numbers), and mistakes (lineage error missed
   by the selection).
4. **Venue (Q-1)** is still unresolved, and it sets how polished stage 5 must be. Ask TTV.
5. **Work split.** Four of seven members cannot reach a GPU. The CPU work they can own now: P0-1,
   P0-5, P0-6, P1-4's analysis code, the full-text verification of the `05` numbers, and the stage-5
   outline.
6. **Tooling papercuts:**
   - The LFS pre-push hook times out on this network. One LFS file in the repo
     (`LLaMA-Factory/evaluation/mmlu/mmlu.zip`) makes every push depend on GitHub's LFS endpoint.
   - The GitHub MCP connector fails authentication.
   - The selftest needs Git Bash first in PATH on Windows.

## D. What Thursday's submission can honestly claim, by scenario

| If by Wednesday night… | Submit |
|---|---|
| no GPU window | issues I-1…I-13 with fixes and feedback evidence; harness and campaign; the verified published numbers; pre-registered predictions; the plan. Results: none, stated plainly |
| only the DiffuGPT models ran | the above plus the first real trial results (I-11): near-zero few-shot accuracy, which itself motivates P1-1 |
| DiffuGPT + DiffuLLaMA ran | plus the first selector-vs-majority comparison on a 7B anchor model |
| everything ran | the full iteration-1 tables, Arbor in review mode on the dev split, the CKA table (descriptive) |
