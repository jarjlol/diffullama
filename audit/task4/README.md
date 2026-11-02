# Audit of Task 4 (codification) and the project around it — 2026-11-02

Branch `audit/task4-codification`, cut from `assignment/codification` at `cc5f1da`. This is a
review, not a deliverable: it says what exists, what is wrong, and what to do next, with evidence.
Every claim is labelled with how it was checked. "Verified" means run or read at the primary
source on 2026-11-02, and "abstract-level" means taken from an arXiv abstract only.

| File | What it covers |
|---|---|
| [01-inventory.md](01-inventory.md) | Everything that exists: assignments, branches, machines, decisions, people |
| [02-brief-compliance.md](02-brief-compliance.md) | The Task 4 brief, clause by clause, against what we have |
| [03-code-audit.md](03-code-audit.md) | Line-level review of `codification/`, with the bugs already fixed and the ones still open |
| [04-science-audit.md](04-science-audit.md) | Whether idea 1.1, as codified, can answer its own research question |
| [05-novelty-and-sota.md](05-novelty-and-sota.md) | Novelty search and the state-of-the-art methods we should compare against |
| [06-campaign.md](06-campaign.md) | The GPU campaign running on Blackwell: design, safety rules, how to control it |
| [07-improvement-plan.md](07-improvement-plan.md) | Ranked actions: before Thursday, iteration 2, and project level |

## Summary

**Where Task 4 stands.** The engineering is in good shape. There is an Arbor task pack with leakage
and correctness gates, a five-model GPU pipeline, and a CPU selftest that passes. The *science* has
not run: on 2026-11-02 no candidate had been generated on a GPU. A supervised campaign is now
queued on Blackwell and waits for another user's benchmark sweep to release both cards
(`06-campaign.md`).

**Fixed today, before any GPU time was spent.** Each of these would have cost a run or a wrong result:

1. DiffuGPT-S cannot take a 4-shot prompt (512-token context). The run would have crashed on question 1.
2. Llama-2 access gate. Replaced with a SHA-256-identical ungated copy.
3. Dream and LLaDA at 256 steps need ~45 GPU-hours each. Cut to 64.
4. **DiffuGPT's CKA would have been measured with causal attention.** `extract_hidden.py` passed no
   mask, and under SDPA the patched GPT-2 then runs causally. That inflates exactly the quantity the
   idea is about. Verified on CPU on Blackwell.

**The three findings that matter most now:**

- **The headline claim may be untestable as configured.** The claim is that CKA predicts reranking
  headroom. Only four models have an AR base, two of them (DiffuGPT-S/M) are likely near 0% on few-shot
  GSM8K, and `analyze.py` drops models with no headroom. That can leave n = 2, with no correlation at
  all. Adding released checkpoints that have AR bases fixes this cheaply (`04` §1, `07` P1-1).
- **Headroom depends on sampling temperature, and each model uses a different one** (0.2 to 0.95).
  Across models the CKA-vs-headroom comparison is confounded by the sampler, not just the model (`04` §2).
- **Arbor's merge gate re-uses the test set every cycle.** After up to 12 gated merges, the final test
  score is selected on test and is optimistic. A three-way split costs no extra generation (`04` §4).

**Brief compliance.** Strong on "identify issues". Missing evidence on "refine by giving feedback to
the model" and "compare with SOTA" (`02`). Recent work gives concrete SOTA comparators: an
ORM reranker on Dream for GSM8K, and Self-Repulsive Sampling on LLaDA with 10 paths. Both use
*Instruct* checkpoints, so we should add those (`05`).

**Project level.** The main-project direction (P-1/P-5) has been pending since September. Idea 1.1 *is*
the inference-only L6 + adaptation-residue direction that `project-docs/10` recommended. Ratifying it
as the main project turns this assignment's results into stages 2–4 instead of a side track (`07` §C).
