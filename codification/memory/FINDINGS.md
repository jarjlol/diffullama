# Findings Memory — readable view (T3, neel's agent, 21 Nov 2026)

Generated from `FINDINGS.jsonl` (53 records). Rebuild it with `python3 codification/memory/build_findings.py`, and verify every number against its evidence file with `--check` (53/53 ok on 2026-11-21). Tiers follow DeepScientist §3.2: idea → implement → progress. Validation fields follow EvoScientist App. D.

## Promotion rule (how a finding moves up a tier)

1. **idea → implement**: the hypothesis is coded as a selector that passes `eval.py`'s gates and has a dev score.
2. **implement → progress**: it beats tuned majority on the **gate** split with a paired test (McNemar and a bootstrap CI excluding 0), per model, not only on the macro score.
3. **progress → claim in the paper**: it holds on the **final** split, touched once with the rule written down beforehand; an ablation removes the new signal and the gain disappears; and a **second dataset** shows the same sign (BORROWED_IDEAS.md §3).
4. A finding that fails a step is kept with its `outcome` and `do_not_repeat`. It is never deleted.
5. Numbers come only from a committed file and are checked by `--check`. Anything else is `[unverified]` and cannot be promoted.

## Tier: progress

- **arbor-4.1** (NOT_FAILED): verification-weighted voting: weight 1.0 if every parsed calculation re-checks, else 0.5. Result: dev 0.296, gate 0.296; final 300: Dream -0.3 pts vs majority (3/4 wins/losses), LLaDA 0.0 (3/3). Verdict: indistinguishable from majority vote; merged only against the seed. Evidence: `codification/ITERATION_1.md`.
- **I-20** (FAILED (WorseThanBaseline)): a small AR scorer reranks DLM candidates above the first sample. Result: Dream test n=500: first 40.4%, oracle 87.8%; gpt2 per-token 34.6%; majority 72.4%. Verdict: refuted for iteration-1 scores; pending re-check on cut text (I-22). Evidence: `codification/ITERATION_1.md`.
- **Selector records**: 14 (final300). Non-reference selectors above majority on any model: ar.gpt2.mean (diffugpt-m, diffullama), ar.llama2-7b.mean (diffugpt-m), ar.qwen0.5b.mean (diffugpt-m), ar.qwen7b.mean (diffugpt-m), ar.tinyllama.mean (diffugpt-m). Every such case is a floor model (DiffuGPT-S/M, DiffuLLaMA) at a difference of one to a few questions. On Dream and LLaDA no AR selector beats majority (iteration-1 scores, see I-22).

## Tier: implement

- **arbor-2.1** (FAILED (WorseThanBaseline), do not repeat): majority vote with a qwen7b per-token NLL tie-break. Result: dev macro 0.290 (14 turns, 460 s); below majority on dev. Verdict: AR likelihood adds nothing on top of the vote. Evidence: `codification/ITERATION_1.md`.
- **arbor-3.1** (FAILED (WorseThanBaseline), do not repeat): likelihood-weighted voting, sum of exp(-per-token NLL) per answer. Result: dev macro 0.282 (12 turns, 166 s). Verdict: AR likelihood as a vote weight loses to plain majority. Evidence: `codification/ITERATION_1.md`.
- **I-16** (FAILED (WorseThanBaseline), do not repeat): Dream at its recommended temperature 0.2 gives enough candidate diversity to rerank. Result: smoke: 2.4 distinct answers per question, first = oracle = 60%. Verdict: refuted; Dream raised to temperature 0.7. Evidence: `codification/ITERATION_1.md`.
- **I-17** (FAILED (NoExecutableWithinBudget)): WikiText-103 validation alone gives 2000 CKA segments. Result: 1854 < 2000 segments of 128 GPT-2 tokens. Verdict: fixed: validation then test. Evidence: `codification/ITERATION_1.md`.
- **I-18** (FAILED (NoExecutableWithinBudget)): long CUDA kernels run safely on both Blackwell GPUs. Result: the display watchdog sometimes kills a CUDA kernel. Verdict: operations pitfall. Evidence: `codification/ITERATION_1.md`.
- **I-19** (FAILED (WorseThanBaseline)): qwen0.5b reranking reads 0.000 because the scorer fails. Result: parser bug: scorer name split at the dot; after the fix 1.0-2.2% on DiffuGPT. Verdict: bug in our harness, fixed. Evidence: `codification/ITERATION_1.md`.
- **Selector records**: 28 (dev, test500). Non-reference selectors above majority on any model: ar.gpt2.mean (diffullama), ar.gpt2.sum (diffullama), ar.llama2-7b.mean (diffullama), ar.llama2-7b.sum (diffullama), ar.qwen0.5b.mean (diffullama), ar.qwen0.5b.sum (diffullama), ar.qwen7b.mean (diffullama), ar.qwen7b.sum (diffullama), ar.tinyllama.mean (diffugpt-s, diffullama), ar.tinyllama.sum (diffullama), ar.gpt2.mean (diffullama), ar.gpt2.sum (diffugpt-s), ar.llama2-7b.mean (diffugpt-s), ar.qwen0.5b.mean (diffugpt-s), ar.qwen7b.mean (diffugpt-s), ar.qwen7b.sum (diffugpt-s), ar.tinyllama.mean (diffugpt-m). Every such case is a floor model (DiffuGPT-S/M, DiffuLLaMA) at a difference of one to a few questions. On Dream and LLaDA no AR selector beats majority (iteration-1 scores, see I-22).

## Tier: idea

- **arbor-node-1** (NOT_FAILED): plain majority vote as a selector. Result: approved at review; never merged (merge gate compared against the seed). Verdict: bar, not a finding. Evidence: `codification/ITERATION_1.md`.
- **arbor-node-2** (FAILED (WorseThanBaseline), do not repeat): cross-model consensus: defer to another model's cached answer for the same question. Result: rejected twice at idea review (cycle 0 node 2, cycle 2 ROOT). Verdict: out of scope: breaks red line R7 (one model's N samples). Evidence: `codification/ITERATION_1.md`.
- **arbor-node-1-child** (FAILED (NoExecutableWithinBudget)): plain majority again, as a depth-2 experiment. Result: skipped at review as a duplicate. Verdict: process note: it was the concrete run of node 1. Evidence: `codification/ITERATION_1.md`.

