# Codification: where iteration 1 stands, and what to change next time

Written 2026-11-04, overnight while Arbor run `iter1b` is going. Self-assessment, not the
instructor's. Update the status column once Arbor and the final-split evaluation are done.

## Progress against the brief

| Brief part | Status (2026-11-04 02:00 IST) | Done |
|---|---|---|
| Codify the best idea using LLMs | Pipeline built (generation, CKA, AR scoring, baselines, Arbor task pack). Arbor `iter1b` running on Blackwell: 4 nodes approved, none executed yet | ~70% |
| Spec gaps / infeasibilities | I-1 to I-20 in `ITERATION_1.md`, each with evidence and a fix; includes the lineage error in the idea itself (I-13) and the CKA causal-mask bug (I-14) | ~95% |
| Refine via feedback to the model | Two rounds: I-20 (matched 7B scorers, target "beat majority vote") and Arbor review replies, logged verbatim in `ITERATION_1.md` §Arbor | ~40% |
| Final results, metric + dataset | GSM8K test n = 500 per model for first / majority / random / AR sum+mean (5 scorers) / oracle; CKA for 4 models. Arbor selector on the 300 held-out final questions still to run | ~75% |
| Compare with SOTA / baselines | Strong internal baselines (self-consistency, oracle, 5 AR scorers); published base-model GSM8K numbers as a fidelity check. No published test-time selection method for diffusion LMs compared | ~50% |

Overall about 70% of iteration 1. Remaining: Arbor finish, `eval_final.sh` on the final
split, bootstrap CIs and McNemar vs majority, results tables, push.

## Self-grade and why

If the remaining steps land: roughly B+ to A-. As of now: B.

Strong:
- Honest negative result with a clean design: the idea as specified (AR-likelihood
  reranking) loses to majority vote by 20+ points on Dream and LLaDA, and the strongest
  form of it (each DLM's own 7B AR base as scorer) does not rescue it.
- Spec-gap log is concrete and evidenced; several gaps were real bugs caught before they
  contaminated results (I-7, I-14, I-15, I-19).
- Held-out discipline: dev for Arbor, 200-question gate for merges, 300 final questions
  Arbor never sees.

Weak:
- The CKA-vs-recovery correlation has n = 4 models, so it cannot support a claim; the
  "preservation predicts gain" half of the idea is untested in any real sense.
- Three of five models sit at the floor (first sample < 2%), so they add little signal.
- Fidelity confounds: 64 denoising steps, 4-shot (Dream published 8-shot), Dream at
  T 0.7, DiffuGPT-S at 1-shot. Our numbers are below published ones and that is explained,
  not fixed.
- No comparison with a published DLM test-time method, and the DLM's own likelihood
  (`dlm.selfscore`) is not computed.
- No significance tests yet.
- Novelty (I-6) is still unchecked.

## Ideas for iteration 2 (ranked by value per GPU-hour)

1. **Paired statistics everywhere** (CPU only): bootstrap CIs and McNemar for every
   selector vs majority, per model. Cheap and removes "is 52.4 vs 48.6 real?" questions.
2. **DLM self-score** (`dlm.selfscore`): rank candidates by the diffusion model's own
   masked-token likelihood / ELBO. The natural competitor to an external AR scorer and the
   obvious gap a grader will ask about. One scoring pass per model, no new generation.
3. **Verifier / reward-model selection as the SOTA comparison**: score candidates with an
   off-the-shelf math process or outcome reward model. Find and verify a published
   diffusion-LM test-time selection method first (primary source, per
   `docs/verification-standards.md`) and reproduce its selector on our cached candidates.
4. **More headroom models**: `diffusionfamily/diffullama-gsm` (released GSM8K adapter)
   gives DiffuLLaMA real headroom; DiffuCoder-7B or Dream-instruct add more non-floor
   points. Each extra model is one more point on the CKA correlation.
5. **Fix the correlation's power problem** differently: vary preservation within one model
   (adaptation checkpoints, or the step-budget arm 16/32/64/128 from idea 1.1) instead of
   relying on 4 different models.
6. **Close fidelity gaps** where cheap: 8-shot for Dream, 128 steps on a 100-question slice
   to measure how much the 64-step cut costs.
7. **Second dataset** (e.g. MATH-500 or SVAMP) so the conclusion is not GSM8K-only.
8. **Novelty check (I-6)**: search 2025-26 work on reranking/self-consistency for
   diffusion LMs before claiming anything new.

## Process lessons (operations)

- **Never let a GPU server sit idle holding shared GPUs.** vLLM held Blackwell GPU 1
  (81 GB, 0% use) for ~8 h while the Arbor launch waited on a human. Start the server in
  the same step as its consumer; if blocked more than ~15 min, stop it (restart ~3 min).
- vLLM for Arbor needs `--max-model-len 131072`: Arbor asks for 16k output tokens per
  call and the 32k limit failed on the 4th turn.
- `--tensor-parallel-size 2` hangs on Blackwell (PCIe P2P reports OK but the custom
  all-reduce never completes). Use `--data-parallel-size 2`.
- Arbor review mode has two gates: idea proposal (free text *replaces* the hypothesis)
  and pre-execution review (free text is a note). Depth-1 nodes are directions, depth-2
  nodes are the concrete experiments, so a depth-2 child that restates its parent is
  not a duplicate. Iteration 1 skipped node 1's child (plain majority) by mistake;
  majority is still measured by `baselines.py`.
- Answer review prompts promptly; each unanswered one costs the 15-min timeout.
- Plan the GPU window: generation and scoring took ~1.5 days on two shared GPUs with
  display-watchdog crashes. Start generation earlier, and budget the Arbor run as a
  separate GPU block right after scoring, not a day later.
- Prefetch every model before the campaign; the lab network intercepts TLS, so downloads
  go through the laptop relay and are slow.
