# 04 — Can idea 1.1, as codified, answer its own question?

Research problem P1 asks two things. (a) Does how much of its AR backbone a diffusion LM preserves
predict how much accuracy a training-free AR reranker recovers? (b) Is that recovery more efficient
than adding denoising steps? The codification answers neither cleanly yet. Below is each threat,
how bad it is, and the cheapest fix.

## 1. The correlation has almost no data points — and may have none

Models with an AR base, so with a CKA score: DiffuGPT-S, DiffuGPT-M, DiffuLLaMA, Dream-Base. That
is n = 4, so Spearman's ρ can take only a handful of values, and the smallest exact p is 1/24 ≈ 0.04
for a perfect ranking. Worse:

- The anchor's GSM8K numbers for DiffuGPT and DiffuLLaMA are **after fine-tuning** (Table 1, ∗).
  LLaMA2-7B itself scores 13.1 at 4-shot (LLaDA Table 1). GPT-2-sized base models are typically close
  to 0% on few-shot GSM8K. **Expectation, not measured:** DiffuGPT-S/M first ≈ oracle ≈ 0.
- `baselines.py:80` sets headroom to `None` when oracle = first, and `analyze.py:72` needs ≥ 3 models.
  Two near-zero DiffuGPTs therefore leave **n = 2 and no correlation at all.**
- CKA values from different architectures (d = 768, 1024, 4096, 3584) are not on a common scale,
  especially with the biased estimator (`03` C-10). Even with n = 4, the ranking partly reflects width.

**Fixes, cheapest first.**
1. Add released diffusion LMs that have an AR base (all ungated, verified on HF 2026-11-02):

   | Checkpoint | AR base |
   |---|---|
   | `diffusionfamily/diffullama-gsm` | Llama-2-7B, through DiffuLLaMA plus a GSM8K LoRA |
   | `Dream-org/Dream-v0-Instruct-7B` | Qwen2.5-7B-Instruct, or Qwen2.5-7B |
   | `apple/DiffuCoder-7B-Base` | Qwen2.5-Coder-7B, per its card |

   That gives n = 7 with an AR base, a minimum p of 1/5040, two within-family pairs
   (DiffuLLaMA vs -gsm, Dream base vs instruct), and much more GSM8K headroom.
2. Make the claim testable *within* a model. Compute a per-question preservation score: CKA or
   cosine between DLM and base hidden states on that question's prompt. Then test whether it predicts
   per-question rerank success (first wrong, AR pick right). n = 500 per model instead of n = 4.
   It costs one forward pass per question per model. This is the most interesting upgrade scientifically.
3. Report the across-model ρ as descriptive, as `analyze.py` already says, and lead with the
   per-model accuracy table.

## 2. Headroom depends on the sampler, and the sampler differs per model

Headroom is (selector − first) / (oracle − first). Oracle@10 rises with sample diversity; the first
sample's accuracy falls with temperature. Current settings: DiffuGPT 0.95, DiffuLLaMA 0.9, LLaDA 0.7,
**Dream 0.2**. Dream at 0.2 may produce near-identical candidates, so it would get low oracle and low
headroom regardless of CKA. Across models, "preservation predicts headroom" is then confounded by
"temperature predicts headroom".

**Fixes.** At minimum, record answer diversity per model: distinct answers per question and the
entropy of the answer distribution, which the gate already partly computes. Use it as a covariate,
and say so. Better: on the dev subset, calibrate each model's temperature to a common diversity
target, or run two temperatures per model, before the test run. The second option costs one extra
dev run per model.

## 3. Half of the research question was dropped

P1(b) compares reranking with "simply increasing the denoising-step budget". The implementation uses
one step budget per model (now 64 for all), so (b) cannot be answered. Since every model is now at
64 steps, the missing arm is cheap. Add **K = 16** on the test subset: 4× fewer forward passes, about
+25% of the 64-step generation time. That allows these comparisons:
- ΔQ_rerank at K = 64 against ΔQ_steps from 16 to 64.
- Rerank at 16 against first-sample at 64: can selection buy back steps?
- Wall-clock per question, now recorded (`gen_seconds`), to put both on one cost axis.

## 4. Selection on the test set

Arbor gates merges on test, up to 12 times (`03` C-7). The selector Arbor finally keeps is the best
of many on test, so its test score is biased upward. Fix: iterate on half of dev, gate on the other
half, and touch test once. Report paired statistics against majority vote: McNemar on per-question
correctness, and a bootstrap 95% CI per model. Without them "beats majority vote" at n = 500 is not
a claim.

## 5. "first" is not the model's real single-answer accuracy

The first candidate is a *temperature sample*. Published numbers use standard (near-greedy)
decoding. A temperature sample understates single-answer accuracy, which inflates both headroom and
every selector's "gain". Fix: one extra low-temperature or greedy decode per question as the
single-answer reference, at about +10% generation. Or at least compare with the published numbers,
as `ITERATION_1.md` already does.

## 6. Other confounds to state, not fix

- Shot count: DiffuGPT-S is 1-shot and everything else 4-shot. Dream's published GSM8K is 8-shot.
- LLaDA runs fully non-autoregressive (one 256-token block) at 64 steps; its paper uses other settings.
- Arbor's `select(item)` sees `item["model"]`, so it can learn per-model rules on dev. That is
  legitimate, but say so, and check that the final rule is not a lookup table.
- GSM8K contamination in Qwen2.5-based Dream is plausible. It affects absolute accuracy, not the selection comparison.

## 7. Write the predictions down before the results arrive

Cheap and it makes iteration 1 look like research, not a fishing trip. Suggested pre-registration,
to be committed before the first test-set table:

1. Summed AR NLL (the seed) is **worse** than majority vote on every model, because it is length-biased.
2. Mean (per-token) AR NLL comes closer to majority vote but does not beat it on 7B models.
3. The best Arbor selector combines agreement with AR likelihood and beats majority vote by a small
   margin on at least two models.
4. DiffuGPT-S/M have near-zero oracle on few-shot GSM8K.
5. Across models, CKA and headroom are not significantly related at n ≤ 4.

If (1)–(3) hold, the defensible headline is the per-model result. Small training-free AR scorers
help or fail to help self-consistency on diffusion LMs, set against 2609.35472's trained ORM
reranker (`05`). The CKA link is an exploratory secondary result.
