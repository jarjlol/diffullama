# rerank_task — choose the best of N diffusion-LM samples (GSM8K)

An Arbor task pack. Arbor reads this README at intake.

## Task

Several diffusion language models (DiffuGPT-S/M, DiffuLLaMA, Dream-7B, LLaDA-8B)
each answered every GSM8K question **N = 10 times** (4-shot prompt). Those samples
are cached in `data/<split>/items.jsonl`. Each one comes with precomputed features,
such as its length and its negative log-likelihood under small autoregressive (AR)
scorer LMs.

`solution.py` defines `select(item) -> int`. It picks which of an item's candidates
to submit. No model runs during evaluation: the loop is CPU-only and takes seconds.

## Metric

`bash eval.sh dev` (iterate) and `bash eval.sh test` (held-out, gates merges) each
print one `score: <float>` line. The score is **macro-averaged accuracy over models,
and higher is better**. Range is 0–1.

The ceiling is not 1. It is the per-model oracle (hit@N: any candidate correct),
and `eval.sh` prints that oracle for each model. The script also warns if a
selector lands within 1% of the oracle, as that usually means gold has leaked.

**Starting point:** the seed `solution.py` picks the lowest *summed* GPT-2 NLL,
which is the idea as originally specified. Reference selectors: run
`python3 baselines.py --split dev`. It lists first sample, random, majority vote
(self-consistency), summed and mean NLL per scorer, and the oracle.

**Goal:** beat majority vote (self-consistency) on held-out test, the strongest
training-free baseline, and close as much of the gap to the oracle as possible.

## Editable / off-limits

- **Editable:** `solution.py` only.
- **Off-limits:** `task.py`, `eval.py`, `eval.sh`, `baselines.py`, and everything
  in `data/`. Do not read `data/*/gold.jsonl` from a solution.
  - `eval.py` enforces this. `solution.py` may import only `math re collections
    statistics itertools functools heapq bisect typing`, and may not use `open`,
    `exec`, `eval`, `__import__` or dunder attributes. A violation scores 0.0.
  - `select()` must return a valid index for every item, or the score is 0.0.

## Item format

```json
{"uid": "gsm8k-test-0001", "model": "dream-base", "question": "...",
 "candidates": [
   {"text": " ... #### 18", "answer": "18",
    "features": {"len_chars": 212, "ar.gpt2.nll_sum": 41.2, "ar.gpt2.n_tokens": 63,
                 "ar.qwen0.5b.nll_sum": 30.9, "ar.qwen0.5b.n_tokens": 58,
                 "ar.tinyllama.nll_sum": 35.0, "ar.tinyllama.n_tokens": 60}}]}
```

- `answer` is the number extracted by `task.extract_answer`. It can be null.
- The NLL features are computed on the trimmed candidate. They are conditioned on
  `"Question: <q>\nAnswer:"` only, without the few-shot exemplars.
- The set of features may vary by model. Treat a missing key as unavailable.

## Data

- `dev`: 500 questions sampled (seed 0) from the GSM8K **train** file.
- `test`: the 1319-question official GSM8K **test** set.

The two splits are disjoint, and the few-shot exemplars come from train, outside
dev. Full details are in `PROVENANCE.md`.
