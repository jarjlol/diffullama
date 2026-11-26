# Decision-model reranking: methods, rates and the changes made after review

This is the methodological record for the post-submission experiment in PR #76.
It is the document to read before running any follow-up. It states what was run,
what was measured, what the #76 reviews changed, and what remains unmeasured.
Every number here is reproduced by a file in `codification/results/` or by a
command in this file.

## 1. Question and design

**Question.** Can a small decision model pick the correct answer from a
diffusion language model's own candidates, better than majority vote and better
than an autoregressive likelihood scorer?

**Why it is not a reasoning claim.** The decision model is shown the question
and the candidate texts and returns a choice. It may be selecting, verifying, or
recognising the answer. The controls in §5 separate those as far as the API
allows; they do not make it a reasoning claim.

**Tasks.**
- GSM8K (Cobbe et al., 2021): 500 test questions per model, exact numeric answer.
- HumanEval (Chen et al., 2021): 144 test problems, execution-based pass@1.

**Models.** Dream-7B (Ye et al., 2025) and LLaDA-8B (Nie et al., 2025). Five
models exist for the iteration-1 table (DiffuGPT-S/M, DiffuLLaMA, Dream, LLaDA);
the decision arm uses the two with a usable oracle.

**Candidates.** N = 10 per question, 64 denoising steps, 4-shot prompt, 256 new
tokens, temperature 0.7. HumanEval uses 512 tokens. These are the iteration-2
settings, not retuned for this experiment.

**Splits.**
- GSM8K: 200 dev for any threshold or wording choice; 500 test scored once.
- HumanEval: 20 dev; 144 test. The split file is
  `codification/results/humaneval_splits.json`.

## 2. Selectors

| selector | what it does | cost |
|---|---|---|
| `first` | candidate A | free |
| `majority` | most common extracted final answer | free |
| `ar.qwen7b.sum` / `.mean` | AR likelihood of the cut text, summed or per token | GPU forward pass |
| `example-filter` | run the candidate on the docstring examples, keep the first that passes | CPU |
| `mbr-exec` | run candidates on docstring inputs, take the largest output cluster | CPU |
| `decision.listwise` | one API call over all ten candidates, returns a probability per option | ~1,544 input tokens |
| `decision.pointwise` | ten API calls, one per candidate, binary correct/incorrect | ~233 input tokens × 10 |
| `qwen.choice` | cached Qwen2.5-7B prompted for a letter (exploratory) | GPU forward pass |
| `oracle` | any candidate passes | free upper bound |

**Cut text.** Generation often continues past the final answer. `candidate_text`
cuts each candidate at the final-answer line before it is shown to the AR scorer
or the decision model. Without the cut, trailing prose distorts both. Cutting
raised Dream's AR accuracy by 13 points.

**Prompts.** The listwise prompt gives the problem and candidates A–J, and asks
which candidate reaches the correct final answer. The pointwise prompt gives the
problem and one candidate and asks whether its final answer agrees with the
correct answer. No chain of thought is requested. The exact prompt text and
SHA-256 hashes are in `codification/results/prompt_sha256.txt` and
`make_openrouter_prompts.py`.

**Decision model.** `perplexity/pplx-decider-flash-v1.3` through OpenRouter's
`/api/alpha/decisions`; the response records the served snapshot
`pplx-decider-flash-v1.3-20261112`. The local 9.6 GB weights were not used because
the download could not finish before the deadline. This is a declared deviation.

## 3. Measured generation and serving rates

| workload | rate | source |
|---|---:|---|
| Dream, HumanEval, N=10, 64 steps, 512 tokens | **45.6 s/problem** (median 50.2, p90 60.5, n=147) | `gen_seconds` in the candidate files |
| Dream, GSM8K, N=10, 256 tokens | **36.8 s/question** | `gen_seconds` |
| LLaDA, GSM8K, N=10, 256 tokens | **55.0 s/question** | `gen_seconds` |
| Dream, MBPP, N=10, 512 tokens | **46.4 s/problem** (n=20; stopped) | `gen_seconds` |

Decision serving (OpenRouter, 12–16 workers): 1,000 listwise GSM8K calls in
~50 s (~20/s); 10,000 pointwise calls in 213 s (~47/s); 1,470 HumanEval
pointwise calls in ~45 s. Listwise prompts average 1,544 tokens (median 1,504,
p95 2,281); pointwise 233.

**Cost.** The `:free` tier was $0. At the paid flash price ($0.01/1M input):
1,000 listwise calls ≈ $0.015; 10,000 pointwise calls ≈ $0.023. Majority is
free. Reranking adds no GPU generation time; it is a post-hoc pass.

## 4. Results (as of 2026-11-26)

| split | first | majority | AR cut | listwise | pointwise | oracle |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K Dream (n=500) | 40.4 | 72.4 | 48.8 | **77.0** | 74.2 | 87.8 |
| GSM8K LLaDA (n=500) | 48.6 | 73.2 | 52.2 | **79.0** | 80.8 | 89.2 |
| HumanEval Dream (n=144) | 10.4 | — | 5.6 | **40.3** | 38.2 | 40.3 |

The GSM8K listwise−majority delta is +5.2 points, question-clustered 95% CI
[3.0, 7.4], p < 1e-4; per-model exact McNemar p = 0.0095 (Dream) and 0.00011
(LLaDA). The old stacked p = 4.9e-6 is withdrawn. HumanEval: listwise selects a
passing candidate on all 58 solvable problems; the example filter is 27.8%
(fixed parser), MBR-exec 6.2%, first 10.4%.

## 5. Controls and what they mean

| control | result | interpretation |
|---|---|---|
| judge-alone (no reasoning text) | 42.8% (428/1,000) | reasoning text adds real signal; not sufficient alone to explain 78% |
| gold injection (115 items with no gold candidate) | 20.9% picked (24/115; chance 12.9%, mean 1/k over 8.46 options) | modest answer knowledge; not enough to explain the result |
| closed-book, two options | 92.5% (37/40) | the model recognises correct answers; the ten-way task is harder |
| candidate-order shuffle | 74.5% both orders; 67% same candidate | little position sensitivity; one seed, pilot |
| probability shuffle | 8.4% | **withdrawn**: equals uniform pick, not a position test |
| HumanEval example subset | 42 solvable with examples: filter 36/42, listwise 42/42 (p = 0.031); 16 without: 4/16 vs 16/16 (p = 4.9e-4) | the aggregate gap is concentrated where the filter has no examples |

## 6. What the #76 review changed

| review point | change in the new runs |
|---|---|
| stacked McNemar double-counts questions | question-clustered bootstrap; per-model exact tests; old macro p withdrawn |
| closed-book confound | judge-alone, gold injection, GSM-Symbolic, HumanEval recognition probe (#81) |
| HumanEval baseline too weak | fixed parser; MBR-exec; **HumanEval+** re-execution (#84) |
| HumanEval likely memorised | recognition probe (#81); HumanEval+ stronger tests (#84); fresh datasets (#79, #82) |
| TLS verification disabled | client verifies by default; `OPENROUTER_INSECURE=1` is an explicit lab opt-in only |
| undeclared deviations | declared in `DECISION_RERANK_RESULTS.md` and the PR body |
| token length contradiction | 30,720 from the model card; the 2,048 sentence is withdrawn |
| raw API responses not retained | client now records `usage`, `latency_s`, `api_model`, `api_id` per call |
| cost not reported | §3, and per-run cost from `usage` |
| parameter sweeps | #83: N=2–10, order ×3, K=256 subset, calibration |
| weak-vote threshold | #81: dev-tuned threshold, test reported once |
| anchor self-consistency not reproduced | cannot be reproduced with this checkpoint and dataset; stated as a limitation, not claimed |
| stronger generator at matched compute | not available in the window; stated as a limitation |
| LLaDA has no AR base | LLaDA is the from-scratch control; no preservation claim for it |
| open-ended decider-only baseline | impossible with a typed-choice endpoint; recognition probe is the substitute, labelled as such |

## 7. Reproduction

```sh
# Generation (GPU box, one model per process)
python3 generate_candidates.py --model dream-base --split test
python3 humaneval_gen.py --model dream-base

# Decision prompts and scoring
python3 make_openrouter_prompts.py --split test
python3 make_openrouter_humaneval.py
python3 openrouter_decisions.py --in prompts.jsonl --out results.jsonl --workers 12

# Features and evaluation
python3 openrouter_to_features.py --results results.jsonl --model dream-base
python3 decision_eval.py --models dream-base llada-base --split test
```

The OpenRouter client reads `$OPENROUTER_API_KEY`; the key is never written to
the output or the repo. On the lab host the firewall MITMs HTTPS, so
`OPENROUTER_INSECURE=1` is set explicitly for the run and documented; a shipped
client must use the system CA or `OPENROUTER_CA_BUNDLE`.

## 8. Limits

- The decision model may recognise answers; the controls bound this but do not
  eliminate it.
- HumanEval is public and likely contaminated; HumanEval+ is stronger but not
  fresh. GSM-Symbolic and Countdown/Sudoku are the fresh tests.
- One decision-model family and one snapshot; the endpoint can change.
- No open-ended decider-only baseline (the endpoint is typed-choice).
- No same-size open instruct judge (not cached; download would not finish).
- The LLaDA code arm collapsed at 64 steps, so HumanEval is Dream-only.
- MBPP is not viable at 64 steps; no MBPP claim is made.
- The 256-step GSM8K arm is a random subsample (n=185/192), not the full 500.
