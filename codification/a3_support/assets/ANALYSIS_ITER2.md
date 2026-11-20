# N4 operator support
Selected on dev only; frozen before gate was loaded.
Winner: tie=earliest, ignore_none=True; SHA-256 651b5bd90bfc2db5e9327d11488cbd0a09898f6a0bf1033be7d7c9d43ec6d3f7

| Variant | Dream dev | LLaDA dev | Macro |
|---|---:|---:|---:|
| earliest, ignore_none=True | 0.7000 | 0.7450 | 0.7225 |
| earliest, ignore_none=False | 0.7000 | 0.7450 | 0.7225 |
| shortest, ignore_none=True | 0.6850 | 0.7550 | 0.7200 |
| shortest, ignore_none=False | 0.6850 | 0.7550 | 0.7200 |
| calculations, ignore_none=True | 0.6950 | 0.7350 | 0.7150 |
| calculations, ignore_none=False | 0.6950 | 0.7350 | 0.7150 |

Gate (selected bar only; no retuning):
{"dream-base": {"accuracy": 0.71, "correct": 142, "n": 200}, "llada-base": {"accuracy": 0.735, "correct": 147, "n": 200}, "macro": 0.7224999999999999}
Plain majority gate:
{"dream-base": {"accuracy": 0.71, "correct": 142, "n": 200}, "llada-base": {"accuracy": 0.735, "correct": 147, "n": 200}, "macro": 0.7224999999999999}

Compute: CPU-only tie-break logic; no extra inference passes. Arithmetic-pattern counts do not verify correctness.
Operator extraction review completed separately: a3_support/EXTRACTION_REVIEW.md (agent review,20 dev candidates; four flagged cases). Full N4 semantic taxonomy/leave-one-out analysis remains pending. No grader changed.

## Failure classes for A3
Heuristic dev-only labels; these are not independently validated semantic diagnoses.
- dream-base: close plurality (vote gap at most one), n=22; example uid gsm8k-train-0133. No gold supplied to Arbor.
- dream-base: confident wrong plurality (semantic cause unclassified), n=7; example uid gsm8k-train-1875. No gold supplied to Arbor.
- llada-base: confident wrong plurality (semantic cause unclassified), n=10; example uid gsm8k-train-0039. No gold supplied to Arbor.
- llada-base: close plurality (vote gap at most one), n=16; example uid gsm8k-train-1194. No gold supplied to Arbor.
