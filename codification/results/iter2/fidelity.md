# Fidelity check (prompt A2 step 3), generated 2026-11-25

Command (run in `codification/pipeline/`; the runs wrote `work/fidelity/<config>.jsonl`):

```bash
python3 fidelity_check.py --report
```

## Raw output

```text
dream-sampler    n=244  greedy 0.824  | iter-1 first sample on same 244 Qs 0.422  | no answer 0  | median 16.5s/q  | published 77.2 (8-shot gsm8k_cot; Dream paper Table 1)
dream-published  n=100  greedy 0.800  | iter-1 first sample on same 100 Qs 0.410  | no answer 0  | median 17.0s/q  | published 77.2 (8-shot gsm8k_cot; Dream paper Table 1)
llada-published  n=199  greedy 0.729  | iter-1 first sample on same 199 Qs 0.497  | no answer 0  | median 17.9s/q  | published 70.3 (4-shot; LLaDA Table 1) / 70.0 (lm-eval, 256 setting; LLaDA EVAL.md)
```

Configs and their sources are in the docstring of `fidelity_check.py` (settings read from the authors' own evaluation scripts on 2026-11-20).

## What is [unverified] or not covered

- Nothing here is interpreted.
- 100 test questions per config, one greedy sample each: a standard error of about 4 points at 75% accuracy.
- The published numbers are the papers' own (see `PUBLISHED` in `fidelity_check.py`); they were not re-run.
- Dream's published prompt (`dream-published`) reproduces the lm-eval gsm8k_cot exemplars from the harness' task file as read on
  2026-11-20; a byte-exact match with the harness was not checked.
