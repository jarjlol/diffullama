# Per-item receipts for the decision-model controls

These are the raw per-item outputs behind `codification/results/controls.json`.
Copied from the serving host on 2026-11-26; the API key is not present.

| file | items | what it is |
|---|---:|---|
| `gsm8k.judge.results.jsonl` | 957 | judge-alone: question + distinct candidate answers, no reasoning; 43 items with a single distinct answer are not prompted (42 correct) → **428/1000** |
| `gsm8k.gold.results.jsonl` | 115 | gold injection: items where no candidate had the gold; gold added as an option → **24/115 picked** |
| `dev.gsm8k.plain.results.jsonl` | 200 | dev listwise, original order |
| `dev.gsm8k.shuf.results.jsonl` | 200 | dev listwise, seeded candidate permutation |
| `humaneval_mbr_dream.jsonl` | 144 | MBR-exec choice per HumanEval problem (output-cluster baseline) |

Regenerate with `codification/pipeline/judge_controls.py` (judge/gold) and
`codification/pipeline/humaneval_mbr.py` (MBR); the scripts are committed and
the prompts are deterministic. Gold chance is the mean of 1/(number of options) over the 115 items: **12.9%** (8.46 options on average), not a rounded “~9%”. The pick rate is 20.9%, so the familiarity signal is modest.

These files are inputs to the paper's controls table. They are not a new
experiment and do not change any headline number.
