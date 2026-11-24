# Iteration-2 evidence (Blackwell copies 21 Nov 2026, 18:20 IST; git copies 23 Nov)

Raw outputs, copied without edits: the first four rows from the `antpc` workstation, the last two from `iter2/aaditya` in git.

Note: the `dlm.selfscore` row in `cvt_llada_test.txt` (0.012) covers only 16/500 questions, and the file marks it INCOMPLETE. The report does not use it. LLaDA's self-score numbers in the report come from `self_llada_gate.txt` (200/200).

| File | Source on Blackwell |
|---|---|
| `step_arm.md` | `~/codify/worker/codification/results/iter2/step_arm.md` |
| `cvt_dream_test.txt` | `~/codify/worker/codification/results/iter2/cpu/cvt_dream_test.txt` |
| `cvt_llada_test.txt` | `~/codify/worker/codification/results/iter2/cpu/cvt_llada_test.txt` |
| `self_dream_gate.txt` | `~/codify/worker/codification/results/iter2/cpu/self_dream_gate.txt` |
| `k256_table.txt` | `~/codify/worker/codification/results/iter2/cpu/k256_table_all.txt` (analysis on all 185/192 rows) |
| `self_llada_gate.txt` | `codification/results/iter2/cpu/self_llada_gate.txt` on `iter2/aaditya` @ `4f7c81a` (added 23 Nov) |
| `pq_analyze.txt` | `codification/results/iter2/cpu/pq_analyze.txt` on `iter2/aaditya` @ `4f7c81a` (added 23 Nov) |

SHA-256: `19f5725e…` (step arm), `97ec796a…` (Dream cut), `703de3db…` (LLaDA cut), `2bc62e9c…` (Dream self-score), `a3d5d530…` (LLaDA self-score), `c8a9b25b…` (per-question preservation).

These are small text outputs only. Candidates, features, model weights and gold labels are not included.
