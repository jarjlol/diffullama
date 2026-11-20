# Operator fallback audited library

T1 support, 21 November. No experiment below is claimed to work. Source: the task
schema in pipeline/assemble.py, existing iteration1 experiments and BORROWED_IDEAS_REVIEW.md.
All features named below must be present and finite in the private task before use.

| ID | Direction / change from iteration1 | Type / scale / risk | Features | Audit / expected test |
|---|---|---|---|---|
| S01 | Use DLM self-score only to break a top-vote tie. | PARAM / micro / LOW | answer, dlm.selfscore (needs A2) | CLEARED R1–R8; tests native confidence with limited deviations. |
| S02 | Native-score weighted votes, one dev-fixed temperature. | PARAM / micro / MEDIUM | answer, dlm.selfscore (needs A2) | CLEARED; unlike3.1 this uses native score, not AR likelihood. |
| S03 | Cut qwen7b per-token tie break. | PARAM / micro / LOW | ar.qwen7b-cut.nll_sum, ar.qwen7b-cut.n_tokens (needs A2) | CLEARED; node2.1 used junk-confounded trim scores, so this is a distinct falsification. |
| S04 | Cut AR answer-group mean score rather than best candidate. | CODE / meso / MEDIUM | answer, ar.qwen7b-cut.* (needs A2) | CLEARED; avoids selecting one lucky likelihood outlier. |
| S05 | Sum ranks of native score and cut AR within vote-tied answers. | ALGO / meso / MEDIUM | answer, dlm.selfscore, ar.qwen7b-cut.* (needs A2) | CLEARED; scale-free fusion, weights fixed on dev. |
| S06 | Use native score only when vote gap<=1; otherwise frozen bar. | CODE / meso / LOW | answer, dlm.selfscore (needs A2) | CLEARED; targets measured close-plurality failures. |
| S07 | Robust median native confidence per answer group. | CODE / meso / MEDIUM | answer, dlm.selfscore (needs A2) | CLEARED; limits one overconfident sample. |
| S08 | Native-score discrepancy abstention: keep bar when native and cut AR disagree. | ALGO / meso / MEDIUM | answer, dlm.selfscore, ar.qwen7b-cut.* (needs A2) | CLEARED; evaluate both fixed and broken questions. |
| S09 | Trim/cut AR discrepancy as a reliability indicator, dev-fixed threshold. | CODE / meso / HIGH | ar.qwen7b.*, ar.qwen7b-cut.* (needs A2) | CLEARED; diagnostic only until held-out gains and ablation. |
| S10 | Native-confidence override of a confident wrong plurality. | ALGO / macro / HIGH | answer, dlm.selfscore (needs A2) | CLEARED; structural alternative requiring strong gate evidence. |
| S11 | Earliest/shortest/calculation-count majority ties; count/ignore None. | PARAM / micro / LOW | answer, text | CLEARED baseline calibration ONLY; preregistered six variants, not a new method. |
| S12 | Cache answers by question across model blocks. | CODE / macro / HIGH | question, answer across calls | REJECTED R7; twice rejected in iteration1. |
| S13 | Read gold.jsonl or synthesize a corrected numeric answer. | CODE / macro / HIGH | forbidden gold/new output | REJECTED R2/R3/R5/R7. |
| S14 | Weight by cross-candidate verified-result consistency. | ALGO / meso / MEDIUM | text, answer | REJECTED (evidence): dev correct/wrong support4/11 Dream,8/15 LLaDA, EXPERIENCE.md. |
| S15 | Question-number derivability prior. | ALGO / meso / MEDIUM | question, answer | REJECTED (evidence):6/13 Dream,6/9 LLaDA, EXPERIENCE.md. |
| S16 | Extraction-commitment tiers. | PARAM / micro / MEDIUM | text, answer | REJECTED (evidence): at most+1 net item per split in old audit; below present >=3 EACH requirement. |
| S17 | Repeat plain majority or unmodified arithmetic verification4.1 as a new method. | CODE / meso / LOW | text, answer | REJECTED (duplicate); use as controls/ablation only. |

Opening priority: S01, S06, S03. They target tie/close-plurality failures, add only
cached CPU logic, and isolate one new signal at a time. S05/S07 follow if that signal
has evidence. After three PARAM proposals require a CODE/ALGO proposal. Do not turn
the limited iteration1 feature audit into a ceiling claim for new self-score/cut features.
