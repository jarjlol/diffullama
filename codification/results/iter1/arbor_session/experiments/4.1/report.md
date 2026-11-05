# Experiment 4.1

**Hypothesis**: Mechanism: verification-weighted voting — for each candidate (on its trimmed text: everything before a line matching /\n\s*(?:Question|Q)\s*:/), parse explicit calculations with the regex "<<\s*(num)\s*([+\-*/])\s*(num)\s*=\s*(num)\s*>>" and "(num)\s*([+\-*/x×])\s*(num)\s*=\s*(num)" (num = -?[\d,]+(\.\d+)?; 'x'/'×' treated as *; divisions by 0 skipped, unparsable pairs skipped); re-evaluate each exactly and flag candidates having >=1 checked calculation with at least one mismatch (or none at all) as unverified. Weight = 1.0 if all checked calculations pass (and >=1 was checked), else 0.5. Answer score = sum of weights of its candidates; select the highest; ties -> earliest candidate index; all answers None -> 0. Uniform across all 5 models; deterministic, no state.
Hypothesis: when the majority-vote answer is wrong, the wrong plurality comes from samples containing an arithmetic slip that a deterministic checker sees and AR likelihood cannot. Coordinator dev diagnostics: in the 29 dream majority-wrong cases the correct candidate has all-passing calcs 9x vs 4x for the plurality (and last-calc==answer 11 vs 8); expected dream dev ~0.715 (+3 items). LLaDA shows the OPPOSITE pattern (correct all-pass 4/26 vs plurality 8/26) so expect llada dev ~0.740 (-1 item); net macro ~0.296 > 0.294 majority.
Observable: B_dev macro >= 0.295 with dream-base ~0.710-0.720; per-model report incl. floor models. If dream < 0.705 the mechanism is dead -> prune.
Conflicts: none — new evidence axis (in-item arithmetic consistency); orthogonal to the pruned AR-NLL axes (2, 3).
**Branch**: `coordinator/n4-1-mechanism-verification-weighted-6f5aa728`
**Attempt**: 1
**Status**: done
**Eval status**: scored (stop_reason=finished)
**Turns**: 25

---



Committed. Final report:

## Report — Idea 4.1: Verification-Weighted Voting

**Changes**: `solution.py` rewritten (the only file edited). Replaced the raw-NLL selector with verification-weighted voting exactly per spec:
- `_trim`: same trim as `task.trim` (split on `/\n\s*(?:Question|Q)\s*:/` case-insensitive, maxsplit 1, strip)
- `_CALC`: single alternation regex, annotated `<<a op b = c>>` (op ∈ `+ - * /`) first, then plain `a op b = c` (op ∈ `+ - * / x ×`); `num = -?[\d,]+(\.\d+)?`; non-overlapping `finditer`
- `_verified`: re-evaluates each match (commas stripped, div-by-0 and unparsable pairs skipped), pass iff `|recomputed − stated| < 1e-6`; candidate verified iff ≥1 checked and all pass
- `select`: weight 1.0/0.5 per candidate; score = sum of weights per extracted answer; ties → answer with earliest candidate index; all None → 0. Uniform, deterministic, stdlib-only (`re`), no state. Weight kept at 0.5, no dev sweep.

**Baseline vs Result** (B_dev, `bash eval.sh dev`):

| model | trunk (raw NLL) | majority ref | weighted (this) | vs majority | fixed / broken | cands w/ ≥1 checked calc |
|---|---|---|---|---|---|---|
| diffugpt-s | 0.015 | 0.015 | 0.015 | 0 | 0 / 0 | 0.10% |
| diffugpt-m | 0.000 | 0.010 | 0.010 | 0 | 0 / 0 | 5.40% |
| diffullama | 0.000 | 0.000 | 0.000 | 0 | 0 / 0 | 53.70% |
| dream-base | 0.140 | 0.700 | **0.715** | +0.015 | 3 / 0 (16 changed) | 63.60% |
| llada-base | 0.345 | 0.745 | **0.740** | −0.005 | 1 / 2 (6 changed) | 68.50% |
| **macro** | 0.148 | 0.294 | **0.2960** | +0.002 | 4 / 2 (32 changed) | — |

**Score**: 0.2960

**Observable check**: B_dev macro 0.296 ≥ 0.295 ✓; dream 0.715 ∈ [0.710, 0.720] ✓ → mechanism is live, not pruned.

**Insight**: Both coordinator predictions hit exactly — dream-base +3 items (every flip was a fix, zero breaks) and llada-base −1 net (1 fix, 2 breaks), confirming the opposite all-pass patterns found in dev diagnostics. The mechanism only fires where candidates contain explicit arithmetic: floor models (0.1–5.4% checked) are untouched, while dream/llada (64–69%) see real flips. The gain is small because weight 0.5 means unverified candidates still vote — slips only *dilute* the wrong plurality rather than veto it — which is exactly the intended conservative behavior.

**Implementation notes**: One spec subtlety — with the single alternation + finditer construction, the plain form can still match inside `<<a x b = c>>` (annotated op excludes `x`), which is the literal behavior the spec prescribes; verified with unit tests (trim, precedence, commas, div-by-0, x/×, negatives, multi-calc, post-Question calc exclusion). Results committed to `results/4.1-verification-weighted-voting/` (summary.md, dev.log).
