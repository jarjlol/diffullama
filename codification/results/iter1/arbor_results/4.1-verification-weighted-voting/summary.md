# Idea 4.1 — verification-weighted voting

Weighted self-consistency voting: each candidate contributes weight 1.0 iff every
explicit calculation in its trimmed text (GSM8K `<<a op b = c>>` and plain
`a op b = c`, op ∈ + - * / x ×, trim = text before `\n\s*(?:Question|Q)\s*:`)
re-evaluates within 1e-6 (div-by-0 / unparsable pairs skipped), and it contains
>=1 checkable calculation; else 0.5. Answer score = sum of weights of candidates
with that extracted answer; ties -> earliest candidate index; all None -> 0.

## Result (B_dev, `bash eval.sh dev`)

**score: 0.2960** (trunk baseline 0.1480; plain-majority reference 0.2940)

| model      | n   | majority | weighted | Δ      | changed | fixed | broken | cands w/ ≥1 checked calc |
|------------|-----|----------|----------|--------|---------|-------|--------|--------------------------|
| diffugpt-m | 200 | 0.0100   | 0.0100   | +0.000 | 1       | 0     | 0      | 5.40%                    |
| diffugpt-s | 200 | 0.0150   | 0.0150   | +0.000 | 0       | 0     | 0      | 0.10%                    |
| diffullama | 200 | 0.0000   | 0.0000   | +0.000 | 9       | 0     | 0      | 53.70%                   |
| dream-base | 200 | 0.7000   | 0.7150   | +0.015 | 16      | 3     | 0      | 63.60%                   |
| llada-base | 200 | 0.7450   | 0.7400   | -0.005 | 6       | 1     | 2      | 68.50%                   |
| **MACRO**  | 1000| **0.2940** | **0.2960** | **+0.002** | 32 | 4 | 2 | — |

## Observations

- Coordinator predictions hit almost exactly: dream-base 0.715 (+3 items: 3 fixed,
  0 broken), llada-base 0.740 (-1 item net: 1 fixed, 2 broken), macro 0.296.
- Observable B_dev macro >= 0.295 met (0.2960); dream 0.715 in [0.710, 0.720].
- Mechanism is dead on the 3 floor models (diffugpt-s/diffugpt-m/diffullama) —
  their candidates rarely contain checkable explicit calculations (0.1–5.4%)
  or correct answers at all, so weighting changes nothing that matters.
- On dream, every flip was a fix (3/3); on llada, flips were net negative
  (1 fixed, 2 broken), matching the opposite pattern found in dev diagnostics
  (llada correct-cand all-pass 4/26 vs plurality 8/26).
- Deterministic, stdlib-only (re), no module state; weight fixed at 0.5 per
  review note (no dev sweep).
