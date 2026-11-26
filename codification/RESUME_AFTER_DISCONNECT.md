# Resume after the SSH/internet disconnect

**State at disconnect (~05:35 IST, 26 Nov).** GSM-Symbolic generation is running
detached on `blackwell` and does not need SSH:

- Dream: GPU 1, `~/gs_sym_dream.log`, output
  `work/gsm_symbolic/candidates/dream-base/test.jsonl`
- LLaDA: GPU 0, `~/gs_sym_llada.log`, output
  `work/gsm_symbolic/candidates/llada-base/test.jsonl`
- 250 questions each (125 P1 + 125 P2), N=10, K=64, 4-shot, 256 tokens.
  ETA ~08:00 (Dream) and ~09:00 (LLaDA) IST. Resume-safe by uid.

**First command when back**

```sh
dssh blackwell 'cd ~/codify/diffullama/codification/pipeline && \
  wc -l work/gsm_symbolic/candidates/*/test.jsonl; \
  tail -2 ~/gs_sym_dream.log; tail -2 ~/gs_sym_llada.log'
```

Then, when both files reach 250:

1. Build the listwise prompts (script: `codification/pipeline/gsm_symbolic_prompts.py`,
   being written by the GSM-Symbolic subagent) and POST to ada.
2. On ada: `source ~/openrouter_env.sh`, run
   `python3 ~/openrouter_decisions.py --in ... --out ... --workers 12`.
3. Convert features and evaluate (first, majority, AR cut, listwise, pointwise,
   oracle; cluster bootstrap; per-model McNemar).
4. Post the table on issue #79 and the sweeps on #83.

**Done while offline (committed locally, push when back)**

- K=256 step sensitivity (#80), commit `532d136`: listwise 87.6% Dream / 83.9%
  LLaDA (macro 85.7) vs majority 81.7; +4.0 macro, cluster CI [+0.9, +7.1],
  p = 0.010. Difference-in-differences vs K=64 on the same 377 questions is
  +0.5 (CI −4.0 to +5.1, p = 0.82), so the advantage does **not** shrink
  measurably at 256 steps; majority catches up in absolute terms.
- N sweep on 100 K=256 questions/model: gains +6.5, +2.0, +4.0, +2.5, +3.5
  points for N=2..10; only N=2's CI excludes zero at this n. Not a list-length
  artifact.
- **Open follow-up:** 94% (Dream) / 76% (LLaDA) of the K=256 cut texts still
  carry same-line junk after `#### N`, because 256-step models continue without
  a newline. AR features were cut identically, so that comparison is fair, but
  the decision prompt is noisier than at K=64. Run a clean-cut robustness check
  on 100 K=256 questions: truncate right after the numeric answer, re-score
  listwise, compare with `k256_decision.json`. Do not overwrite the original.

**Paused, needs the connection/API**

- GSM-Symbolic scoring (the main run is still generating on blackwell).
- HumanEval recognition probe (#81).
- HumanEval+ re-execution (#84).
- Countdown/Sudoku generator is **done** (commit `ad57bf6`, #82); only the
  20-puzzle GPU gate waits for the cards.

**Does not need SSH**

- Paper rewrite #77 and report rewrite #78 (GitHub only).
- The methods document `codification/DECISION_RERANK_METHODS.md` is committed.

**OpenRouter key** is in `~/openrouter_env.sh` on ada (mode 600), not in git.
