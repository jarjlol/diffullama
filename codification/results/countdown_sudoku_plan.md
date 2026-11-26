# Countdown / Sudoku feasibility plan (issue #82)

Status: **generators, exact checkers, prompts and self-test are ready on CPU. No GPU
run has started** — the GPUs are still on GSM-Symbolic. This file is the frozen plan
for the 20-puzzle oracle gate.

Files:

- `codification/pipeline/countdown_sudoku.py` — seeded generators, exact checkers,
  prompt templates, normalisers, `--selftest`, `--make`.
- `codification/pipeline/countdown_sudoku_gen.py` — GPU entry point; N=10, K=64,
  256 tokens, one model per process. Reuses `generate_candidates.BACKENDS`, like
  `gsm_symbolic_gen.py`.
- `codification/pipeline/work/countdown_sudoku/{countdown,sudoku}.jsonl` — generated
  by `--make` (git-ignored; regenerate from seed 0).

## Generators and checkers

**Countdown.** 100 puzzles, seed 0, uids `countdown-0000..0099`. Each puzzle has 4–6
distinct integers from 1–25 and a target in 100–999. A puzzle is built by folding the
numbers pairwise with `+ - * /`, keeping every intermediate a positive integer, so
every puzzle has a witness expression that uses each number exactly once. The witness
is stored as `solution` for auditing and is never put in the prompt.

Exact checker `check_countdown`: a recursive-descent parser over `+ - * /`, parentheses
and unary signs (`×`, `÷`, `−`, `\times`, `\div` and one level of `\frac` are
normalised first); values are `Fraction`s, so equality is exact. It rejects the answer
unless (a) the multiset of integer literals equals the puzzle's numbers, and (b) the
value equals the target. A trailing `= N` is allowed only when `N` agrees with the
expression.

100-puzzle audit (seed 0): all 100 witnesses accepted; number counts 4:43, 5:23, 6:34;
targets 102–962.

**Sudoku.** 100 puzzles, seed 0, uids `sudoku-0000..0099`. 4x4 with 2x2 boxes. A full
grid is produced by randomised backtracking, then cells are removed greedily while the
solution remains unique, stopping at 6 givens. 6 givens is a deliberate floor: fully
minimal carving gave 4 givens for 67/100, which would make the oracle gate needlessly
harsh. 100/100 puzzles have exactly 6 givens and exactly one solution (verified by
counting up to two solutions).

Exact checker `check_sudoku`: parses four lines of four digits (spaces, `|`, markdown
tables and row labels are tolerated), requires every row, column and 2x2 box to be a
permutation of 1–4, and requires every given cell to be unchanged.

Both checkers are pure functions and are also used by `countdown_sudoku_gen.py` to
grade each candidate at generation time; there is no second grading path.

## Prompt templates and normalisers

One-shot prompt, final answer only. The exemplars are fixed and are not in the 100
puzzles.

- Countdown: `Numbers: 8 6 4 3`, `Target: 30`, `Expression: (8 + 6 - 4) * 3`, then the
  real puzzle and `Expression:`. The model is told to use every number exactly once.
- Sudoku: a filled 4x4 example, then the real puzzle as four lines of digits with `0`
  for blanks and `Grid:`. The model is told to answer with four space-separated digits
  per line.

`normalise_countdown` returns the last arithmetic span that parses (syntax only; it
does not see the puzzle). `normalise_sudoku` returns the last readable 4x4 grid. The
generation record keeps `parsed` (right shape) separate from `ok` (exactly correct), so
a zero caused by output format is visible instead of being silently counted as a wrong
answer.

## Self-test

```
$ python3 codification/pipeline/countdown_sudoku.py --selftest
PASS  countdown=5 (targets 142-885, numbers 5-6)  sudoku=5 (givens 6-6)
```

The test generates 5+5 puzzles, validates every witness/solution with the exact
checkers, re-solves every Sudoku from the givens and asserts uniqueness, and checks
that reuse of a number, a wrong target, a swapped grid row and an unfilled grid are all
rejected. Seed-to-seed determinism is checked separately (`--make --seed 7` twice gives
byte-identical files).

Sample puzzle (countdown-0000): numbers `14 20 15 13`, target `412`, witness
`(20 + ((15 + 13) * 14))`.

Sample puzzle (sudoku-0000):

```
0 0 3 1
0 0 0 0
0 2 0 0
0 3 1 2
```

## 20-puzzle oracle gate

Protocol, identical to the GSM8K / GSM-Symbolic generation settings: N=10 candidates,
K=64 denoising steps, 256 new tokens, temperature/top-p from `config.json`. 20 puzzles
per model: **10 Countdown + 10 Sudoku** (`--limit 10` applies per task). The 20-puzzle
budget is the gate, so it is split across the two tasks; the per-task breakdown is
reported.

Expected cost, using the measured GSM8K rates in `DECISION_RERANK_RESULTS.md` (Dream
36.8 s/question, LLaDA 55.0 s/question at N=10, 256 tokens). The Countdown/Sudoku
prompts are much shorter than the 4-shot GSM8K prompt, so these are conservative
upper bounds.

| arm | puzzles | Dream | LLaDA |
|---|---:|---:|---:|
| Countdown | 10 | 368 s (6.1 min) | 550 s (9.2 min) |
| Sudoku | 10 | 368 s (6.1 min) | 550 s (9.2 min) |
| **total** | **20** | **736 s (12.3 min)** | **1100 s (18.3 min)** |

Total generation is **30.6 GPU-minutes**. Add ~1.5 min (Dream-7B) and ~2 min (LLaDA-8B)
for model load: ~34 min wall clock if the two models share one GPU sequentially, or
~21 min if they run on two GPUs in parallel. This is a ceiling; no reranking cost is
included because the gate is oracle-only.

### Exact commands

Run on the GPU box after GSM-Symbolic releases the GPUs:

```bash
cd /home/antpc/codify/diffullama
git checkout iter2/decision-rerank && git pull
cd codification/pipeline
export HF_HOME=/path/to/hf-cache          # same cache as the GSM8K runs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# CPU: deterministic puzzles (100 each; the gate consumes the first 10 of each)
python3 countdown_sudoku.py --make --n 100 --seed 0

# 10 Countdown + 10 Sudoku per model; resumable by uid
CUDA_VISIBLE_DEVICES=0 ~/venvs/dream/bin/python countdown_sudoku_gen.py --model dream-base --limit 10
CUDA_VISIBLE_DEVICES=1 ~/venvs/llada/bin/python countdown_sudoku_gen.py --model llada-base --limit 10
```

Each command prints `oracle pass@10 = k/n` and `right-shaped = parsed/total`. To
re-print the table without regenerating, rerun the same command; completed uids are
skipped and the summary is computed from the output file. Outputs:
`work/countdown_sudoku/candidates/<model>/<task>.jsonl` (git-ignored).

## Go / no-go rule (pre-registered)

Grade a puzzle as passed if **at least one of its 10 candidates is exactly correct**
under the checker above. Report `pass@10` per task and combined.

- **0/20 combined → stop.** At K=64 the models produce no exactly checkable answer, so
  there is no oracle headroom to rerank; report 0/20 and do not spend the 100-puzzle
  budget.
- **One task 0/10, the other > 0 → drop the dead task** and continue with the live one;
  a task with no oracle has nothing to select.
- **Any task > 0 → go** on that task: 100 puzzles, same protocol, then the normal
  selectors (first, majority, AR, listwise, pointwise, oracle) and the order control.

No prompt tuning after seeing the gate. If the right-shaped rate is near zero while
the oracle is zero, that is still a stop under this rule; a prompt variant would be a
new, separately labelled arm, not a rescue of this one.

## If the gate passes: full run cost

100 Countdown + 100 Sudoku per model is 200 question-runs per model: Dream
200 × 36.8 s = 2.05 h, LLaDA 200 × 55.0 s = 3.06 h, **5.1 GPU-hours** of generation
plus model load. At 100 total per model (50+50) it is 2.6 h. No new dependencies are
needed for generation; the decision-model prompts can reuse the typed-choice schema in
`openrouter_decisions.py`, with a Countdown/Sudoku request builder analogous to
`make_openrouter_prompts.py`.
