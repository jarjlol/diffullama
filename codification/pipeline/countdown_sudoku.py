"""Countdown and Sudoku: seeded generators, exact checkers, prompts, normalisers.

Both tasks are procedural, exactly checkable and fresh: nothing here is copied
from a benchmark. Pure stdlib.

    python3 countdown_sudoku.py --selftest
    python3 countdown_sudoku.py --make --n 100 --seed 0

--make writes work/countdown_sudoku/{countdown,sudoku}.jsonl. A record keeps
`solution` (a witness expression / the full grid) so the generator can be
audited; the prompt builders never read it.

Countdown rules used here: use each given number exactly once, with + - * /
and parentheses. Intermediate fractions are allowed by the checker (the
generator's own witness only uses positive integers). Sudoku is 4x4 with 2x2
boxes; puzzles are carved from a full grid and kept unique.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "work" / "countdown_sudoku"

# ------------------------------------------------------------------ countdown

COUNTDOWN_MIN_NUMS, COUNTDOWN_MAX_NUMS = 4, 6
COUNTDOWN_NUM_POOL = range(1, 26)          # distinct integers 1..25
COUNTDOWN_TARGET_MIN, COUNTDOWN_TARGET_MAX = 100, 999


def _countdown_fold(nums: list[int], rng: random.Random):
    """Fold `nums` pairwise with random ops, keeping positive integer values.

    Returns (value, expression) or None if this random fold cannot stay inside
    the rules. Every number is consumed exactly once.
    """
    vals = [Fraction(n) for n in nums]
    exprs = [str(n) for n in nums]
    while len(vals) > 1:
        i, j = rng.sample(range(len(vals)), 2)
        a, b, ea, eb = vals[i], vals[j], exprs[i], exprs[j]
        options = [
            (a + b, f"({ea} + {eb})"),
            (a * b, f"({ea} * {eb})"),
        ]
        if a > b:
            options.append((a - b, f"({ea} - {eb})"))
        elif b > a:
            options.append((b - a, f"({eb} - {ea})"))
        if b and a % b == 0:
            options.append((a / b, f"({ea} / {eb})"))
        if a and b % a == 0:
            options.append((b / a, f"({eb} / {ea})"))
        val, expr = rng.choice(options)
        if val <= 0 or val.denominator != 1 or val > COUNTDOWN_TARGET_MAX * 1000:
            return None                      # keep intermediate and final values sane
        for k in sorted((i, j), reverse=True):
            del vals[k], exprs[k]
        vals.append(val)
        exprs.append(expr)
    return vals[0], exprs[0]


def make_countdown(rng: random.Random) -> dict:
    for _ in range(5000):
        n = rng.choice(range(COUNTDOWN_MIN_NUMS, COUNTDOWN_MAX_NUMS + 1))
        nums = rng.sample(list(COUNTDOWN_NUM_POOL), n)
        got = _countdown_fold(nums, rng)
        if got is None:
            continue
        val, expr = got
        if val.denominator == 1 and COUNTDOWN_TARGET_MIN <= val <= COUNTDOWN_TARGET_MAX:
            return {"numbers": nums, "target": int(val), "solution": expr}
    raise RuntimeError("countdown generation failed; widen the target range")


def generate_countdown(n: int, seed: int = 0) -> list[dict]:
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}:countdown:{i}")
        p = make_countdown(rng)
        out.append({"uid": f"countdown-{i:04d}", "task": "countdown", **p})
    return out


class _ExprParser:
    """Recursive descent over + - * /, parentheses and unary signs."""

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def parse(self) -> tuple[Fraction, Counter]:
        val = self._expr()
        self._ws()
        if self.i != len(self.s):
            raise ValueError(f"unexpected {self.s[self.i]!r}")
        used: Counter = Counter()
        # Count number literals in a second pass; the grammar above is unambiguous.
        for m in re.finditer(r"\d+", self.s):
            used[int(m.group())] += 1
        return val, used

    def _ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _expr(self) -> Fraction:
        val = self._term()
        while True:
            self._ws()
            if self.i < len(self.s) and self.s[self.i] in "+-":
                op = self.s[self.i]; self.i += 1
                rhs = self._term()
                val = val + rhs if op == "+" else val - rhs
            else:
                return val

    def _term(self) -> Fraction:
        val = self._factor()
        while True:
            self._ws()
            if self.i < len(self.s) and self.s[self.i] in "*/":
                op = self.s[self.i]; self.i += 1
                rhs = self._factor()
                if op == "/" and rhs == 0:
                    raise ValueError("division by zero")
                val = val * rhs if op == "*" else val / rhs
            else:
                return val

    def _factor(self) -> Fraction:
        self._ws()
        if self.i >= len(self.s):
            raise ValueError("unexpected end")
        c = self.s[self.i]
        if c in "+-":
            self.i += 1
            v = self._factor()
            return v if c == "+" else -v
        if c == "(":
            self.i += 1
            v = self._expr()
            self._ws()
            if self.i >= len(self.s) or self.s[self.i] != ")":
                raise ValueError("missing ')'")
            self.i += 1
            return v
        if c.isdigit():
            j = self.i
            while j < len(self.s) and self.s[j].isdigit():
                j += 1
            v = Fraction(int(self.s[self.i:j]))
            self.i = j
            return v
        raise ValueError(f"unexpected {c!r}")


def clean_math(text: str) -> str:
    """Map the operators models actually emit onto the parser's ASCII set."""
    text = text.replace("×", "*").replace("÷", "/").replace("·", "*")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\\times|\\cdot", "*", text)
    text = re.sub(r"\\div", "/", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = text.replace("$", "")
    # \frac{a}{b} -> ((a)/(b)); one level, enough for short model answers.
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", text)
    return text


def parse_expression(text: str) -> tuple[Fraction, Counter]:
    """Parse a Countdown expression; return (value, Counter of integer literals).

    An optional trailing "= <integer>" is accepted, but only if it agrees with
    the expression. Anything outside the grammar is rejected.
    """
    s = clean_math(text).strip().rstrip(".")
    if "=" in s:
        parts = s.split("=")
        if len(parts) != 2:
            raise ValueError("more than one '='")
        left, right = parts[0], parts[1].strip()
        if not re.fullmatch(r"-?\d+", right):
            raise ValueError("right-hand side of '=' is not an integer")
        val, used = _ExprParser(left).parse()
        if val != int(right):
            raise ValueError(f"expression is {val}, not the written {right}")
        return val, used
    return _ExprParser(s).parse()


def check_countdown(expr: str | None, puzzle: dict) -> tuple[bool, str]:
    """Exact check: numbers used exactly once (as a multiset) and value == target."""
    if not expr:
        return False, "empty answer"
    try:
        val, used = parse_expression(expr)
    except (ValueError, ZeroDivisionError) as e:
        return False, f"parse error: {e}"
    want = Counter(puzzle["numbers"])
    if used != want:
        missing = want - used
        extra = used - want
        return False, f"number use differs (missing {dict(missing)}, extra {dict(extra)})"
    if val != puzzle["target"]:
        return False, f"value {val} != target {puzzle['target']}"
    return True, "ok"


_SPAN = re.compile(r"[0-9()+\-*/.][0-9()+\-*/. \t]*")


def normalise_countdown(text: str) -> str | None:
    """Last arithmetic span that parses; falls back to the last non-empty line.

    Deliberately syntax-only: it does not see the puzzle, so a wrong-but-valid
    expression is left for check_countdown to reject.
    """
    if not text:
        return None
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    body = clean_math(body)
    spans = [m.group().strip().rstrip(".") for m in _SPAN.finditer(body)]
    for span in reversed(spans):
        if not any(c in span for c in "+-*/()"):
            continue
        try:
            parse_expression(span)
            return span
        except (ValueError, ZeroDivisionError):
            continue
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    return lines[-1] if lines else None


# --------------------------------------------------------------------- sudoku

SUDOKU_SIZE, SUDOKU_BOX, SUDOKU_CELLS = 4, 2, 16
SUDOKU_MIN_GIVENS = 6                        # stop carving at a fair "medium" puzzle


def _sudoku_peers() -> list[set[int]]:
    peers: list[set[int]] = [set() for _ in range(SUDOKU_CELLS)]
    for i in range(SUDOKU_CELLS):
        r, c = divmod(i, SUDOKU_SIZE)
        for k in range(SUDOKU_SIZE):
            peers[i].add(r * SUDOKU_SIZE + k)
            peers[i].add(k * SUDOKU_SIZE + c)
        br, bc = (r // SUDOKU_BOX) * SUDOKU_BOX, (c // SUDOKU_BOX) * SUDOKU_BOX
        for dr in range(SUDOKU_BOX):
            for dc in range(SUDOKU_BOX):
                peers[i].add((br + dr) * SUDOKU_SIZE + bc + dc)
        peers[i].discard(i)
    return peers


PEERS = _sudoku_peers()


def solve_sudoku(grid: list[int], rng: random.Random | None = None, limit: int = 2) -> list[list[int]]:
    """Backtracking with MRV. Returns at most `limit` solutions (1 is enough to test uniqueness)."""
    g = list(grid)
    sols: list[list[int]] = []

    def rec():
        if len(sols) >= limit:
            return
        best_i, best_c = -1, None
        for i in range(SUDOKU_CELLS):
            if g[i] == 0:
                used = {g[j] for j in PEERS[i]}
                cands = [v for v in range(1, SUDOKU_SIZE + 1) if v not in used]
                if not cands:
                    return
                if best_c is None or len(cands) < len(best_c):
                    best_i, best_c = i, cands
                    if len(cands) == 1:
                        break
        if best_i < 0:
            sols.append(list(g))
            return
        vals = list(best_c)
        if rng is not None:
            rng.shuffle(vals)
        for v in vals:
            g[best_i] = v
            rec()
            g[best_i] = 0
            if len(sols) >= limit:
                return

    rec()
    return sols


def make_sudoku(rng: random.Random) -> dict:
    full = solve_sudoku([0] * SUDOKU_CELLS, rng=rng, limit=1)[0]
    puzzle = list(full)
    order = list(range(SUDOKU_CELLS))
    rng.shuffle(order)
    for i in order:                            # carve greedily, keep uniqueness
        if sum(1 for v in puzzle if v) <= SUDOKU_MIN_GIVENS:
            break
        keep = puzzle[i]
        puzzle[i] = 0
        if len(solve_sudoku(puzzle, limit=2)) != 1:
            puzzle[i] = keep
    return {
        "size": SUDOKU_SIZE,
        "box": SUDOKU_BOX,
        "puzzle": puzzle,
        "givens": sum(1 for v in puzzle if v),
        "solution": full,
    }


def generate_sudoku(n: int, seed: int = 0) -> list[dict]:
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}:sudoku:{i}")
        out.append({"uid": f"sudoku-{i:04d}", "task": "sudoku", **make_sudoku(rng)})
    return out


def format_grid(grid: list[int] | list[list[int]]) -> str:
    flat = list(grid[0]) + list(grid[1]) + list(grid[2]) + list(grid[3]) if isinstance(grid[0], list) else list(grid)
    return "\n".join(" ".join(str(flat[r * SUDOKU_SIZE + c]) for c in range(SUDOKU_SIZE))
                     for r in range(SUDOKU_SIZE))


def parse_grid(text: str | None) -> list[list[int]] | None:
    """Read a 4x4 grid from four lines of four digits (0 allowed while parsing)."""
    if not text:
        return None
    rows: list[list[int]] = []
    for line in text.splitlines():
        if "```" in line:
            continue
        line = re.sub(r"^\s*[\[(]?\d+[\])]?\s*[:.)-]\s*", "", line)   # "1: 2 3 4 1"
        toks = re.findall(r"\d", line)
        if len(toks) == SUDOKU_SIZE:
            rows.append([int(t) for t in toks])
    if len(rows) >= SUDOKU_SIZE:
        return rows[-SUDOKU_SIZE:]
    digits = re.findall(r"[1-4]", text)
    if len(digits) >= SUDOKU_CELLS:              # one-line "1234 1234 ..." grid
        flat = [int(d) for d in digits[-SUDOKU_CELLS:]]
        return [flat[r * SUDOKU_SIZE:(r + 1) * SUDOKU_SIZE] for r in range(SUDOKU_SIZE)]
    return None


def _sudoku_conflicts(grid: list[list[int]]) -> list[str]:
    errs = []
    for label, unit in _sudoku_units().items():
        vals = [grid[i // SUDOKU_SIZE][i % SUDOKU_SIZE] for i in unit]
        if sorted(vals) != list(range(1, SUDOKU_SIZE + 1)):
            errs.append(f"{label} is not 1-{SUDOKU_SIZE} exactly once")
    return errs


def _sudoku_units() -> dict[str, list[int]]:
    units: dict[str, list[int]] = {}
    for r in range(SUDOKU_SIZE):
        units[f"row {r}"] = [r * SUDOKU_SIZE + c for c in range(SUDOKU_SIZE)]
    for c in range(SUDOKU_SIZE):
        units[f"column {c}"] = [r * SUDOKU_SIZE + c for r in range(SUDOKU_SIZE)]
    k = 0
    for br in range(0, SUDOKU_SIZE, SUDOKU_BOX):
        for bc in range(0, SUDOKU_SIZE, SUDOKU_BOX):
            k += 1
            units[f"box {k}"] = [(br + dr) * SUDOKU_SIZE + bc + dc
                                 for dr in range(SUDOKU_BOX) for dc in range(SUDOKU_BOX)]
    return units


def check_sudoku(text: str | None, puzzle: dict) -> tuple[bool, str]:
    """Exact check: valid 4x4 grid and every given cell preserved."""
    grid = parse_grid(text) if isinstance(text, str) else text
    if grid is None:
        return False, "could not read a 4x4 grid"
    if len(grid) != SUDOKU_SIZE or any(len(row) != SUDOKU_SIZE for row in grid):
        return False, "grid is not 4 rows of 4"
    if any(v not in range(1, SUDOKU_SIZE + 1) for row in grid for v in row):
        return False, "grid has a cell outside 1-4 (unfilled?)"
    errs = _sudoku_conflicts(grid)
    if errs:
        return False, "; ".join(errs)
    for i, given in enumerate(puzzle["puzzle"]):
        if given and grid[i // SUDOKU_SIZE][i % SUDOKU_SIZE] != given:
            return False, f"given at ({i // SUDOKU_SIZE},{i % SUDOKU_SIZE}) changed"
    return True, "ok"


def normalise_sudoku(text: str | None) -> str | None:
    grid = parse_grid(text)
    return format_grid(grid) if grid else None


# -------------------------------------------------------------------- prompts

COUNTDOWN_EXAMPLE = {"numbers": [8, 6, 4, 3], "target": 30,
                     "solution": "(8 + 6 - 4) * 3"}

SUDOKU_EXAMPLE = {
    "puzzle": [1, 0, 0, 0,
               0, 0, 1, 0,
               0, 1, 0, 0,
               0, 0, 0, 1],
    "solution": [1, 2, 3, 4,
                 3, 4, 1, 2,
                 2, 1, 4, 3,
                 4, 3, 2, 1],
}


def countdown_prompt(numbers: list[int], target: int) -> str:
    """One-shot prompt; asks for the final expression only."""
    head = ("Use each of the numbers exactly once, with +, -, * and / and parentheses, "
            "to reach the target. Every number must be used once; you may combine results. "
            "Answer with the final expression only, no explanation.")
    ex = COUNTDOWN_EXAMPLE
    return (
        f"{head}\n\n"
        f"Numbers: {' '.join(map(str, ex['numbers']))}\n"
        f"Target: {ex['target']}\n"
        f"Expression: {ex['solution']}\n\n"
        f"Numbers: {' '.join(map(str, numbers))}\n"
        f"Target: {target}\n"
        f"Expression:"
    )


def sudoku_prompt(puzzle: list[int]) -> str:
    """One-shot prompt; asks for the completed grid only."""
    ex = SUDOKU_EXAMPLE
    return (
        "Fill this 4x4 Sudoku grid. Every row, every column and every 2x2 box must "
        "contain the digits 1, 2, 3, 4 exactly once. Use 0 for empty cells in the "
        "puzzle. Answer with the completed grid only: four lines, four space-separated "
        "digits per line, no explanation.\n\n"
        f"Puzzle:\n{format_grid(ex['puzzle'])}\n"
        f"Grid:\n{format_grid(ex['solution'])}\n\n"
        f"Puzzle:\n{format_grid(puzzle)}\n"
        "Grid:"
    )


# --------------------------------------------------------------------- write

def write_puzzles(out_dir: Path, n: int, seed: int, which: str = "both") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if which in ("countdown", "both"):
        path = out_dir / "countdown.jsonl"
        rows = generate_countdown(n, seed)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths["countdown"] = path
        print(f"wrote {len(rows)} countdown puzzles -> {path}")
    if which in ("sudoku", "both"):
        path = out_dir / "sudoku.jsonl"
        rows = generate_sudoku(n, seed)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths["sudoku"] = path
        print(f"wrote {len(rows)} sudoku puzzles -> {path}")
    return paths


# ------------------------------------------------------------------ selftest

def selftest() -> int:
    cps = generate_countdown(5, seed=12345)
    for p in cps:
        ok, why = check_countdown(p["solution"], p)
        assert ok, (p["uid"], why)
        nums = p["numbers"]
        assert len(nums) in range(COUNTDOWN_MIN_NUMS, COUNTDOWN_MAX_NUMS + 1)
        assert COUNTDOWN_TARGET_MIN <= p["target"] <= COUNTDOWN_TARGET_MAX
        # negative: reusing one number must fail the multiset check
        bad = f"{nums[0]} + {nums[0]}"
        assert not check_countdown(bad, p)[0], p["uid"]
        # negative: a valid expression for the wrong target must fail
        assert not check_countdown(f"({nums[0]} + {nums[1]})", p)[0], p["uid"]
        # normaliser round-trip through a chatty completion, including × and ÷
        text = f"Let me think.\nExpression: {p['solution'].replace('*', '×')}\n"
        norm = normalise_countdown(text)
        assert norm and check_countdown(norm, p)[0], (p["uid"], norm)

    sps = generate_sudoku(5, seed=12345)
    for p in sps:
        grid = p["solution"]
        ok, why = check_sudoku(format_grid(grid), p)
        assert ok, (p["uid"], why)
        sols = solve_sudoku(p["puzzle"], limit=2)
        assert len(sols) == 1 and sols[0] == grid, p["uid"]
        assert SUDOKU_MIN_GIVENS <= p["givens"] < SUDOKU_CELLS, (p["uid"], p["givens"])
        # normaliser round-trip through a chatty completion
        norm = normalise_sudoku(f"Here is the grid:\n{format_grid(grid)}\n")
        assert norm and check_sudoku(norm, p)[0], (p["uid"], norm)
        # negative: swapping two cells in a row must break the row constraint
        bad = list(grid)
        bad[0], bad[1] = bad[1], bad[0]
        assert not check_sudoku(format_grid(bad), p)[0], p["uid"]
        # negative: an unfilled cell is not a complete grid
        assert not check_sudoku(format_grid(p["puzzle"]), p)[0], p["uid"]

    print(f"PASS  countdown=5 (targets {min(p['target'] for p in cps)}-"
          f"{max(p['target'] for p in cps)}, numbers {min(len(p['numbers']) for p in cps)}-"
          f"{max(len(p['numbers']) for p in cps)})  sudoku=5 (givens "
          f"{min(p['givens'] for p in sps)}-{max(p['givens'] for p in sps)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="generate 5+5 puzzles and validate them")
    ap.add_argument("--make", action="store_true", help="write the 100-puzzle files")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["countdown", "sudoku", "both"], default="both")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.make:
        write_puzzles(a.out, a.n, a.seed, a.task)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
