"""Every dataclass and enum EGR uses. No logic lives here -- this is the vocabulary the
other modules speak. See docs/01-architecture.md Sec 3.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Verdict(StrEnum):
    PASS = "pass"                  # every test passed
    FAIL = "fail"                  # a test assertion failed
    ERROR = "error"                # the program raised
    SYNTAX_ERROR = "syntax_error"  # ast.parse rejected it
    TIMEOUT = "timeout"            # exceeded the wall clock
    HARNESS_ERROR = "harness_error"  # OUR bug -- never counted as a model result (H-6)


@dataclass(frozen=True)
class Task:
    task_id: str                  # "HumanEval/0"
    prompt: str                   # signature + docstring
    seed_program: str             # what depth 0 starts from
    tests: str                    # rendered, runnable asserts
    entry_point: str              # function under repair
    ground_truth_line: int | None = None  # mutants only; scores localization
    oracle_program: str | None = None  # mutants only; the known-correct source. Read ONLY by
    # MockBackend's "oracle"/"noisy" modes (docs/04-build-phases.md Phase 1) -- a real
    # DiffusionBackend never sees this field. Not part of the original architecture doc's
    # Task fields; added because MockBackend's documented "fills the hole from the
    # known-correct source" behaviour has to get that source from somewhere (P5: additive).


@dataclass(frozen=True)
class Span:
    """A half-open interval [start, end). The unit carried between modules -- never a bare int."""
    start: int
    end: int
    kind: Literal["char", "token", "canvas"]

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Span end {self.end} < start {self.start}")


@dataclass(frozen=True)
class TestOutcome:
    name: str
    passed: bool
    exc_type: str | None
    message: str | None
    frames: tuple[tuple[str, int], ...]  # (filename, lineno), innermost last


@dataclass(frozen=True)
class Observation:
    """Everything one execution of the candidate told us."""
    verdict: Verdict
    outcomes: tuple[TestOutcome, ...]
    spectra: Mapping[int, tuple[int, int]]  # line -> (n_passing_tests, n_failing_tests)
    exec_counts: Mapping[int, int]          # line -> times executed, failing tests only
    syntax_error: tuple[int, int, str] | None  # (lineno, offset, msg)
    stdout_tail: str
    duration_s: float


@dataclass(frozen=True)
class Evidence:
    """The normalised failure description. One kind, whatever the verdict was."""
    kind: Literal["syntax", "assertion", "exception", "timeout"]
    lines: tuple[int, ...]        # candidate fault lines, best first
    scores: Mapping[int, float]   # line -> suspiciousness, for the annotation channel
    summary: str                  # one line of human-readable text
    signature: str                # stable hash -- drives no-progress detection


@dataclass(frozen=True)
class RemaskPlan:
    canvas_spans: tuple[Span, ...]  # kind == "canvas", already shift-corrected
    n_masked: int                   # the budget actually spent
    scope: str                      # "leaf" | "parent_leaf" | "function"
    rationale: str                  # why these spans -- goes into the run record


@dataclass(frozen=True)
class Attempt:
    depth: int
    program: str
    observation: Observation
    evidence: Evidence | None
    plan: RemaskPlan | None
    tokens_changed: int | None  # edit locality, vs. the previous depth
    wall_s: float


@dataclass(frozen=True)
class RunRecord:
    task: Task
    attempts: tuple[Attempt, ...]
    solved: bool
    depth: int | None = None          # depth at which it solved, if solved
    aborted: str | None = None        # "harness" | "no_progress" | "no_maskable_region" | None
    best: Attempt | None = None       # best attempt seen, tracked separately from the last one (H-13)


@dataclass
class Config:
    """One run's knobs. Every ablation in docs/02-experiment-plan.md is a sweep over this."""
    benchmark: str
    backend: str
    policy: str
    max_depth: int = 5
    scope: str = "parent_leaf"        # "leaf" | "parent_leaf" | "function"
    annotate: str = "none"            # "none" | "comment" | "compact"
    budget: int = 24                  # token budget handed to every policy, matched
    slack: int = 4                    # extra hole positions beyond the witness span (H-3)
    steps: int = 32
    temperature: float = 0.9
    seed: int = 0
    n: int | None = None              # task limit, None = all
    mock_mode: str = "oracle"         # MockBackend only: "oracle" | "noisy" | "stuck"
    mock_p: float = 0.5               # MockBackend "noisy" mode: P(correct token)
    out: str = "runs/out.jsonl"
    extra: dict = field(default_factory=dict)
