"""Line-event tracer and traceback-frame extraction. Runs INSIDE the sandboxed subprocess,
active only for the candidate's own file -- H-12: a badly scoped tracer perturbs what it
measures. Stdlib only; no coverage.py dependency. See docs/01-architecture.md Sec 6.3.
"""
from __future__ import annotations

import sys
import traceback
from collections import Counter
from collections.abc import Callable


class LineTracer:
    """Context manager. While active, counts line hits, restricted to `filename`.

    A per-line hit count (not just a hit/miss set) is what lets a runaway loop show up as
    "line 4 ran 40,000 times" rather than merely "line 4 ran" -- the signal `localize.py` uses
    for the `timeout` evidence kind.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.hit_counts: Counter[int] = Counter()
        self._prev: Callable | None = None

    def _trace(self, frame, event, arg):
        if event == "line" and frame.f_code.co_filename == self.filename:
            self.hit_counts[frame.f_lineno] += 1
        return self._trace

    def __enter__(self) -> "LineTracer":
        self._prev = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        sys.settrace(self._prev)
        return False  # never suppress the candidate's exception


def extract_frames(exc: BaseException, filename: str) -> tuple[tuple[str, int], ...]:
    """(filename, lineno) pairs from `exc`'s traceback, restricted to `filename`, innermost
    last -- exactly the "failure frame" `localize.py` uses for the `exception` evidence kind.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    return tuple((f.filename, f.lineno) for f in frames if f.filename == filename)
