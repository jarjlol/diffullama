"""Central logging setup.

Every module does `log = logging.getLogger(__name__)` and nothing else -- this file is the
only place that configures handlers, so imports never have side effects. `cli.py` calls
`setup()` once, at start-up. Tests get whatever the stdlib default is (silent).
"""
from __future__ import annotations

import logging


def setup(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )
