"""Shared test fixtures. Not part of the package's public API."""
from __future__ import annotations

from egr.types import Span


class CharBackend:
    """A trivial one-char-per-token tokenizer, used ONLY to exercise canvas.py's index
    arithmetic in isolation from a real SentencePiece tokenizer. Every character maps to
    exactly one id and one Span, so the correct canvas positions can be worked out by hand
    and compared against what the code produces.
    """
    name = "char-stub"
    bos_id = -1

    def tokenize(self, text: str) -> tuple[list[int], list[Span]]:
        ids = [ord(c) for c in text]
        spans = [Span(i, i + 1, "char") for i in range(len(text))]
        return ids, spans

    def detokenize(self, ids) -> str:
        return "".join(chr(i) for i in ids)
