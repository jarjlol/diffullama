"""
Extractor Agent (BAGELS-style explicit/implicit limitation extraction,
Al Azher et al. 2025, arXiv:2505.18207 Sec 3.2).

BAGELS' method: identify limitation-related keywords, then extract from that
point until a terminal section marker (acknowledgements, grant, future work,
discussion, conclusion, appendix), refine with an LLM instructed NOT to
paraphrase/alter/invent content -- only select and clean up genuine spans.

Adaptation for this paper: BAGELS assumes papers discuss limitations in one
contiguous block (a Discussion/Limitations section). The anchor paper (per a
direct grep -- see below) has NO dedicated Limitations section; candidate
sentences are scattered as single admissions throughout Introduction,
Experiment, Discussion, and Appendix. So instead of "scan to next terminal
marker," this extracts a local paragraph-window around each keyword hit --
a necessary, documented deviation from BAGELS' original method, not a
silent substitution.

No hosted LLM API is available in this environment (same constraint as the
litreview/ pipeline). The "LLM refinement" step BAGELS uses to filter noise
from regex hits is performed by Claude directly, reading extract_candidates.txt
and keeping/discarding spans by hand -- documented exactly as transparently
as litreview/REPORT.md documents the same substitution.
"""
import re

KEYWORDS = [
    "limitation", "shortcoming", "drawback", "does not", "do not", "cannot",
    "unable to", "fail", "we leave", "future work", "remains unexplored",
    "out of scope", "not fully", "still lack", "no longer",
]

TERMINAL_MARKERS = re.compile(
    r"^\s*(ACKNOWLEDG|R EFERENCES|A PPENDIX)", re.IGNORECASE
)


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def find_section(lines, line_idx):
    """Walk backward from line_idx to the nearest section header, for provenance."""
    header_re = re.compile(r"^\s*\d+(\.\d+)?\s+[A-Z]")
    for i in range(line_idx, -1, -1):
        if header_re.match(lines[i]):
            return lines[i].strip()
    return "(preamble/abstract)"


def extract_candidates(path, window=2):
    lines = load_lines(path)
    hits = []
    seen_spans = set()
    for i, line in enumerate(lines):
        low = line.lower()
        for kw in KEYWORDS:
            if kw in low:
                start = max(0, i - window)
                end = min(len(lines), i + window + 1)
                span_text = " ".join(l.strip() for l in lines[start:end] if l.strip())
                span_key = (start, end)
                if span_key in seen_spans or len(span_text) < 20:
                    continue
                seen_spans.add(span_key)
                hits.append({
                    "line": i + 1,
                    "keyword": kw,
                    "section": find_section(lines, i),
                    "span": span_text,
                })
                break  # one hit per line is enough
    return hits


def main():
    hits = extract_candidates("../data/anchor_fulltext.txt")
    print(f"Found {len(hits)} candidate spans (regex keyword scan, before LLM refinement)\n")
    with open("../data/extractor_candidates.txt", "w", encoding="utf-8") as f:
        for h in hits:
            line = f"[L{h['line']}] ({h['section']}) kw={h['keyword']!r}\n  {h['span']}\n\n"
            f.write(line)
            print(line, end="")


if __name__ == "__main__":
    main()
