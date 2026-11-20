"""Frozen operator-support majority bar; selected on dev only, never retuned on gate."""
from collections import Counter
import re
TIE = 'earliest'
IGNORE_NONE = True
_MARKERS = re.compile(r"\n\s*(?:Question|Q)\s*:", re.I)
_CALC = re.compile(r"-?[\d,]+(?:\.\d+)?\s*[+\-*/x×]\s*-?[\d,]+(?:\.\d+)?\s*=")
def select(item):
    candidates = item["candidates"]
    answers = [c.get("answer") for c in candidates]
    counts = Counter(a for a in answers if a is not None or not IGNORE_NONE)
    if not counts:
        return 0
    top = max(counts.values())
    pool = [i for i,a in enumerate(answers) if a in counts and counts[a] == top]
    if TIE == "earliest":
        return pool[0]
    def key(i):
        text = _MARKERS.split(candidates[i].get("text", ""), maxsplit=1)[0].strip()
        return (len(text) if TIE == "shortest" else -len(_CALC.findall(text)), i)
    return min(pool, key=key)
