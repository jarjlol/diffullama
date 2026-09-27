"""Stage 2 -- entity-centric knowledge store and retrieval (paper sec 3.2, Eq. 1-2).

Store. K is an m x m co-occurrence matrix over entities extracted from the titles and
abstracts of the corpus (the paper restricts extraction to titles and abstracts too).
It records each entity's count and the count of each entity pair within a paper.

Retrieval. The top-k entities NOT already present in the input papers, by Eq. 2:

    argmax_{I, |I|=k}  prod_{e_i in I} ( prod_{e_j in E_input} P(e_j | e_i) ) * P(e_i)

computed in log space with add-alpha smoothing, since one unseen pair would otherwise
zero the product.

DEVIATION (REPORT 3.2). The paper extracts entities with the BLINK entity linker
(Wu et al., 2020), a neural Wikipedia linker. BLINK cannot be installed under this
directory's stdlib-only rule, so entities are extracted by a deterministic rule-based
extractor: (a) acronyms and mixed-case technical names (GSM8K, LLaDA, DiffuLLaMA),
and (b) 2-3 word noun phrases that recur across at least two papers. It extracts more
entities per paper than BLINK's reported ~3 and is not linked to Wikipedia, so it
captures field vocabulary rather than encyclopedic entities.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, IDEATION, read_json, write_json  # noqa: E402

GENERIC = set("""
paper propose proposed proposes show shows shown results result approach approaches method methods
novel new based using use used achieve achieves achieved significantly significant performance
existing recent work works study studies demonstrate demonstrates however allows allow enable
enables enabling various different several large small high low first two three many well also
extensive experiments experimental framework model models task tasks data problem problems yet
further including include state art strong better best while across without within compared
furthermore moreover additionally specifically particular general key important main potential
""".split())
STOP = set("""a an the of to in on for with and or by from at as is are was were be been this that these
those it its we our their which into than then such via can may also not but both each through over under
between among per up down out about only more most less higher lower faster slower better worse times time
up https http www github com org io html pdf code available""".split())
# Verbs and verb forms that make an n-gram a clause fragment rather than a noun phrase.
VERBS = set("""generate generates generating generated improve improves improving improved reduce reduces
reducing reduced produce produces producing produced perform performs performing performed outperform
outperforms outperforming require requires requiring required achieve introduce introduces introducing
present presents presenting leverage leverages leveraging enable learn learns learning-based remain remains
make makes making obtain obtains yield yields yielding match matches matching reach reaches support supports
provide provides providing offer offers offering suffer suffers apply applies applying train trains trained
evaluate evaluates evaluated adapt adapts converting convert converts rely relies relying retain retains
speedups speedup have has had recently emerged emerge emerging compelling alternative promising remarkable
impressive notable powerful effective efficient simple competitive comparable superior""".split())
NAME_ALLOW_LOWER = {"diffullama", "diffugpt", "llada", "gsm8k"}


def acronyms(text: str) -> set[str]:
    out = set()
    for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{1,20}\b", text):
        uppers = sum(ch.isupper() for ch in tok)
        has_digit = any(ch.isdigit() for ch in tok)
        if (uppers >= 2 or (uppers >= 1 and has_digit) or tok.lower() in NAME_ALLOW_LOWER) \
                and not tok.isupper() or (tok.isupper() and 2 <= len(tok) <= 8):
            if tok.lower() not in STOP and not tok.isdigit():
                out.add(tok.strip("-"))
    return out


def phrases(text: str) -> list[str]:
    out = []
    # n-grams never cross punctuation: "text generation, discrete ..." must not yield
    # "generation discrete".
    for clause in re.split(r"[.,;:()\[\]!?\"]", text.lower()):
        out.extend(_clause_phrases(re.findall(r"[a-z][a-z\-]+", clause)))
    return out


def _clause_phrases(words: list[str]) -> list[str]:
    out = []
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if any(w in STOP or w in GENERIC or w in VERBS for w in gram):
                continue
            last = gram[-1]
            if last.endswith("s") and not last.endswith("ss") and len(last) > 3:
                gram = gram[:-1] + [last[:-1]]      # models -> model
            out.append(" ".join(gram))
    return out


def extract(corpus: list[dict]) -> list[list[str]]:
    per_paper_phrases = [set(phrases(p["title"] + ". " + p["abstract"])) for p in corpus]
    df = Counter(ph for s in per_paper_phrases for ph in s)
    ents = []
    for p, phs in zip(corpus, per_paper_phrases):
        text = p["title"] + ". " + p["abstract"]
        kept = {ph for ph in phs if df[ph] >= 2}
        ents.append(sorted(kept | acronyms(text)))
    return ents


def main() -> int:
    cfg = read_json(IDEATION / "config.json")
    lit = read_json(DATA / "literature.json")
    corpus = lit["entity_corpus"]
    alpha, k = cfg["smoothing_alpha"], cfg["n_entities"]

    paper_ents = extract(corpus)
    count: Counter[str] = Counter()
    pair: Counter[tuple[str, str]] = Counter()
    for ents in paper_ents:
        count.update(ents)
        for a, b in combinations(sorted(set(ents)), 2):
            pair[(a, b)] += 1
    vocab = sorted(count)
    m, total = len(vocab), sum(count.values())

    def cooc(a: str, b: str) -> int:
        return pair[(a, b) if a < b else (b, a)]

    # E_input: entities of the target paper and the related papers actually given to the LLM
    input_titles = {lit["target"]["title"], *[r["title"] for r in lit["related"]]}
    e_input = sorted({e for p, ents in zip(corpus, paper_ents) if p["title"] in input_titles for e in ents})

    scored = []
    for ei in vocab:
        if ei in e_input:
            continue
        logp = math.log(count[ei] / total)                                   # P(e_i)
        denom = count[ei] + alpha * m
        logp += sum(math.log((cooc(ej, ei) + alpha) / denom) for ej in e_input)  # prod P(e_j|e_i)
        scored.append((logp, ei))
    scored.sort(reverse=True)
    retrieved = [e for _, e in scored[:k]]

    write_json(DATA / "entity_store.json", {
        "extractor": "rule-based substitute for BLINK (see module docstring)",
        "n_papers": len(corpus), "n_entities": m,
        "mean_entities_per_paper": round(sum(map(len, paper_ents)) / len(paper_ents), 2),
        "entity_counts": dict(count.most_common()),
        "pair_counts": {f"{a} || {b}": c for (a, b), c in pair.most_common(2000)},
    })
    write_json(DATA / "retrieved_entities.json", {
        "equation": "paper Eq. 2, log-space, add-alpha smoothing",
        "alpha": alpha, "k": k, "n_input_entities": len(e_input),
        "input_entities": e_input,
        "retrieved": [{"entity": e, "log_score": round(s, 3), "count": count[e]} for s, e in scored[:k]],
    })
    print(f"papers              : {len(corpus)}")
    print(f"unique entities (m) : {m}   mean/paper: {sum(map(len, paper_ents)) / len(paper_ents):.1f}  (BLINK in paper: ~3)")
    print(f"input entities      : {len(e_input)}  (target + {len(lit['related'])} related papers)")
    print(f"retrieved top-{k}     : {retrieved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
