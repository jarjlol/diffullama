"""Stage 3 -- the four worker agents of arXiv:2601.11578 section 3.1.

Execution order (paper Fig. 1):

    Extractor  ->  Analyzer  ->  Reviewer  ->  Citation
                        |
                        v
             Judge  ->  Self-Feedback (conditional, <8/10, max 2 retries)
                        |
                        v
                     Master

Each agent's task instruction below is transcribed from the paper's description of
that agent, keeping its quoted phrasing where the paper gives it.  This module
builds each agent's input payload and writes the exact prompt it was run with, so
every generated limitation is traceable to a prompt and an input.

The LLM itself is external (no hosted API in this build environment) -- see
REPORT.md section 4.  Prompts go to data/agent_requests/, outputs are read back
from data/agent_outputs.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_json, read_text, write_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "limitations" / "data"
LITREVIEW = ROOT / "litreview" / "data"
REQ = DATA / "agent_requests"

# Paper section 3.1 -- provenance labels the Master Agent must preserve.
PROVENANCE = {
    "extractor": "Author-stated",
    "analyzer": "Inferred",
    "reviewer": "Peer-review-derived",
    "citation": "Cited",
}

OUTPUT_CONTRACT = """
Return JSON only:
{"limitations": [{"id": "<AGENT>-1",
                  "statement": "<one specific limitation, 2-4 sentences>",
                  "evidence": "<quote or section reference from the INPUT>",
                  "provenance": "<PROVENANCE>"}]}

Rules:
- Be specific to THIS paper. Reject generic statements ("limited generalizability",
  "dataset bias") unless the paper's own text makes them concrete.
- Every statement must be traceable to the INPUT provided. Do not invent numbers.
- Produce at most 8 limitations. Fewer, sharper items beat more, vaguer ones.
"""

AGENTS = {
    "extractor": {
        "role": "Extractor Agent",
        "task": (
            "Extract all explicitly stated limitations as mentioned by the authors. "
            "Focus on the Discussion, Conclusion, and Future Work sections, and on "
            "any hedging the authors use elsewhere in the paper. Follow the 'chain of "
            "limitations' methodology: where the authors concede a weakness and then "
            "justify or defer it, capture both the concession and the justification. "
            "Record verbatim wording and the location it came from. Do not infer "
            "anything the authors did not write."
        ),
        "input_kind": "anchor_minus_limitations",
    },
    "analyzer": {
        "role": "Analyzer Agent",
        "task": (
            "Act as a methodological auditor. Identify implicit limitations the authors "
            "did NOT state: unaddressed confounders, inadequate statistical power, "
            "missing ablations, claims extrapolated past the evidence, design choices "
            "adopted for convenience and never tested, and results whose interpretation "
            "the paper's own numbers do not support. For each, name the specific "
            "methodological gap and the evidence in the paper that reveals it."
        ),
        "input_kind": "anchor_full",
    },
    "reviewer": {
        "role": "Reviewer Agent",
        "task": (
            "Simulate the viewpoint of a peer reviewer for a top-tier venue. Evaluate "
            "reproducibility, transparency, and ethical considerations: whether the "
            "reported setup is sufficient to reproduce the results, whether claims in "
            "the abstract and introduction are supported by the experiments actually "
            "run, whether baselines are fair and fairly configured, and whether "
            "released artifacts match what the paper describes. Raise the weaknesses a "
            "reviewer would write in a review, not a summary of the paper."
        ),
        "input_kind": "anchor_full",
    },
    "citation": {
        "role": "Citation Agent",
        "task": (
            "Identify and articulate limitations for the current paper, focusing on "
            "shortcomings that align with findings in the related works supplied below. "
            "These works are drawn from the paper's reference pool and from the "
            "surrounding literature. Use them to place the input paper in context: where "
            "does a related work test something this paper assumed, supersede a choice "
            "this paper made, or report a result that undercuts one of its claims? "
            "Every limitation must cite the specific retrieved work that grounds it."
        ),
        "input_kind": "anchor_plus_rag",
    },
}

PROMPT_TEMPLATE = """\
# {role}

You are the {role} in the multi-agent limitation-generation framework of
arXiv:2601.11578 (Al Azher, Guo, Alhoori). You operate on one input paper.

## TASK
{task}

## OUTPUT CONTRACT
{contract}

## INPUT
{payload}
"""


def _norm_heading(h: str) -> str:
    """pdftotext spaces out small-caps headings: '6   C ONCLUSION' -> '6conclusion'.

    Matching on the raw heading silently fails, which is how the Extractor's
    exclusion filter initially dropped nothing at all.  Normalise before matching.
    """
    return "".join(h.split()).lower()


# Everything from this marker onward is Table 12: qualitative *model output*
# samples, not authorial prose.  Agents must not read generated text as a claim
# the authors made -- but the samples are legitimate evidence about output quality,
# so they are kept and explicitly labelled rather than dropped.
_SAMPLE_MARKER = "generation examples"


def _anchor_text(drop_limitation_sections: bool = False) -> str:
    sections = read_json(DATA / "anchor_sections.json")
    if drop_limitation_sections:
        before = len(sections)
        sections = [s for s in sections
                    if "conclusion" not in _norm_heading(s["heading"])
                    and "limitation" not in _norm_heading(s["heading"])]
        assert len(sections) < before, (
            "exclusion filter removed nothing -- heading normalisation is broken"
        )

    out, in_samples = [], False
    for s in sections:
        if _SAMPLE_MARKER in s["text"].lower():
            in_samples = True
        tag = " [MODEL-GENERATED SAMPLE OUTPUT, NOT AUTHOR PROSE]" if in_samples else ""
        out.append(f"## {s['heading']}{tag}\n{s['text']}")
    return "\n\n".join(out)


def _rag_block() -> str:
    retained = read_json(DATA / "rag_retained.json")
    return "\n\n".join(
        f"[{c['chunk_id']}] ({c['provenance']}, {c.get('year')}, rerank {c['rerank_score']}/10) "
        f"{c['source_title']}\n{c['text'][:1200]}"
        for c in retained
    )


def build_payload(kind: str) -> str:
    if kind == "anchor_minus_limitations":
        return _anchor_text(drop_limitation_sections=True)
    if kind == "anchor_full":
        return _anchor_text()
    if kind == "anchor_plus_rag":
        return (f"### INPUT PAPER\n{_anchor_text()[:40000]}\n\n"
                f"### RETRIEVED RELATED WORKS (RAG, re-ranked >= 8/10)\n{_rag_block()}")
    raise ValueError(f"unknown input kind: {kind}")


def main() -> int:
    REQ.mkdir(parents=True, exist_ok=True)
    for name, spec in AGENTS.items():
        payload = build_payload(spec["input_kind"])
        prompt = PROMPT_TEMPLATE.format(
            role=spec["role"],
            task=spec["task"],
            contract=OUTPUT_CONTRACT.replace("<PROVENANCE>", PROVENANCE[name]),
            payload=payload,
        )
        path = REQ / f"{name}.md"
        write_text(path, prompt)
        print(f"{spec['role']:<18} input={spec['input_kind']:<26} "
              f"payload={len(payload):>7,} chars -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
