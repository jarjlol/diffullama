# Limitation Generation / Research Gap Identification

Implements techniques from three papers (BAGELS, LimitGen, Multi-Agent LLMs for Generating Research
Limitations) to extract limitations of the anchor paper (DiffuLLaMA) and state a continuation
research problem. **Start here:** [`REPORT.md`](REPORT.md) for full methodology; the required
deliverable is [`output/limitations_and_research_problem.md`](output/limitations_and_research_problem.md).

Built independently of `litreview/` and `project-docs/` — no findings, conclusions, or code are
reused from either; see REPORT.md §1 for why.

> ⚠️ **Read [`CORRECTIONS.md`](CORRECTIONS.md) before citing any limitation by its L-number.** Two
> incompatible runs of this pipeline are in circulation with clashing numbering (the committed run’s
> `L8` is the circulated run’s `L6`), and one factual error has been identified and traced to a shared
> upstream source. Nothing in `output/` or `data/` has been edited; the corrections are additive.

## Pipeline

```
scripts/
  extract_stated_limitations.py   Extractor: regex keyword scan of the anchor's full text
  citation_agent.py               Citation Agent: fresh OpenAlex + arXiv retrieval per candidate
data/
  anchor_fulltext.txt              fresh pdftotext extraction (independent copy)
  extractor_candidates.txt         raw regex hits (14)
  extractor_refined.md             LLM-refinement pass: 5 kept, 9 discarded, with reasons
  analyzer_findings.md             8 inferred implicit limitations (independent methodological read)
  citation_agent_results.txt       raw retrieval output
  citation_agent_verdict.md        groundedness judgments per candidate
  master_consolidated.md           Judge scoring + taxonomy/provenance tagging + final ranked list
output/
  limitations_and_research_problem.md   the assignment's required deliverable
```

Run order: `extract_stated_limitations.py` → (manual LLM refinement) → (manual Analyzer pass) →
`citation_agent.py` → (manual Judge/Master consolidation) → final deliverable. The manual steps are
documented, not silent — see REPORT.md §4 for why (no hosted LLM API in this build environment, same
constraint `litreview/` already documents).
