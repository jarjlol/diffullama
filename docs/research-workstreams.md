# Research workstreams — triggered detail

> Load only when the task matches Triggers in `AGENTS.md`:
> litreview, limitations, survey, QUAL-SG, assignment, submission.

The course project runs as several separate assignment deliverables plus one main research
project. Each lives in its own top-level directory and is self-contained.

## Must do

- **Treat each assignment directory as independent.** `litreview/` and `limitations/` do not
  share code or reuse each other's findings — that separation is deliberate and documented in
  each one's `REPORT.md`. Do not "helpfully" wire them together.
- **Start from the directory's own `README.md`, then its `REPORT.md`.** Each has a README
  that maps the pipeline and a REPORT that documents methodology, deviations from the source
  paper, results, and honest limitations.
- **Keep the pure-stdlib constraint.** These pipelines use only the Python standard library
  plus `requests`. No numpy, sklearn, or sentence-transformers — they were built where no
  package installer was available, and the TF-IDF/cosine/percentile-rank helpers are
  hand-written for that reason.
- **Document substitutions honestly.** Where a source paper specifies an LLM API call or a
  learned embedding model and the implementation substitutes something else, that is stated
  explicitly in the REPORT with the reason. Maintain that standard in any extension.
- **Preserve positional citation integrity.** A generated survey cites references by index
  (`[1]`–`[48]`) against a specific reference file. Swapping the underlying reference set
  without regenerating the prose silently desyncs every citation marker.

## Must not

- Do not overwrite a submitted deliverable to apply an improvement. The established pattern
  is additive: write improved results to new `*_llm_judged`-style files and document what
  fully adopting them would require.
- Do not present a retrieval-based reference set as evidence that an LLM avoided
  hallucinating. If references were *retrieved* and then verified against the same database
  they came from, that check is close to tautological — say so.

## Current status and blockers

Assignment status, the still-open main-project direction decision, and submission logistics
are tracked in `project-docs/01-project-brief.md`, `02-decision-log.md`, and
`04-open-questions.md`. Read those before asking whether something is done.

## Example

```sh
# Each pipeline runs from its own scripts/ directory, in documented order.
cd litreview/scripts
python3 retrieve.py        # each script prints its own progress/results
python3 arxiv_fetch.py     # no silent failure mode: bad output is visible in the ranking
```
