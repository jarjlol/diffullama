# Verification standards — triggered detail

> Load only when the task matches Triggers in `AGENTS.md`:
> citation, paper, claim, novelty, verify, benchmark, index mapping.

These are not generic good practices. Every rule below exists because this project already
made the mistake. The full incident log with root causes is `project-docs/05-mistakes-and-bugs.md`.

## Must do

- **Read the primary source before treating a claim as established.** Full paper text, not
  the abstract and not a search-engine or fetch-tool summary. Five separate errors here trace
  to trusting a summary; two of those would have shipped a wrong research claim.
- **Verify every citation resolves** to a real, correct paper before using it. Check the DOI
  or arXiv ID actually loads and the title matches what you claim it is.
- **Read the code a config name controls, not the name.** A config called
  `remasking='low_confidence'` in one of this project's reference models does the opposite
  of what the name suggests.
- **Assert length/bounds on any hand-built index mapping** before using it
  (`assert len(scores) == len(pool)`). Prefer an explicit `{key: value}` map over a
  positional array — a wrong entry is then one wrong value, not a silent cascading shift.
- **Label unverified claims as unverified.** `project-docs/03-established-facts.md` uses
  PRIMARY / SECONDARY / WEAK confidence tags; match that convention. Do not promote a
  single-source claim to a fact because it is convenient.
- **Re-check novelty at every milestone.** This field publishes ~10 preprints a week. A
  concurrent paper posted three months earlier nearly invalidated a semester's framing here
  before someone found it.
- **Pass `encoding='utf-8'` to every `open()`**, and set `PYTHONIOENCODING=utf-8` when output
  may contain non-ASCII. Three separate crashes here came from the Windows cp1252 default.

## Must not

- Do not report an API/infrastructure failure as a substantive finding. An unauthenticated
  Semantic Scholar call returns HTTP 429 immediately; treating that as "paper not found"
  once produced a near-1.0 hallucination rate that looked like a real result.
- Do not trust a fuzzy-match retrieval hit without a similarity threshold. OpenAlex returns
  *something* for almost any query — six real preprints once matched unrelated papers at
  0.18–0.33 similarity and were silently counted as hallucinated.
- Do not silently edit away a retraction or a wrong earlier conclusion. Append a dated
  correction underneath it so the reasoning stays auditable.

## Example

```python
# Verify a retrieved reference actually matches before counting it as found.
score = title_similarity(query_title, result["title"])
if score < MATCH_THRESHOLD:
    status = "not_found"          # not "hallucinated" - be precise about which
elif http_status == 429:
    status = "infra_failure"      # excluded from denominator, never scored as a miss
else:
    status = "verified"
```
