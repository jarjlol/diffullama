# Final Submission (Task 5): research paper, report and repo

**Deadline: Thursday 26 Nov 2026, 1 pm** (time zone not stated; assume IST [unverified]). Treat **Wed 25 Nov evening** as the real deadline; Thu morning is only a buffer for a broken upload. Tracker: epic #17; milestone "Task 5: Final Submission".

The folder named **Final Submission** in our group folder holds exactly three items:

1. the **paper**, 4 pages + 1 reference page: `paper/` (sections in `paper/sections/`, built PDF in `paper/`);
2. the **report**, 2 to 4 pages, observations and efforts for each task: `report/` (issue #15);
3. the **GitHub repo** of the pipeline and the code behind the paper: this repo (see "Pipeline code" below).

Creating that folder and uploading is a human step (#16).

## Who writes what

| Section | File | Issue | Owner |
|---|---|---|---|
| Abstract | `paper/sections/abstract.md` | #13 | aryan (last) |
| Introduction | `paper/sections/introduction.md` | #8 | neel |
| Related Work | `paper/sections/related_work.md` | #9 | aaditya |
| Problem Formulation | `paper/sections/problem_formulation.md` | #10 | neel |
| Methodology | `paper/sections/methodology.md` | #11 | aaditya |
| Experiments / Results | `paper/sections/experiments.md` | #12 | arjun |
| Conclusion | `paper/sections/conclusion.md` | #13 | aryan (last) |
| Report | `report/` | #15 | aryan |
| Assembly (skeleton is done) | | #14 | neel |
| Epic tracker, upload to the group folder | | #17, #16 | aaditya, arjun |

Each stub quotes the professor's instruction for its section and lists the source files.

The label `to:<handle>` on each issue is authoritative if this table is stale. khushi and diya do not use agents and have no Task 5 issues. Agent start-up prompts per member: `AGENT_PROMPTS.md`.

## Schedule

| Date | Checkpoint |
|---|---|
| Fri 20 Nov | skeleton merged; decision #7 (how much of Task 4 iteration 2) answered |
| **Sun 22 Nov** | **results freeze** |
| Mon 23 Nov | all section drafts in |
| Tue 24 Nov | review of every section by a different member |
| **Wed 25 Nov** | assembly, page count, bibliography check, PDFs; **everything in the Final Submission folder by the evening** (#16) |
| **Thu 26 Nov, 1 pm** | **deadline** (buffer only) |

## Rules

- One section per branch `mesh/<issue>-<slug>`, from `main`, only that section's file in the PR.
- Receipts for every claim; `[unverified]` for anything not checked (see `docs/verification-standards.md`).
- Do not overwrite a submitted deliverable; additions only.
- The paper has **one reference page in total**: keep your references few and verified; report your count in your status comment.

## Pipeline code (for deliverable 3)

| Assignment | Code and output |
|---|---|
| Literature review | `litreview/` (on `main`) |
| Limitations / gap extraction | `limitations/` (on `main`; multi-agent variant on branch `assignment/limitations-multiagent`) |
| Ideation | `ideation/` on branch `assignment/research-ideation` |
| Codification (Task 4) | `codification/` on branch `assignment/codification` |

### How to rerun each, and what needs a GPU

No step needs a secret committed to the repo, model weights in git, or a large file: generated candidates (`items.jsonl`) and hidden states are git-ignored and live on the workstation or the team Drive.

| Pipeline | Rerun | GPU? | Checked |
|---|---|---|---|
| Literature review (`litreview/`) | `cd litreview/scripts && python3 retrieve.py && python3 arxiv_fetch.py && python3 merge_finalize.py && python3 format_refs.py`; survey generation is a manual LLM step (`litreview/REPORT.md` §4); then `python3 evaluate.py` | No. Python stdlib + `requests`; retrieval needs network access to OpenAlex and arXiv | `evaluate.py` reran offline on a copy, 2026-11-20 |
| Limitations (`limitations/`) | `cd limitations/scripts && python3 extract_stated_limitations.py`, then the manual LLM refinement / Analyzer steps, then `python3 citation_agent.py`, then manual Judge / Master (`limitations/README.md`, run order) | No. stdlib + `requests`; the Citation Agent needs OpenAlex/arXiv | not rerun [unverified] |
| Ideation (`ideation/`, branch `assignment/research-ideation`) | `python3 ideation/scripts/run_pipeline.py --estimate` (864 calls at the default config); `python3 ideation/scripts/run_pipeline.py` (manual backend: writes prompts, stops) or `LLM_API_KEY=... python3 ideation/scripts/run_pipeline.py --backend openai` (any OpenAI-compatible endpoint, e.g. a local vLLM) | No, unless the LLM is served locally | `python3 ideation/scripts/selftest.py`: **10/10 passed** at e5f3383, 2026-11-20 |
| Codification, Task 4 (`codification/`, branch `assignment/codification`) | GPU side, one model per process and one venv per model family: `pipeline/prepare_data.py` → `generate_candidates.py` → `score_candidates.py` → `assemble.py` → `measure_preservation.py` (`codification/README.md` §B). CPU side, from cached items: `python3 rerank_task/baselines.py --split test`, `python3 pipeline/analyze.py --split test`, `python3 paired_stats.py ...` (`codification/HANDOFF.md` §3). Arbor: `codification/HANDOFF.md` "Rerun Arbor" | **Yes** for generation, scoring, hidden states and the Arbor LLM (both GPUs of the Blackwell workstation); **no** for selectors, statistics and the wiring test | `python3 pipeline/selftest.py`: **SELFTEST PASSED** at 6ffafa5, 2026-11-20 (mock models, CPU) |

Iteration 1's numbers are committed under `codification/results/iter1/numbers/`; reproducing them on CPU needs the cached `items.jsonl` files (git-ignored, shared on the team Drive). Iteration 2: `codification/ITERATION_2.md` on branch `iter2/aaditya-worker` until merged.
