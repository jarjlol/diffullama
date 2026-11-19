# Final Submission (Task 5): research paper, report and repo

**Deadline: Thursday 26 Nov 2026.** Tracker: epic #17; milestone "Task 5: Final Submission".

The folder named **Final Submission** in our group folder holds exactly three items:

1. the **paper**, 4 pages + 1 reference page: `paper/` (sections in `paper/sections/`, built PDF in `paper/`);
2. the **report**, 2 to 4 pages, observations and efforts for each task: `report/` (issue #15);
3. the **GitHub repo** of the pipeline and the code behind the paper: this repo (see "Pipeline code" below).

Creating that folder and uploading is a human step (#16).

## Who writes what

| Section | File | Issue | Owner |
|---|---|---|---|
| Abstract | `paper/sections/abstract.md` | #13 | aryan (last) |
| Introduction | `paper/sections/introduction.md` | #8 | diya |
| Related Work | `paper/sections/related_work.md` | #9 | khushi |
| Problem Formulation | `paper/sections/problem_formulation.md` | #10 | neel |
| Methodology | `paper/sections/methodology.md` | #11 | aaditya |
| Experiments / Results | `paper/sections/experiments.md` | #12 | arjun |
| Conclusion | `paper/sections/conclusion.md` | #13 | aryan (last) |
| Report | `report/` | #15 | aalhad |
| Assembly + this skeleton | | #14 | aalhad |

Each stub quotes the professor's instruction for its section and lists the source files.

## Schedule

| Date | Checkpoint |
|---|---|
| Fri 20 Nov | skeleton merged; decision #7 (how much of Task 4 iteration 2) answered |
| **Sun 22 Nov** | **results freeze** |
| Mon 23 Nov | all section drafts in |
| Tue 24 Nov | review of every section by a different member |
| Wed 25 Nov | assembly, page count, bibliography check, PDFs |
| **Thu 26 Nov** | **submit** |

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

The assembly issue (#14, part 3) adds how to rerun each and what needs a GPU.
