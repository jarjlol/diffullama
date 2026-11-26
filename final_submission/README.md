# Final Submission (Task 5): research paper, report and repo

**Deadline: Thursday 26 Nov 2026, 13:00 IST.** Tracker: epic #17; milestone "Task 5: Final Submission". The upload itself is the human step #16.

The folder named **Final Submission** in our group folder holds exactly three items:

1. the **paper**, 4 pages of content plus 1 reference page ("4+1(reference) page paper"): canonical built PDF at `paper/paper.pdf` from editable IEEEtran journal LaTeX sources in `paper/latex/` (4 body + 1 reference page, 23 verified references; rebuild with Tectonic or `pdflatex`+`bibtex`; earlier Markdown build retained in `paper/sections/` and `paper/build.py`);
2. the **report**, 2 to 4 pages, observations and efforts for each task: `report/report.md` and built 3-page `report/report.pdf` (issues #15, #78);
3. the **GitHub repo** of the pipeline and the code behind the paper: this repo (see "Pipeline code" below).

Frozen build, checked against `main` on 26 Nov 09:20 IST: `paper/paper.pdf` is **5 pages** (4 body + 1 reference; last PDF commit `4bc968a`), `report/report.pdf` is **3 pages** (last PDF commit `01792b3`), neither PDF contains team member names or email addresses, and every SHA-256 row in the reproducibility table below resolves at its stated commit (`git show <commit>:<path> | sha256sum`). The canonical paper source is `main`'s `paper/latex/`; branches `mesh/75-latex-paper` and `mesh/76-decision-paper` are superseded and kept only for history.

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

The label `to:<handle>` on each issue is authoritative if this table is stale. khushi does not use an agent; diya's Claude Code agent joined on 21 Nov for reviews and non-GPU tasks. Agent start-up prompts per member: `AGENT_PROMPTS.md`.

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
- The paper is **4 body pages + 1 reference page** (the professor's 25 Nov announcement, "4+1(reference) page paper", relayed on #69; this replaces our earlier reading that references were uncapped). `build.py` therefore leaves the provenance appendix out of the PDF by default. The appendix stays in the repo at `final_submission/paper/appendix_provenance.md`, and `build.py --with-appendix` renders it. Cite what the text needs and verify every entry.

## Pipeline code (for deliverable 3)

How the pipeline ran — what was manual, what was agent-assisted, how the work was coordinated, and the published methods it used: [docs/pipeline-story.md](../docs/pipeline-story.md).

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
| Codification, Task 4 (`codification/`, branch `assignment/codification`) | GPU side, from `codification/pipeline/` (one model per process, one venv per model family): `prepare_data.py` → `generate_candidates.py` → `score_candidates.py` → `assemble.py` → `measure_preservation.py` (`codification/README.md` §B). CPU side, from the repo root, with cached items: `python3 codification/rerank_task/baselines.py --split test`, `python3 codification/pipeline/analyze.py --split test`, `python3 codification/paired_stats.py <task_dir> <split> <solution.py>` (`codification/HANDOFF.md` §3). Arbor: `codification/HANDOFF.md` "Rerun Arbor" | **Yes** for generation, scoring, hidden states and the Arbor LLM (both GPUs of the Blackwell workstation); **no** for selectors, statistics and the wiring test | `python3 pipeline/selftest.py`: **SELFTEST PASSED** at 6ffafa5, 2026-11-20 (mock models, CPU) |

### Reproducibility: the artifacts behind the paper's numbers

SHA-256 of the committed result files the paper's tables are drawn from (computed with `git show origin/<branch>:<path> | sha256sum` on 2026-11-21; in a fresh clone the branch exists only as `origin/<branch>`; they match `final_submission/paper/first_draft/SOURCES.json` on branch `iter2/aryan`):

| Artifact | Branch | SHA-256 |
|---|---|---|
| `codification/results/iter1/numbers/baselines_final300.json` | `assignment/codification` @ 6ffafa5 | `14418cab14bda4c1f244a03b267677ab3aa7b61fd36a9489466458e74ce7c9be` |
| `codification/results/iter1/numbers/paired_final300.txt` | same | `f0d48567843ecf7a58db1f204fc0a39a150b22cb8959fbf3ede5ced97ab948f3` |
| `codification/results/iter1/numbers/analyze_test500.txt` | same | `d60d98eb5d974a75dfac3eea550159b5ab94e13430c182516ecb410ecae18c9d` |
| `codification/results/iter2/aryan/fidelity_complete.json` | `iter2/aryan` | `a45661b442c6f22c3e5c8d7a331d3dbc82d70983a20141ca2f2ebd290efb52a4` |
| `codification/ITERATION_2.md` | `iter2/aaditya-worker` @ d1a8c15 (PR #58) | `516fb3c37cb598a083af98492a1d365a0cf5f6cf905e83e40e125f425093f58a` (iteration-2 write-up; §5 is the 256-step check) |
| `codification/results/iter2/k256_preregistration.md` | same | `f91bd7d547e7c605f70fc9a072dbd39fcdaa60603d24abef55b976ed5e01329e` (design and stopping rule, written before any 256-step row) |
| `codification/results/iter2/k256_step_arm_output.txt` | same | `b6c80046d703643750b2f9c4e9efe27fac52dac70bfc1f29c1e38ef8ce837fa4` (the single analysis run behind Table 2's K = 256 rows) |
| `codification/results/iter2/k256_results.md` | same | `8e3934a14e2cf32d85bb90dcec004c1fd9075b5e538f1e044cdc6f37f355d9a3` (256-step write-up) |
| `codification/results/iter2/step_arm.md` | same | `19f5725ed5237810f625d38bc0b01a071d136d0923ff525f3813c9653ae99714` (Table 2, K = 16/32/64) |
| `codification/results/iter2/cpu/cvt_dream_test.txt` | same | `97ec796a3b0ebc1fe179bd8f83423ed2f18656f3cb71c1b963849e11a037b294` (cut re-score, Dream (try 2)) |
| `codification/results/iter2/cpu/cvt_llada_test.txt` | same | `703de3dbc91c541bb8220ebf1c09fc0753a4a0968b889158e28ed0fecaa2df02` (cut re-score, LLaDA (its self-score row is 16/500, unused)) |
| `codification/results/iter2/cvt_llada_test_full500.txt` | same | `efb1f85b2a7d5fd61504669064d9b3b81c94091c2b03a5ac8d32396099e037f5` (LLaDA self-score, 500/500) |
| `codification/results/iter2/cpu/self_dream_gate.txt` | same | `2bc62e9cf4a5f54dec3d045c24620104e25bb9a8c0571df81b8b8d66d0811ed5` (self-score, Dream gate) |
| `codification/results/iter2/cpu/self_llada_gate.txt` | same | `a3d5d530c85c4352661afd16dc930e523b4e186edcca6305528b88b0374dea79` (self-score, LLaDA gate) |
| `codification/results/iter2/cpu/pq_analyze.txt` | same | `c8a9b25b2a7a2bb05688b8304d08b18708c91bbf46794a0b11d3c122f654e57c` (per-question preservation test) |
| `codification/DECISION_RERANK_PREREG.md` | `iter2/aaditya` @ 6e0e9de (PR #76) | `e764deb17cb9cf90477abef6b931d15924b808637fd524243ba146ffa943e0b4` (pre-registration) |
| `codification/DECISION_RERANK_METHODS.md` | same | `87dd09fe5f62d24f0c3ae3d9de4bbf118fa947f624c4d49a78fc29a76466bb09` (methods, rates and post-review changes) |
| `codification/DECISION_RERANK_RESULTS.md` | same | `34b88e92177af2df0663cf8e993803b4b632d423f909296de25238786daad2fc` (results and post-review corrections) |
| `codification/results/gsm8k_preds.jsonl` | same | `2f21d3da57a559190064c613b50705f7b54524c52f08f6b00b645f5be5741b21` (GSM8K K=64 per-item predictions, 1,000 rows) |
| `codification/results/humaneval_decision.json` | same | `e9271a545a7450091eda52fb9211ed9abe5269367f68ac524369fb14cfebe778` (HumanEval summary, fixed docstring parser + MBR-exec) |
| `codification/results/humaneval_preds.jsonl` | same | `a1d558b2a043b0fa4de018d6d52abb2c8ac37a5ed4f7544e50aca8a7ddcbe04f` (HumanEval per-item predictions with `has_examples`, 144 rows) |
| `codification/results/controls.json` | same | `d685b8b890fa58fb0ac9d0b8029a425ed65857716914ffa102d256ebd403880d` (controls summary; exact gold-injection chance 12.9%) |
| `codification/results/receipts/README.md` | same | `fc20405a609f9c17e0900a4319f9e7e717dfcc4e1f55cf2765c9000348cf9c9c` (index of per-item control JSONL receipts in `results/receipts/`) |
| `codification/results/k256_decision.json` | same | `a5c205c00b8ff480c1f06896ea8ab3b243017ccc58def47eb3f5bf0ddcc9b648` (K=256 step sensitivity summary, #80) |
| `codification/results/k256_preds.jsonl` | same | `03da6b444888f10d527264edf7fa7a842814bb019578c2cfe701dc6dbcbe804a` (K=256 per-item predictions, 377 rows) |
| `codification/results/k256_nsweep.json` | same | `f4976394d8c5116b313dcd64d4c4a62ebe59a0aa99faa2a98744bf47c4a49e88` (K=256 N-sweep, #83) |
| `codification/results/humaneval_plus_result.json` | same | `f11f9ecd06ec1fb338589eccafb7f00b5171932a5af36f603ee217246d5a7549` (HumanEval+ re-execution, n=142, #84) |
| `codification/results/humaneval_recognition_result.json` | same | `884233035c932ce2b072f40a2b104ce79e3be7c774250f38783b5eed69574c6a` (HumanEval recognition probe, 59/60, #81a) |
| `codification/results/humaneval_recognition_nodoc_result.json` | same | `f67d8f8869fce4c9f4921ed8304985ca38d268c8f7a98f38a9a2b401846fae27` (HumanEval no-docstring control, 16/20, #81a) |
| `codification/results/weak_vote_threshold.json` | same | `0b650d601bae044f98f04f9664b93577096cbd57347475ffce7ce6dbef297d89` (dev-tuned weak-vote hybrid rule T=5, #81b) |

Iteration-2 rows added 24 Nov (same `git show … | sha256sum` method; `cpu/*` digests are byte-identical to the report's `report/evidence/iter2/` copies). GPU time, from the records (not per-job accounting):
- **Iteration 1:** about 34 GPU-hours, a planned estimate, not measured (`codification/ITERATION_1.md` run plan, branch `assignment/codification`).
- **256-step check:** two Blackwell windows on Mon 23 Nov, 08:52–16:21 and 17:08–19:26 IST, on both GPUs, give about 19.6 GPU-hours. A shared Ada GPU added 23 Dream and 41 LLaDA rows (`codification/results/iter2/k256_results.md` and `ops/window/RUNBOOK.md` on branch `iter2/aaditya`; window times from #51). This is reserved window time, not busy time.
- **Lost:** about 8.5 h of GPU time (06:58 to ~15:30 IST, Sat 21 Nov) after a GPU fault (Xid 79). The failed queue lines were re-queued, and no results were lost (`codification/results/iter2/README.md` §7, branch `iter2/aaditya`). Another lab user then held both GPUs from Sat 21 Nov ~17:02 IST (#4) to Mon 23 Nov 08:00 IST (#49).
- **Not timed:** the other iteration-2 GPU jobs (fidelity, step arm K = 16/32, self-score) were not timed per job. Their s/question figures are in `final_submission/report/evidence/iter2/step_arm.md`.

Iteration 1's numbers are committed under `codification/results/iter1/numbers/`; reproducing them on CPU needs the cached `items.jsonl` files (git-ignored, shared on the team Drive). Iteration 2: `codification/ITERATION_2.md` on branch `iter2/aaditya-worker` (into `iter2/aaditya` via PR #58).
