# Codification handoff (written for Arjun and Neel on 2026-11-05; for the whole team since 2026-11-06)

Everything needed to carry Task 4 (codification) on from here. Read this first, then
`ITERATION_1.md` (what was done and found) and `NEXT_ITERATION.md` (what to do next).
Branch: `assignment/codification` on GitHub. Everything below is pushed; nothing lives only
on Aaditya's laptop any more.

**Iteration 2 (added 2026-11-06).** Every member has a task. Start your agent with prompt T0 in
`ITERATION_2_PROMPTS.md`, then run your own prompts from the table at its top. The design for
iteration 2, which borrows mechanisms from DeepScientist, EvoScientist and AutoSOTA around Arbor,
is `BORROWED_IDEAS.md`. A plain-language history of the whole project is `../project-docs/timeline.html`.
A second, independent review of what to borrow (`BORROWED_IDEAS_REVIEW.md`, 2026-11-06) reread the
papers and Arbor's v0.1.4 code. Read its §1 before A3, because it found four things that change the
Arbor setup:
- Arbor's own iteration-1 memory recommends the banned cross-model cache.
- `eval.py` does not stop cross-item state.
- `merge_threshold` is not enforced at the merge gate.
- Arbor's native guards (protected paths, plugin, plateau detector) were all off in iteration 1.

Original split (2026-11-05): **Arjun** owns anything that needs the Blackwell workstation (he has
access, Neel does not). **Neel** owns the write-up, the submission and the iteration-2 plan,
which need only this repo. Aryan also has workstation access and takes the GPU prompts A1-A3.

## 1. Where things stand

Iteration 1 is complete: run, written up and pushed. Due Thu 2026-11-05.

- **Idea 1.1 is refuted as specified.** Picking the candidate an AR model scores as most
  likely loses to majority vote by 20+ points on Dream and LLaDA, even with each model's own
  7B AR base as scorer. Numbers: `ITERATION_1.md` §Results.
- **Arbor** (run `iter1b`, 68 min, Qwen3.8-27B-FP8 on local vLLM) tried 4 directions and
  merged one, verification-weighted voting (4.1). On the 300 final questions Arbor never
  saw, it ties majority vote: Dream −0.3 points (CI [−2.0, +1.3]), LLaDA 0.0, McNemar p = 1.0.
- **CKA vs headroom:** ρ between +0.4 and +1.0 across scorers, p ≥ 0.083 with n = 4 models,
  set by Dream alone. Descriptive only.
- **Spec gaps:** I-1 to I-20 in `ITERATION_1.md` §Issue log.
- **Feedback to the model:** every reply given to Arbor, verbatim, in `ITERATION_1.md` §Arbor.

What is weak, and the ranked plan for iteration 2: `NEXT_ITERATION.md`. The cheapest
high-value items are `dlm.selfscore` (the DLM's own likelihood as a reranker), a
reward-model or published DLM selection method as the SOTA comparison, and a novelty
check (I-6).

## 2. Evidence in this repo (`results/iter1/`)

| Path | What it is |
|---|---|
| `solution_merged_4_1.py` | The selector Arbor merged (Arbor repo `master` = `fc998f4`). The repo's `rerank_task/solution.py` is still the seed (GPT-2 NLL), on purpose |
| `arbor_session/` | Arbor's session folder for `iter1b`: `REPORT.md`, `COORDINATOR_FINAL_REPORT.txt`, the idea tree (`.coordinator/idea_tree.md`), each experiment's diff, metrics and report, and the raw event log |
| `arbor_results/` | Arbor's own summary of the merged idea |
| `arbor_git_log.txt` | Commit history of the Arbor repo |
| `numbers/baselines_{dev,test500,final300}.json` | Every selector's accuracy per model (`baselines.py --json`) |
| `numbers/paired_{dev,gate200,final300}.txt` | `paired_stats.py` output for 4.1 vs majority and vs first |
| `numbers/analyze_test500.txt`, `preservation.json` | CKA per model (four variants) and the CKA-vs-headroom table |
| `ops/` | Scripts that ran on Blackwell outside the repo: venv setup, model prefetch, vLLM start, Arbor start, review-reply helpers |

Splits: dev = the first 200 of the 500 shuffled GSM8K-train questions in
`rerank_task/data/dev`. Test = the first 500 of the official 1319, split by
`split_gate_final.py` into gate 200 (Arbor's merge check) and final 300 (never shown to
Arbor). "test500" means all 500.

## 3. The Blackwell workstation (Arjun)

**Access.** Same jump-host route Arjun already uses; login `antpc`. This account is shared:
other projects run under it too (on 2026-11-05 a `specpower` vLLM held GPU 0 under
`antpc`). Check what a process is before assuming it is ours.

**Hard rules.**
- Never kill or signal a process you did not start. Check `nvidia-smi` and
  `ps -o user,etime,cmd -p <pid>` first.
- Do not touch `~/diffullama`: it holds a teammate's ~1000 lines of uncommitted edits. Our
  clone is `~/codify/diffullama`.
- Never leave a GPU server (vLLM, a loaded model) idle on the shared cards while waiting on
  a decision. vLLM sat idle on GPU 1 for 8 h on 2026-11-03. Start it in the same step as
  its consumer and stop it when the job ends.
- The lab network intercepts TLS (Sophos). Do not disable certificate checks. Hugging
  Face is unreachable from the box: models were downloaded on a laptop and copied over by
  scp, then checked by sha256. Run everything with `HF_HUB_OFFLINE=1`.

**Layout (home `~antpc`).**

| Path | What |
|---|---|
| `~/codify/diffullama` | Our clone of this repo. GitHub is unreliable from the box, so it was synced with git bundles: `git bundle create x.bundle assignment/codification` on a laptop, scp it, then `git fetch ~/x.bundle assignment/codification` and fast-forward |
| `~/codify/diffullama/codification/pipeline/work/` | 4.1 GB, git-ignored: `candidates/` (N = 10 samples per question per model), `features/` (AR scores), `hidden/` (CKA activations), `preservation.json` |
| `~/codify/campaign/` | Campaign supervisor state (`status.json`, `campaign.log`, `results/`). Touch `STOP` to pause it, `ABORT` to stop it. It is not running now |
| `~/codify/*.sh`, `arbor_*.py` | Same as `results/iter1/ops/` |
| `~/arbor-rerank/` | The Arbor task repo. `master` = merged 4.1. Session in `.arbor/sessions/iter1b/` |
| `~/arbor-rerank-final/` | The 300 final questions (`items.jsonl`, `gold.jsonl`), kept outside Arbor's repo |
| `~/arbor-rerank-eval/` | Copy of the Arbor repo with the final split in `data/test`, for `paired_stats.py` |
| `~/venvs/{anchor,dream,llada,scorer,vllm,arbor}` | One venv per model family (incompatible transformers pins; see `README.md` §B), plus vLLM 0.31 and Arbor 0.1.4 |

**Hardware quirks.** GPU 0 drives the display; its watchdog sometimes kills long kernels
("launch timed out"), so generation jobs retry up to 8 times and resume. Tensor-parallel
vLLM (`--tensor-parallel-size 2`) hangs on this box; use `--data-parallel-size 2`.

**Reproduce the numbers** (CPU only, a few minutes):

```bash
cd ~/codify/diffullama/codification/pipeline
python3 ../rerank_task/baselines.py --split test         # all selectors, 500 test questions
python3 analyze.py --split test                          # CKA vs headroom
python3 ../paired_stats.py ~/arbor-rerank-eval test ~/arbor-rerank/solution.py   # final 300
```

**Rerun Arbor** (needs both GPUs for about an hour; check they are free first):

```bash
bash ~/codify/start_vllm.sh                  # about 3 min to load; tail ~/codify/vllm.log
tmux new -s arbor 'bash ~/codify/run_arbor.sh'
# review prompts: python3 ~/codify/arbor_pending.py ~/arbor-rerank/.arbor/sessions/<run>/events.jsonl
# reply:          echo "/approve" | bash ~/codify/arbor_reply.sh
kill -- -$(ps -o pgid= -p $(cat ~/codify/vllm.pid) | tr -d ' ')   # stop vLLM when done
```

Change `--run-name` in `run_arbor.sh` for a new run. Arbor lessons that cost time in
iteration 1:
- vLLM needs `--max-model-len 131072`, because Arbor asks for 16k output tokens per call.
- Qwen3 emits XML tool calls, so serve with `--tool-call-parser qwen3_xml`.
- Answer review gates with `/approve`, `/skip`, `/edit <note>` or `/answer <text>`. Plain
  text goes to Arbor's helper LLM, which is slow and once never submitted.
- At the idea gate, free text *replaces* the hypothesis. At the pre-execution gate it is a
  note.
- Depth-2 nodes are the concrete experiments under a depth-1 direction, so a child that
  restates its parent is not a duplicate.
- Each unanswered prompt waits 15 minutes before timing out.
- Arbor's merge gate compares against the trunk (the seed), not against majority vote.
  For iteration 2, make the seed `solution.py` majority vote so a merge means "beats
  self-consistency".
- The gate merges on any strict held-out gain. `merge_threshold` is only a hint in the
  coordinator prompt (Arbor v0.1.4 `src/coordinator/tools/git_ops.py`). The reviewer
  enforces any margin.
- `arbor run ... --yes` does not reuse earlier runs' `EXPERIENCE.md` (recall happens only in
  the intake chat). Pass memory in explicitly, and never the raw iteration-1 file
  (`BORROWED_IDEAS_REVIEW.md` §3).

## 4. Other machines and clean-up

- **ada** (Arjun's RTX 5000 Ada, user `arjunthakur`): `~/codify-ada` holds about 25 GB of
  venvs, HF cache and Dream generation outputs, all already merged into Blackwell's `work/`.
  Arjun can delete it whenever he wants the space back.
- **Aaditya's laptop**: a 28 GB model download cache, not needed by anyone else.

## 5. Submitting iteration 1

The deliverable is `ITERATION_1.md`, with `results/iter1/` as evidence and
`NEXT_ITERATION.md` as the self-assessment and plan. The brief's five parts map to sections
in the table at the top of `ITERATION_1.md`. If the submission needs a PDF, export
`ITERATION_1.md`; the tables are plain Markdown.

## 6. Agents working on this

AI coding agents read `AGENTS.md` at the repo root, which points here. Verification rules
(primary sources only, and never call a claim verified unless checked) are in
`docs/verification-standards.md`. They apply to the novelty check and to any published
number added for the SOTA comparison.
