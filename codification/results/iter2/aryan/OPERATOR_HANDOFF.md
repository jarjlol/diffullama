# Aryan operator checkpoint 21 November 2026

User-confirmed role: `aryan/codex`, sole Blackwell GPU operator. No PRs or merges.
All new job requests come as asks on #4. Maintain comment `7048350518` in place;
`mesh status 4` cannot currently edit it because it is a note, not an active claim.
Do not claim the entire tracker or create repeated acknowledgement comments.

## Live state checked

At 01:02 IST both GPUs were at 100% load with about 6.7 GB free, occupied by the
other lab workload. No new team GPU job was launched. Existing processes were kept:

- `2409442`: `resume_at.sh 0 dream-base dream 2026-11-21 03:00`.
- `2409444`: `resume_at.sh 1 llada-base llada 2026-11-21 03:00`.
- `2447076`: `cpu_queue.sh`.

PIDs are observations, not identities to act on later. Recheck actual command lines.
`STOP_gpu0/1` remain from the agreed overnight pause. The resume scripts remove them
at the scheduled time after their idle checks, then start each lane and its queue.
Never restart healthy consumers, signal inherited/foreign processes, or modify the
live `~/codify/diffullama` checkout while a lane runs. `~/diffullama` is off limits.
Pending GPU queues contain Tier 1 cut scores, reverse assist, self-score, preservation
extraction and 256-step candidates. Preserve Tier 1 priority and one job per GPU.

The app heartbeat **Operate Blackwell iteration 2** is active every 30 minutes in
this thread (`operate-blackwell-iteration-2`). It must inspect logs, progress and
asks, update the one operator status, and continue useful work until a member says
stop. Remote resume/GPU/CPU queues do not depend on a laptop remaining connected.
The heartbeat is local to the Codex app; its scheduling is separate from those
remote processes. Keep a staffed active agent turn during Arbor review gates.

## A3 bootstrap completed

`~/arbor-rerank-iter2` is a new clean Git task, exported with `make_arbor_repo.sh`
copied from `iter2/aaditya`, outside the lane checkout. It contains Dream/LLaDA only,
200 dev and 200 gate questions per model, ordinary majority seed, and unchanged
protected evaluator. The source dev gold file covered more questions than the
generated dev items; bootstrap filters it to the existing approved 200 item IDs
rather than treating the gold-file length as the generated-subset length.

CPU eval: dev Dream/LLaDA 0.700/0.745; gate 0.710/0.735; macro 0.7225 on both.
`bootstrap_receipt.txt` records the outputs and protected-file hashes. No final-split
path was opened, no final metric calculated, no GPU or LLM loaded by the bootstrap.
Private task commits are `ea745c5` and `4899204`, with Aryan's verified identity.

This is **not ready for search**. Outstanding: complete cut AR/self-score features;
T1 plugin/patch and T4 strategies from Aalhad; T3 curated memory and N4 frozen tuned
majority/failure classes from Neel; actual team agreement before changing eval.py.
The strict exporter refuses an existing destination: use a fresh ready-task folder
or explicitly prepare the private bootstrap before launch. Do not weaken checks.

`paired_stats.py --reference FROZEN_SEED.py` now reports a separate tuned comparison,
in addition to plain majority/first. Its synthetic regression passes; the real N4
reference has not landed, so no tuned-baseline result is claimed. Gate review needs
at least 3 net questions on each non-floor model plus paired statistics. Final split
is evaluated once after a rule is frozen, with the frozen N4 bar and an ablation.

## Validation and next actions

- Existing pipeline selftest and six A3 boundary tests passed before this checkpoint.
- On Blackwell, `self_score.py --selftest` and `per_question_preservation.py selftest`
  pass with `~/venvs/scorer/bin/python`, `CUDA_VISIBLE_DEVICES=""`, offline mode and
  OMP threads 8. System Python lacks torch; use the scorer environment for these tests.
- The new paired-reference regression is queued as `a3_paired_reference_tests` in
  `cpu_queue.txt`; inspect its receipt before claiming it passed on Blackwell.
- Read `logs/resume_gpu*.log` and fresh `nvidia-smi` around the 03:00 window. Post the
  actual start/stop timestamps from receipts in the existing status comment.
- Fetch new team assets into the separate preparation workspace; never merge or edit
  the live lane checkout. Do not start vLLM before the ready task and review consumer.
- Paper freeze: 22 Nov 18:00 IST. Useful later results belong in the report.
