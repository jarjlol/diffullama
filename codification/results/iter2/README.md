# Iteration 2: what runs where, and how to continue (written 2026-11-21 for the GPU operator, Aryan's agent)

Everything below is on Blackwell (login `antpc`, shared with other lab users). Code: branch `iter2/aaditya-worker`
(clone `~/codify/worker`, work dir `~/codify/diffullama/codification/pipeline/work/`, git-ignored). Rules: `codification/HANDOFF.md` §3
(never kill a process you did not start, no idle GPU holds, `HF_HUB_OFFLINE=1`, do not touch `~/diffullama`).

## 1. GPU side (`~/codify/iter2/`)

| What | Where | Notes |
|---|---|---|
| Lanes: fidelity, then K = 16 and 32 generation, then AR scoring | `lane.sh <gpu> <model> <venv>`, log `logs/gpu<N>.log`, ends with `lane gpuN DONE` | one model per GPU; resumes by question id |
| Queues, run after each lane | `queue.sh <gpu>` reads `queue_gpu<N>.txt` top to bottom; started lines move to `queue_gpu<N>.done`; log `logs/queue_gpu<N>.log` | format in the header of `queue.sh`; each line retried 8 times |
| Restart after the other lab user's job (03:00 IST) | `resume_at.sh` | |
| Stop | `touch STOP_gpu0`, `STOP_gpu1` (queues), `STOP_cpu` (CPU queue) | stop files stop work between jobs |
| Worker wrappers | `~/codify/iter2/worker/run_selfscore.sh <model> <split> <cut\|trim>`, `run_pq_extract.sh <dream\|qwen7b>` | already placed in the GPU queues |

Do not run two jobs on one GPU. Check `nvidia-smi` and the lab's other processes before anything new.

## 2. CPU side: `cpu_queue.sh` (never touches a GPU)

`~/codify/iter2/cpu_queue.txt` holds `name|inputs|command` lines. Each job waits until its input files have enough lines (or both lanes
are DONE), runs under `nice -n 10`, and writes `~/codify/worker/codification/results/iter2/cpu/<name>.txt` (command, raw output, exit code)
plus a `<name>.json` where the script produces one. A failing job is retried up to 3 times. Log: `logs/cpu_queue.log`; finished jobs:
`cpu_queue.done`. Dry run: `bash ~/codify/iter2/cpu_queue.sh --once` (prints WAIT or READY per job). Start:
`setsid -f nohup nice -n 10 bash ~/codify/iter2/cpu_queue.sh >/dev/null 2>&1` (it refuses to start twice).

## 3. Which files appear when (all under `work/`)

| File | Written by | Meaning |
|---|---|---|
| `fidelity/{dream-sampler,dream-published,llada-published}.jsonl` | lane | greedy runs at published settings, 100 questions |
| `candidates/<model>@k16/test.jsonl`, `@k32`, `@k256` | lane / queue | 500 questions (100 for K = 256), N = 10, T 0.7 |
| `features/<model>[@k]/test.<scorer>[-cut].jsonl` | `score_candidates.py [--cut]` | AR scores; `-cut` = text cut after the `#### N` line (junk fix, I-22) |
| `features/<model>/{dev,test}.dlmself.jsonl` | `self_score.py` | the diffusion model's own likelihood, feature `dlm.selfscore`, cut text |
| `features/<model>/test.dlmselftrim.jsonl` | `self_score.py --text trim` | lowest priority |
| `preservation_pq/{dream,qwen7b}.test.pt` | `per_question_preservation.py extract` | per-question hidden states |

## 4. Results files (`~/codify/worker/codification/results/iter2/cpu/`) and where they go

| Job | Output | Goes into `codification/ITERATION_2.md` |
|---|---|---|
| `cvt_*_test`, `cvt_*_dev` | cut vs trim AR selectors, paired vs majority | Results 2 |
| `self_*_dev`, `self_*_gate` | `dlm.selfscore` vs majority (dev and gate only; the final 300 are not read) | Results 4 |
| `k256_table` | K = 256 vs K = 64 vs greedy on 100 questions | Results 5 |
| `pq_analyze` | per-question preservation | Results 6 |
| `reports` | `step_arm.md`, `fidelity.md` (written one level up in `results/iter2/`) | Results 1 and 3 |

## 5. How to continue

1. Read `~/codify/iter2/WORKER_STATUS.md`, then `logs/gpu0.log`, `gpu1.log`, `queue_gpu*.log`, `cpu_queue.log`; `tail` is enough.
2. If a GPU queue runs dry before the results freeze (Sun 22 Nov 18:00 IST), ask the senior session for the next job; ideas in `ITERATION_2_PROMPTS.md` "Priority tiers".
3. When jobs finish, fill the TODO tables of `codification/ITERATION_2.md` from the result files, receipts only; leave "What this says" to a human.
4. Move code and results off the box with a git bundle (GitHub is unreliable from Blackwell, `HANDOFF.md` §3); push only `iter2/aaditya-worker`. Never commit `items.jsonl`, candidates, weights.
5. The final 300 test questions are not read by anything here. Run them once, after the senior session or Aaditya says so.

## 6. A3 (Arbor iteration 2) window and extra-credit jobs (appended 2026-11-21 by the worker acting as operator)

- Each GPU queue ends with: `SH ~/codify/iter2/a3/a3_window.sh <gpu>` (A3 window), extra-credit jobs, a second window line, more extra-credit, a third window line.
  `a3_window.sh` (source: `ops/a3/`) makes the two queues rendezvous: the first to arrive parks for 25 min, the second becomes the controller
  and runs prerequisites, vLLM (DP = 2), Arbor UNATTENDED (`interaction_mode: auto` in a copy of the config; review mode NOT used, see
  `~/codify/iter2/a3/REVIEW_MODE_NOT_USED.txt`), stops vLLM, scores a merged `solution.py` once on the final 300 (CPU, `a3_final_eval.sh`) and writes
  `window.done`. State, log and markers: `~/codify/iter2/a3/` (`a3.log`, `attempts`, `window.done|abort|gaveup`, `final_eval.done`).
  Unmet prerequisites make the window abort and the queues carry on; at most 3 windows are tried.
- Extra credit appended per GPU: step arm K = 16 and 32 on the first 200 dev questions (+ AR scoring, trim and cut), self-score trim on dev,
  256-step majority on test questions 101-200 (+ scoring).
- Matching CPU jobs (cpu_queue.txt): `step_arm_dev`, `selftrim_*_dev`, `k256_200_*`, `a3_summary`.
- Snapshots of the queues as armed: `ops/snapshot_queue_gpu0.txt`, `snapshot_queue_gpu1.txt`, `snapshot_cpu_queue.txt`.
- Dry run of the window (two queues, separate state dir, no vLLM or Arbor): the first queue parked, the second became controller, all steps ran
  in order, the evaluation on the gate split (400 items) ran, `window.done` written, the parked queue was released.
- Not tested: the real vLLM start (`~/codify/start_vllm.sh`) and a real Arbor run under this controller; they are the same commands as iteration 1.

## 7. Handover to the GPU operators (Aryan's and Arjun's agents), 21 Nov 2026 16:00 IST: read this first

**What happened.** GPU 0 fell off the bus at 06:58 (Xid 79: `journalctl -k | grep -i xid`). CUDA stopped working on both GPUs, every lane and queue line then failed 8 times and was dropped (the old queue consumed ~8.5 h of lines). Aaditya rebooted Blackwell at ~15:30; both GPUs and CUDA work again (a matmul test on each). The queues were rebuilt in priority order and restarted at 15:34. Nothing was lost: every line consumed after 06:58 is back in the rebuilt queues (checked by script).

**Fault guard (new, `~/codify/iter2/queue_body_v3.sh`, copy in `ops/`).** After a job FAILS, the queue runs a 5-second CUDA matmul on its GPU. If that fails, or the job's output contains `CUDA unknown error`, `unspecified launch failure`, `Xid`, `fallen off the bus` or `Unable to determine the device handle`, the queue puts the job line BACK at the top of `queue_gpu<N>.txt`, touches `STOP_gpu<N>`, writes `~/codify/iter2/GPU_FAULT_gpu<N>.txt` (time, reason, last error lines) and EXITS without consuming more lines. Same if `nvidia-smi` cannot see the GPU while the queue waits for it. Tested with fake failing jobs and fake failed health checks (`ops/test_queue_guard.sh`: ordinary failure -> 8 retries, line recorded, next job runs; failed check -> line back at top, STOP file, fault file; CUDA error text -> fault even if the check passes; real check on GPU 1 passes in ~3 s).

**If you find `GPU_FAULT_gpuN.txt`:** (1) `nvidia-smi -L` and `journalctl -k | grep -i -E "xid|nvrm" | tail`; (2) if the GPU is missing or CUDA fails, the box needs a reboot: `antpc` has no passwordless sudo, so ask Aaditya (or whoever has the password) and tell the lab users first; (3) after the reboot: `nvidia-smi -L`, a one-line torch matmul per GPU, then `rm ~/codify/iter2/STOP_gpuN ~/codify/iter2/GPU_FAULT_gpuN.txt` and `cd ~/codify/iter2 && setsid -f nohup ./queue.sh N >/dev/null 2>&1 </dev/null`. The job that failed is already the first line. (4) if the fault is not a GPU fault (ordinary bug), the guard does not trigger: the job is retried 8 times and recorded with rc=1 in `queue_gpuN.done`; re-add the line by hand after fixing it.

**State at 15:55 IST.** `queue.sh 0` (pid 6926) and `queue.sh 1` (pid 6928) were running the old body; they hand over to v3 by themselves when their current job ends (`swap_to_v3.sh`, log `logs/swap_to_v3.log`; during the hand-over `STOP_gpu0/1` exist for a short time, that is expected, do not delete them before `logs/swap_to_v3.log` shows both "swapped"). `cpu_queue.sh` (pid 6931) runs CPU jobs. GPU 0 = Dream, GPU 1 = LLaDA. Logs: `logs/queue_gpu0.log`, `queue_gpu1.log`, `cpu_queue.log`; finished lines: `queue_gpu*.done`, `cpu_queue.done`. Queue snapshot: `ops/snapshot_*` (older, rebuilt at 15:34).

**Operator rules (unchanged, from `HANDOFF.md` §3).** One operator per GPU: agree on `#4` who takes GPU 0 (Dream, with the CPU queue) and who takes GPU 1 (LLaDA) before touching a queue; suggested split: aryan/codex GPU 0 + CPU queue, arjun/opencode GPU 1. Never kill or signal a process you did not start (stop with `STOP_gpuN`). Do not touch `~/diffullama`. `HF_HUB_OFFLINE=1`. Every new line must be resumable by uid and tested (`--limit 5` or a mock) first. No idle holds. Add jobs by APPENDING lines to `queue_gpu<N>.txt` (format in the header of `queue.sh`: `<venv> script args` or `SH script args`); short scoring jobs go into `a3/batch.sh <listfile>` lines so the queue does not wait 2 minutes between them.

**What is queued, in order (per GPU).** Tier 1: finish LLaDA K = 32 generation, score K = 16/32, cut re-scoring of the iteration-1 candidates, a `lane gpuN DONE` marker (starts the `reports` CPU job). Tier 2: self-score on cut text (test, dev), Dream preservation extraction. Then the A3 window (`a3/a3_window.sh <gpu>`: the two queues rendezvous, vLLM DP=2, Arbor unattended in `auto` mode, final-split evaluation once; a window that STARTS after Sun 22 Nov 14:00 IST is labelled post-freeze, report only; see section 6), then 256-step runs, dev step arm, trimmed self-scores, more A3 attempts.

**Estimated times (unverified, IST).** Tier 1 done ~16:30 today; self-scores 17:30-18:30; A3 window ~18:30-22:30; 256-step runs done 02:00-04:30 Sunday; extra credit 08:00-14:00 Sunday. **Paper results freeze: Sun 22 Nov 18:00 IST.** Measured: Dream K=16 9.4 s/question, LLaDA K=32 26.6 s/question, self-score (cut text) 3.9 s/question for Dream.

**What still needs people.** Fill the TODO tables of `codification/ITERATION_2.md` from `results/iter2/cpu/*.txt` (receipts only), write "What this says", move code and results to git by bundle (`HANDOFF.md` §3, push only `iter2/aaditya-worker`, never `items.jsonl`, candidates, weights). CPU job requests (like Neel's N4): append to `cpu_queue.txt` (`name|inputs|command`, see section 2).
