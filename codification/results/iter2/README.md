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
