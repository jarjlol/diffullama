# GPU window runbook: K = 256 check (Sun 22 Nov 2026, written 21:45 IST)

For Arjun and Aryan (the only other people with GPU access) in case Aaditya is not at the keyboard when a GPU frees up.
Design and analysis are fixed in `../../k256_preregistration.md` (committed before any K = 256 row existed). Do not change them.
Question: do Dream-v0-Base-7B and LLaDA-8B-Base keep the "AR rerank loses to majority" result at the published budget of 256 denoising steps
(N = 10, T = 0.7, same sampler as the 64-step runs)? Questions are taken in the seeded random order `../../k256_order.txt`.

## State at 21:45 IST

| Machine | State | What is already done |
|---|---|---|
| Ada (`ssh ada`, user arjunthakur) | free of krispy, but other people's processes hold 12-16 GB | LLaDA test self-score finished (500 rows). Dream K = 256 lane is running its wait loop. |
| Blackwell (`ssh blackwell`, user antpc) | krispy's `chi144` job holds 58 GB on each GPU at 100% | Window scripts staged, queues NOT started, `STOP_gpu0/1` files present. |

Ada lane: `~/codify-ada/ss/ada_run2.sh` (pid in `pgrep -af "bash ada_run2"`), log `~/codify-ada/ss/logs/ada_run.log`.
It waits (holds no GPU memory) until 22000 MiB are free for 10 minutes in a row, then runs Dream K = 256 from the BACK of the
random order (Blackwell goes from the front, so the two never do the same question), at most 60 rows, in batches of 5 samples
(`CODIFY_SAMPLE_CHUNK=5`; the first attempt with all 10 samples at once ran out of memory because the top-p sort needs 3.6 GiB extra).
The batching changes only peak memory, not the sampling distribution. It yields at once if free memory drops under 1000 MiB.
Stop it with `touch ~/codify-ada/ss/STOP`. Resume = delete that file and start `ADA_K256_MAX=60 nohup bash ada_run2.sh >> logs/ada_run2.nohup 2>&1 &`
from `~/codify-ada/ss`. It resumes by uid, a stop loses at most the question in flight.

## Blackwell: when krispy's job ends

Krispy said he pauses after one job completes and gave no end time. The window is as long as our run, capped by us at 5 hours
(Aaditya's hard ceiling is 8 hours; 5 h gives about 100 Dream and 65 LLaDA questions, enough for the 21-24 point gap we expect to resolve).

1. Check it is really free (both lines must show no compute process, memory under 1500 MiB):
   `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`
2. Do not start unless krispy has confirmed the pause is on (the GPUs being idle for a few minutes can also just be the gap between his jobs).
   The go is Aaditya's go, or, if he cannot be reached, Arjun or Aryan after krispy has confirmed. The main session is not a go source.
3. Only AFTER step 2, declare the window and start the two queues (they wait for a GPU to be idle 5 min, which is why step 2 must come first:
   an armed queue would start by itself in a gap between krispy's jobs):
   ```
   bash ~/codify/iter2/window/start_window.sh 5
   cd ~/codify/iter2
   nohup bash queue.sh 0 >> logs/gpu0.log 2>&1 &
   nohup bash queue.sh 1 >> logs/gpu1.log 2>&1 &
   ```
   Length counts from the first real start, not from now. GPU 0 = Dream, GPU 1 = LLaDA. Generation stops 25 minutes before the end,
   the last 25 minutes score (qwen7b, gpt2, on cut and uncut text, then DLM self-score cut and trim).
4. Watch: `tail -f ~/codify/iter2/logs/gpu0.log`; rows: `wc -l ~/codify/diffullama/codification/pipeline/work/candidates/dream-base@k256/test.jsonl`.
5. Abort at any time: `touch ~/codify/iter2/STOP_gpu0 ~/codify/iter2/STOP_gpu1` (the wrapper stops the job within 30 s).
   The wrapper also yields by itself if a process that is not ours appears on a GPU. Never start anything else on Blackwell while the window runs.
6. Do not keep vLLM or any model loaded on a shared GPU while waiting for a human (standing rule).

## After the runs (CPU only, anyone)

1. Ada rows to Blackwell: copy `~/codify-ada/ss/work/candidates/dream-base@k256/test.jsonl` to
   `~/codify/diffullama/codification/pipeline/work/candidates/dream-base@k256/test.ext.jsonl`, then run
   `bash ~/codify/iter2/window/merge_ext.sh dream-base@k256` (keeps the Blackwell row on a uid clash).
   Then score the merged file on Blackwell (the queue's scoring lines do this when the window runs; if there was no window, run
   the `score_candidates.py` lines from `queue_gpu0.window.txt` by hand when a GPU is free).
2. LLaDA test DLM self-score: Ada finished all 500 rows; Blackwell has 273. The Ada copy is staged at
   `~/codify/iter2/from_ada/llada-base.test.dlmself.jsonl` (sha256 f588d09b...). Append the rows whose uid is missing in
   `work/features/llada-base/test.dlmself.jsonl`. Both come from the same model and code, only the GPU differs.
3. Analysis exactly as pre-registered: `python3 codification/pipeline/step_arm.py` (prints the K = 256 table, primary and secondary
   lines, hosts, realised n). Report per-row host/GPU and realised n. s/question comes from Blackwell rows only (Ada's gen_seconds is not comparable). T = 0.7 at K = 256 is not a like-for-like comparison with the
   published greedy 0.81; say so in the text.
4. Paper: one row pair in the step-budget table plus one sentence. Owners: Neel / Aryan (PR #47 is Aryan's).

## Rules that stay in force

- Shared machines: never hold GPU memory while idle, yield to other users, ask before anything longer than the declared window.
- The window ends at its declared length even if the results look unfinished; an unfinished subsample is reported with its n.
- Post status on issue #49 (`mesh post 49 --type note --body-file f`), not on #2/#4/#15/#17/#41 (referee loop guard).
