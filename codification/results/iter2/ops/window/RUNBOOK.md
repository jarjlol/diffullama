# GPU window runbook: K = 256 check (Sun 22 Nov 2026, written 21:45 IST)

For Arjun and Aryan (the only other people with GPU access) in case Aaditya is not at the keyboard when a GPU frees up.
Design and analysis are fixed in `../../k256_preregistration.md` (committed before any K = 256 row existed). Do not change them.
Question: do Dream-v0-Base-7B and LLaDA-8B-Base keep the "AR rerank loses to majority" result at the published budget of 256 denoising steps
(N = 10, T = 0.7, same sampler as the 64-step runs)? Questions are taken in the seeded random order `../../k256_order.txt`.

## State at 22:15 IST

| Machine | State | What is already done |
|---|---|---|
| Ada (`ssh ada`, user arjunthakur) | free of krispy, but other people's processes hold 12-16 GB | LLaDA test self-score finished (500 rows). Dream K = 256 lane is running its wait loop. |
| Blackwell (`ssh blackwell`, user antpc) | krispy's `chi144` job holds 58 GB on each GPU at 100% | Window of 8 h declared, both queues running and waiting (no GPU memory held), nothing starts before Mon 08:00 IST. |

Ada lane: `~/codify-ada/ss/ada_run2.sh` (pid in `pgrep -af "bash ada_run2"`), log `~/codify-ada/ss/logs/ada_run.log`.
It waits (holds no GPU memory) until 22000 MiB are free for 10 minutes in a row, then runs Dream K = 256 from the BACK of the
random order (Blackwell goes from the front, so the two never do the same question), at most 60 rows, in batches of 5 samples
(`CODIFY_SAMPLE_CHUNK=5`; the first attempt with all 10 samples at once ran out of memory because the top-p sort needs 3.6 GiB extra).
The batching changes only peak memory, not the sampling distribution. It yields at once if free memory drops under 1000 MiB.
Stop it with `touch ~/codify-ada/ss/STOP`. Resume = delete that file and start `ADA_K256_MAX=60 nohup bash ada_run2.sh >> logs/ada_run2.nohup 2>&1 &`
from `~/codify-ada/ss`. It resumes by uid, a stop loses at most the question in flight.

## Blackwell: armed for Mon 23 Nov 08:00 IST (started Sun 22:11 IST by the worker session on Aaditya's instruction)

Aaditya's decision: window of 8 hours at most, krispy allowed our runs from 08:00 IST Monday, everything essential finished well before the 8 hours are used,
and the GPUs go back to krispy at once if he needs them earlier.
Both queues (`queue.sh 0` = Dream, `queue.sh 1` = LLaDA) are RUNNING NOW and hold no GPU memory. Two gates keep them from starting early:
the queue's own "GPU busy" wait, and `WINDOW_NOT_BEFORE` (= 08:00 IST Mon) in `tb.sh`, which also needs a GPU idle for 5 minutes. The 8 hours count from the
first real start (stamped in `WINDOW_START`), so a late start does not shorten the run. Timeline from a start at 08:00:
- 08:00-13:30 generation (stops itself after 5 h 30 min), then scoring (qwen7b, gpt2 on cut and uncut text, then DLM self-score cut and trim) with the rest of the 8 h as buffer.
- Essential outputs (K = 256 rows and `ar.qwen7b-cut.sum` scores for the primary test) should be done by about 14:00; the ceiling is 16:00.

What to check and do:
1. Before 08:00, nothing to do. Look at `tail ~/codify/iter2/logs/queue_gpu0.log ~/codify/iter2/logs/queue_gpu1.log` and `~/codify/iter2/logs/tb_gpu0.log` after 08:00.
   `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` shows whether krispy's job is still on.
2. SURRENDER at any time, e.g. krispy asks for the GPUs: `touch ~/codify/iter2/STOP_gpu0 ~/codify/iter2/STOP_gpu1` (the wrapper stops the job within 30 s and the queue ends).
   The wrapper also yields by itself if a process that is not ours appears on a GPU. Rows already written stay valid (one row per finished question).
   The scoring lines also need a GPU; if we surrendered before scoring, either ask krispy for a short scoring slot (about 20 min) or score the finished rows on Ada
   (copy the rows to Ada). Checked Sun 22 Nov 22:25 IST, CPU only: Ada's `~/codify-ada/venv-dream` (torch 2.7.1, transformers 4.46.2) runs `score_candidates.py --scorer mock` on a 2-row copy in a scratch
   dir, so the code path works. NOT usable yet for the real scorers: Ada's model cache (`~/codify-ada/hf`) holds Dream and LLaDA only, no Qwen2.5-7B and no GPT-2, and HF is offline there. Using Ada for
   scoring first needs a download of Qwen2.5-7B (about 15 GB; the disk has 384 GB free but is 90% full) by Arjun or whoever agrees to it.
3. To cancel the whole thing: `touch` the two STOP files, then `pgrep -af "bash queue.sh"` and `kill <pid>` for our two queue pids only (a `pkill -f` over ssh matches the ssh shell's own command line and kills it; `cpu_queue.sh` is a different job).
4. Do not start anything else on Blackwell during the window. Do not keep vLLM or any model loaded on a shared GPU while waiting for a human.
5. Watch rows: `wc -l ~/codify/diffullama/codification/pipeline/work/candidates/dream-base@k256/test.jsonl` (and `llada-base@k256`).
   To change the length, use `start_window.sh` again only if no window job has started (it removes WINDOW_START). Never extend a window that has started.

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
