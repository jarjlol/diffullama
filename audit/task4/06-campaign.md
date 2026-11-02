# 06 — The GPU campaign on Blackwell

`codification/campaign/campaign.py` runs on the workstation under `setsid nohup`, from the clone
`~/codify/diffullama`. It started on 2026-11-02 at 12:26 UTC (pid 924187) and does not depend on
anyone's laptop or session.

## What it waits for

User `atharva` holds both cards with a vLLM benchmark sweep: `run_cell.py --gpus 0,1`, Llama-2-13B,
about 90 GB per card. The sweep restarts its server between cells, so a card can look free for a few
seconds between runs. Launching then would starve the next cell. The rule is therefore:

> A GPU is usable only after **20 consecutive minutes** in which it has no process that is not ours,
> less than 2,000 MiB in use that is not ours, **and** no vLLM, sglang, TGI, torchrun or deepspeed
> process of anyone else is alive anywhere on the machine.

While the sweep's driver (`run_cell.py`, which has `vllm` in its path) is alive, nothing launches,
even between cells.

## Guarantees towards other users and the machine

- **It only signals its own processes.** Jobs start in their own session (`start_new_session`).
  ABORT kills only those process groups. Nothing else is ever signalled.
- **One job per GPU**, pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES`, so CUDA
  numbering matches `nvidia-smi`. DiffuLLaMA's `device_map="auto"` cannot spread onto the other card.
- **Re-checks before every job.** If another user starts on a card mid-campaign, that card's 20-minute
  clock resets and no new job of ours goes there. Running jobs are not killed; they finish.
- **Writes** only under `~/codify` and adds to the shared Hugging Face cache. It never touches
  `~/diffullama` or `~/venvs/{dl,vllm}`.
- **Will not launch** with less than 150 GB of free disk. 2.8 TB was free at start.
- Model prefetch runs at `nice 10`. GPU jobs run at normal priority. Raising our priority above
  other users' would need root and would break the first rule.
- `OMP_NUM_THREADS=8` per job on a 128-core machine.

## What it runs, in priority order

1. **Smoke**: 20 dev questions per model, cheapest first (DiffuGPT-S, -M, then the 7B models).
2. **Gate** (CPU, instant). It needs ≥ 50% of candidates with an extractable answer and ≥ 2 distinct
   candidates per question. It logs first-sample and oracle accuracy and seconds per question. Low
   accuracy is logged as a finding, not a failure. A failed gate skips that model's remaining jobs.
3. **Small models end to end**: DiffuGPT dev/test, scoring with three AR scorers, CKA hidden states.
   This gives a complete pipeline result within about an hour of the cards freeing.
4. **7B models**: DiffuLLaMA, Dream and LLaDA dev (200) and test (500, adaptive), then their scoring and CKA.
5. After each model's scoring completes: `assemble.py` (fully scored models only), then
   `baselines.py` into `~/codify/campaign/results/`. After CKA: `measure_preservation.py` and `analyze.py`.

**Adaptive test size.** When a 7B test job launches, the measured seconds per question from the
gates is used to project the remaining 7B generation. If it does not fit before **Wednesday 22:00
IST**, the test subset drops to 300, then 200. The decision is logged.

Every job is resumable. A failed job is retried once and then marked failed. A restarted
supervisor reads `done.json` and continues.

## Control

```bash
ssh jump-blackwell
cat ~/codify/campaign/status.json          # GPUs, what it waits on, running jobs, gates, s/question
tail -f ~/codify/campaign/campaign.log      # LAUNCH / DONE / GATE / RETRY / FAILED / REPORT events
ls ~/codify/campaign/results/               # baseline tables as models finish
touch ~/codify/campaign/STOP                # finish running jobs, start nothing new, exit
touch ~/codify/campaign/ABORT               # SIGTERM our running jobs, then stop
cd ~/codify/diffullama/codification/campaign && setsid nohup python3 campaign.py >/dev/null 2>&1 &   # resume
```

## Known limits

- If the sweep runs for days, iteration 1 has no 7B results. Fallbacks: ask atharva for a window, or
  run the DiffuGPT models on the idle ada box (RTX 5000 Ada, 32 GB), which needs its own setup.
- Arbor is **not** in the campaign. It needs a vLLM server for its LLM and a person in `review` mode.
  Start it by hand once `results/` has test tables (README §C).
- The time estimates are paper estimates (2 × params × tokens × steps at ~150 TFLOPS effective) until
  the gates measure real seconds per question.
