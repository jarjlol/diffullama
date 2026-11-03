"""Second GPU for one long generation job: run the same split in reverse on the other card.

    setsid nohup python3 assist.py --job gen:llada-base:test >> ~/codify/campaign/assist.log 2>&1 &

Waits until the campaign is running JOB on one GPU and has nothing else left to do except
JOB's own scoring, and the other GPU is idle. Then it runs generate_candidates.py --reverse on
that GPU, in its own work dir. Every minute it copies each side's rows to the other as
<split>.ext.jsonl (temp file, then rename), so neither redoes the other's questions; the
campaign's job merges the side file when it ends. To the campaign this process is just
another user on that GPU, so nothing else is launched there while it runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time

from campaign import HOME, PIPE, STATE, build_jobs, cfg, py, read_jsonl

ASSIST_WORK = HOME / "codify" / "assist-work"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z {msg}", flush=True)


def copy(src, dst) -> int:
    """Atomically replace dst with src's rows; returns the row count."""
    if not src.exists():
        return 0
    tmp = dst.with_suffix(".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return len(read_jsonl(dst))


def gpu_idle(idx: int) -> bool:
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    uuid = {int(i): u.strip() for i, u in (l.split(",") for l in uuid.splitlines())}[idx]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    return uuid not in apps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="campaign gen job id, e.g. gen:llada-base:test")
    a = ap.parse_args()
    c = cfg()
    _, model, split = a.job.split(":")
    others = {j.id for j in build_jobs(c)} - {a.job} - {f"score:{model}:{split}:{s}" for s in c["scorers"]}
    main_out = PIPE / "work" / "candidates" / model / f"{split}.jsonl"
    mine = ASSIST_WORK / "candidates" / model / f"{split}.jsonl"
    mine.parent.mkdir(parents=True, exist_ok=True)
    data = ASSIST_WORK / "data"
    if not data.exists():
        data.symlink_to(PIPE / "work" / "data")

    log(f"waiting: {a.job} running, every other job done, the other GPU idle")
    while True:
        st = json.loads((STATE / "status.json").read_text())
        done = set(json.loads((STATE / "done.json").read_text()))
        run = st["running"].get(a.job)
        if a.job in done:
            log(f"{a.job} already finished; nothing to do"); return 0
        if run and others <= done and not set(st["running"]) - {a.job}:
            other = 1 - run["gpu"]
            if gpu_idle(other):
                break
        time.sleep(60)
    limit = int(re.findall(rf"LAUNCH {re.escape(a.job)} on GPU \d+ \(pid \d+, try \d+\) limit (\d+)",
                           (STATE / "campaign.log").read_text())[-1])
    log(f"start on GPU {other}: {a.job} limit {limit} in reverse (campaign has it on GPU {run['gpu']})")
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(other),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CODIFY_WORK_DIR=str(ASSIST_WORK))
    cmd = [py(c["models"][model]["venv"]), "generate_candidates.py", "--model", model, "--split", split,
           "--limit", str(limit), "--reverse"]
    for t in range(1, 9):                                 # display watchdog: resume, as the campaign does
        proc = subprocess.Popen(cmd, cwd=PIPE, env=env, stdout=open(STATE / "logs" / "assist_gen.log", "a"),
                                stderr=subprocess.STDOUT)
        while proc.poll() is None:
            a_rows = copy(mine, main_out.with_name(f"{split}.ext.jsonl"))
            b_rows = copy(main_out, mine.with_name(f"{split}.ext.jsonl"))
            if a.job in set(json.loads((STATE / "done.json").read_text())):
                log("campaign job finished; stopping the assist"); proc.terminate(); proc.wait(); break
            time.sleep(60)
        copy(mine, main_out.with_name(f"{split}.ext.jsonl"))
        log(f"assist try {t} exit {proc.returncode}; assist rows {len(read_jsonl(mine))}, campaign rows {len(read_jsonl(main_out))}")
        if proc.returncode == 0 or a.job in set(json.loads((STATE / "done.json").read_text())):
            return 0
        time.sleep(30)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
