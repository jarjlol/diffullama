"""Iteration-1 GPU campaign on the shared Blackwell workstation. Stdlib only, Linux only.

    nohup python3 campaign.py >/dev/null 2>&1 &     # supervisor; state in ~/codify/campaign/
    python3 campaign.py --dry-run --polls 3         # decide and log, launch nothing
    touch ~/codify/campaign/STOP                    # let running jobs finish, start no new ones
    touch ~/codify/campaign/ABORT                   # stop OUR running jobs (never anyone else's)
    cat ~/codify/campaign/status.json               # live state; events in campaign.log

House rules, because the machine is shared:
  * It never signals a process it did not start. ABORT kills only the process groups it created.
  * A GPU is used only after it has been free for SETTLE_MIN consecutive minutes. Free means no
    other process on it, under MEM_IDLE_MIB of memory not ours, and no vLLM process of anyone
    else alive anywhere (a benchmark sweep restarts its server between runs, so a short gap is
    not a free GPU).
  * One job per GPU, pinned with CUDA_VISIBLE_DEVICES in PCI order. Before each new job the GPU
    is re-checked. If another user has started on it meanwhile, the campaign waits.
  * It writes only under ~/codify, plus the shared Hugging Face cache, which it only adds to.

Plan (codification/ITERATION_1.md): per model a 20-question smoke run on dev, a mechanical gate,
then the dev/test subset, AR scoring with three scorers, hidden-state extraction for CKA, and
after every finished model a refreshed baseline table in ~/codify/campaign/results/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = HOME / "codify" / "diffullama"
PIPE = REPO / "codification" / "pipeline"
TASK = REPO / "codification" / "rerank_task"
STATE = HOME / "codify" / "campaign"
VENVS = HOME / "venvs"
SYS_PY = "/usr/bin/python3"

MODELS = ["diffugpt-s", "diffugpt-m", "diffullama", "dream-base", "llada-base"]
SEVEN_B = {"diffullama", "dream-base", "llada-base"}
SCORERS = ["gpt2", "qwen0.5b", "tinyllama"]
SMOKE_N, DEV_N, TEST_N = 20, 200, 500
TEST_FALLBACKS = [300, 200]
DEADLINE = dt.datetime(2026, 11, 4, 16, 30, tzinfo=dt.timezone.utc)   # Wed 22:00 IST: all generation done
MEM_IDLE_MIB = 2000
MIN_FREE_DISK_GB = 150
FOREIGN_GPU_RE = re.compile(r"vllm|VLLM::|sglang|text-generation-launcher|torchrun|deepspeed", re.I)
PREFETCH = ["diffusionfamily/diffugpt-s", "diffusionfamily/diffugpt-m", "openai-community/gpt2",
            "openai-community/gpt2-medium", "diffusionfamily/diffullama", "Dream-org/Dream-v0-Base-7B",
            "GSAI-ML/LLaDA-8B-Base", "NousResearch/Llama-2-7b-hf", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"]


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def log(msg: str) -> None:
    line = f"{now():%Y-%m-%d %H:%M:%S}Z {msg}"
    with open(STATE / "campaign.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sh(args, timeout=60) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def cfg() -> dict:
    return json.loads((PIPE / "config.json").read_text(encoding="utf-8"))


def py(venv: str) -> str:
    return str(VENVS / venv / "bin" / "python")


# ------------------------------------------------------------------ the machine

def ancestors(pid: int) -> set[int]:
    out = set()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return out


def snapshot(ours: set[int]) -> dict:
    """GPU state, split into what is ours and what is anyone else's."""
    gpus = {}
    for line in sh(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]).splitlines():
        i, uuid, mem = [x.strip() for x in line.split(",")]
        gpus[uuid] = {"index": int(i), "mem": int(mem), "our_mem": 0, "foreign": []}
    if not gpus:
        return {"ok": False, "gpus": {}, "foreign_servers": []}
    for line in sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or parts[0] not in gpus:
            continue
        pid, mem = int(parts[1]), int(parts[2]) if parts[2].isdigit() else 0
        if ancestors(pid) & ours:
            gpus[parts[0]]["our_mem"] += mem
        else:
            who = sh(["ps", "-o", "user=,comm=", "-p", str(pid)]).strip() or "?"
            gpus[parts[0]]["foreign"].append(f"{pid} {who}")
    servers = []
    for line in sh(["ps", "-eo", "pid=,user=,args="]).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and FOREIGN_GPU_RE.search(parts[2]) and not (ancestors(int(parts[0])) & ours) \
                and "campaign.py" not in parts[2] and "grep" not in parts[2]:
            servers.append(f"{parts[0]} {parts[1]} {parts[2][:90]}")
    return {"ok": True, "gpus": {g["index"]: g for g in gpus.values()}, "foreign_servers": servers}


# ------------------------------------------------------------------ jobs

class Job:
    def __init__(self, jid, kind, model, deps=(), gpu=True, **kw):
        self.id, self.kind, self.model, self.deps, self.gpu = jid, kind, model, list(deps), gpu
        self.kw, self.state, self.tries, self.proc, self.gpu_index, self.started = kw, "pending", 0, None, None, None
        self.note = ""


def build_jobs(c: dict) -> list[Job]:
    jobs: list[Job] = []
    small = [m for m in MODELS if m not in SEVEN_B]
    order = small + [m for m in MODELS if m in SEVEN_B]
    for m in order:                                                   # smoke + gate, cheapest first
        jobs.append(Job(f"smoke:{m}", "gen", m, split="dev", limit=SMOKE_N))
        jobs.append(Job(f"gate:{m}", "gate", m, [f"smoke:{m}"], gpu=False))
    for group in (small, [m for m in MODELS if m in SEVEN_B]):        # small models end to end first
        for m in group:
            for sp in ("dev", "test"):
                jobs.append(Job(f"gen:{m}:{sp}", "gen", m, [f"gate:{m}"], split=sp,
                                limit=DEV_N if sp == "dev" else TEST_N))
        for m in group:
            for sp in ("dev", "test"):
                for s in SCORERS:
                    jobs.append(Job(f"score:{m}:{sp}:{s}", "score", m, [f"gen:{m}:{sp}"], split=sp, scorer=s))
            if c["models"][m]["base"]:
                jobs.append(Job(f"hidden:{m}:dlm", "hidden", m, [f"gate:{m}"], role="dlm"))
                jobs.append(Job(f"hidden:{m}:base", "hidden", m, [f"hidden:{m}:dlm"], role="base"))
    return jobs


def command(job: Job, c: dict) -> list[str]:
    spec = c["models"][job.model]
    if job.kind == "gen":
        return [py(spec["venv"]), "generate_candidates.py", "--model", job.model, "--split", job.kw["split"],
                "--limit", str(job.kw["limit"])]
    if job.kind == "score":
        return [py("scorer"), "score_candidates.py", "--model", job.model, "--split", job.kw["split"],
                "--scorer", job.kw["scorer"]]
    if job.kind == "hidden":
        return [py(spec["venv"] if job.kw["role"] == "dlm" else "scorer"), "extract_hidden.py",
                "--model", job.model, "--role", job.kw["role"]]
    raise ValueError(job.kind)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gate(job: Job, c: dict, timing: dict) -> tuple[bool, str]:
    """Mechanical checks on the smoke run. Low accuracy is a finding, not a failure."""
    sys.path.insert(0, str(TASK))
    import task                                                       # protected grading code
    rows = read_jsonl(PIPE / "work" / "candidates" / job.model / "dev.jsonl")[:SMOKE_N]
    gold = {r["uid"]: r["answer"] for r in read_jsonl(PIPE / "work" / "data" / "dev.jsonl")}
    n = c["n_candidates"]
    if len(rows) < SMOKE_N or any(len(r["candidates"]) != n for r in rows):
        return False, f"{len(rows)} rows, expected {SMOKE_N} x {n} candidates"
    cands = [cd for r in rows for cd in r["candidates"]]
    extract = sum(cd["answer"] is not None for cd in cands) / len(cands)
    distinct = statistics.mean(len({cd["text"] for cd in r["candidates"]}) for r in rows)
    secs = [r["gen_seconds"] for r in rows if "gen_seconds" in r]
    spq = statistics.median(secs) if secs else None
    first = sum(task.is_correct(r["candidates"][0]["answer"], gold.get(r["uid"])) for r in rows) / len(rows)
    oracle = sum(any(task.is_correct(cd["answer"], gold.get(r["uid"])) for cd in r["candidates"]) for r in rows) / len(rows)
    if spq:
        timing[job.model] = spq
    msg = (f"extract {extract:.0%}, distinct/question {distinct:.1f}, first {first:.0%}, oracle {oracle:.0%}, "
           f"{spq if spq is None else round(spq, 1)} s/question")
    ok = extract >= 0.5 and distinct >= 2.0
    return ok, msg + ("" if ok else " -- FAILED: answers missing or candidates not diverse")


def pick_test_limit(job: Job, jobs: list[Job], timing: dict, n_gpus: int) -> int:
    """Shrink the 7B test subset if the projected generation time misses the deadline."""
    if job.model not in SEVEN_B:
        return TEST_N
    hours_left = (DEADLINE - now()).total_seconds() / 3600
    for limit in [TEST_N] + TEST_FALLBACKS:
        need = 0.0
        for j in jobs:
            if j.kind == "gen" and j.model in SEVEN_B and j.state in ("pending", "running") and not j.id.startswith("smoke"):
                q = limit if j.kw["split"] == "test" else j.kw["limit"]
                need += timing.get(j.model, 60.0) * q / 3600
        if need / max(n_gpus, 1) <= hours_left:
            return limit
    return TEST_FALLBACKS[-1]


def refresh_report(jobs: list[Job]) -> None:
    """Assemble only fully scored models, then baselines (and analyze once CKA exists)."""
    res = STATE / "results"; res.mkdir(exist_ok=True)
    for sp in ("dev", "test"):
        ready = [m for m in MODELS if all(j.state == "done" for j in jobs if j.kind == "score" and j.model == m and j.kw["split"] == sp)
                 and any(j.kind == "score" and j.model == m and j.kw["split"] == sp for j in jobs)]
        if not ready:
            continue
        out = sh([SYS_PY, str(PIPE / "assemble.py"), "--split", sp, "--models", *ready], timeout=600)
        log(f"REPORT assemble {sp}: {out.strip().splitlines()[-1] if out.strip() else 'no output'}")
        table = sh([SYS_PY, str(TASK / "baselines.py"), "--split", sp, "--json"], timeout=600)
        if table.strip():
            (res / f"baselines_{sp}.json").write_text(table, encoding="utf-8")
            (res / f"baselines_{sp}.txt").write_text(sh([SYS_PY, str(TASK / "baselines.py"), "--split", sp], timeout=600), encoding="utf-8")
    if (PIPE / "work" / "preservation.json").exists():
        (res / "analyze_test.txt").write_text(sh([SYS_PY, str(PIPE / "analyze.py"), "--split", "test"], timeout=600), encoding="utf-8")


def measure_cka(jobs: list[Job]) -> None:
    have = [m for m in MODELS if all(j.state == "done" for j in jobs if j.kind == "hidden" and j.model == m)
            and any(j.kind == "hidden" and j.model == m for j in jobs)]
    if have:
        out = subprocess.run([py("scorer"), "measure_preservation.py", "--skip-extract", "--model", *have],
                             cwd=PIPE, capture_output=True, text=True, timeout=1800)
        log(f"REPORT CKA {have}: {(out.stdout + out.stderr).strip()[-400:]}")


# ------------------------------------------------------------------ main loop

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--polls", type=int, default=0, help="stop after N polls (0 = run to completion)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--settle-min", type=float, default=20)
    a = ap.parse_args()
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    lock = open(STATE / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another campaign.py holds the lock", file=sys.stderr); return 1
    c = cfg(); jobs = build_jobs(c); by_id = {j.id: j for j in jobs}
    done_file = STATE / "done.json"                                   # survives a supervisor restart
    for jid in json.loads(done_file.read_text()) if done_file.exists() else []:
        if jid in by_id:
            by_id[jid].state = "done"
    timing = json.loads((STATE / "timing.json").read_text()) if (STATE / "timing.json").exists() else {}
    free_since: dict[int, float] = {}
    prefetch = None
    if not a.dry_run:
        script = "; ".join(f"from huggingface_hub import snapshot_download as s; s('{r}')" for r in PREFETCH)
        prefetch = subprocess.Popen(["nice", "-n", "10", py("scorer"), "-c", script], stdout=open(STATE / "logs" / "prefetch.log", "a"),
                                    stderr=subprocess.STDOUT, start_new_session=True)
    log(f"START pid {os.getpid()} dry_run={a.dry_run} settle={a.settle_min}min jobs={len(jobs)} done={sum(j.state == 'done' for j in jobs)}")
    polls = 0
    while True:
        polls += 1
        ours = {os.getpid()} | {j.proc.pid for j in jobs if j.proc}
        snap = snapshot(ours)
        # finished jobs
        for j in jobs:
            if j.proc and j.proc.poll() is not None:
                rc = j.proc.returncode; j.proc = None
                mins = (time.time() - j.started) / 60
                if rc == 0:
                    j.state = "done"; log(f"DONE {j.id} on GPU {j.gpu_index} in {mins:.0f} min")
                    if j.kind == "score" and all(x.state == "done" for x in jobs if x.kind == "score" and x.model == j.model):
                        refresh_report(jobs)
                    if j.kind == "hidden" and j.kw["role"] == "base":
                        measure_cka(jobs); refresh_report(jobs)
                elif j.tries < 2:
                    j.state = "pending"; log(f"RETRY {j.id} exit {rc} after {mins:.0f} min (log {STATE}/logs/{j.id.replace(':', '_')}.log)")
                else:
                    j.state = "failed"; log(f"FAILED {j.id} exit {rc} twice (log {STATE}/logs/{j.id.replace(':', '_')}.log)")
        # skip dependents of failures
        changed = True
        while changed:
            changed = False
            for j in jobs:
                if j.state == "pending" and any(by_id[d].state in ("failed", "skipped") for d in j.deps):
                    j.state = "skipped"; changed = True; log(f"SKIP {j.id}: a dependency failed")
        # CPU gates
        for j in jobs:
            if j.kind == "gate" and j.state == "pending" and all(by_id[d].state == "done" for d in j.deps):
                ok, msg = gate(j, c, timing)
                j.state, j.note = ("done" if ok else "failed"), msg
                log(f"GATE {j.model}: {msg}")
                (STATE / "timing.json").write_text(json.dumps(timing, indent=1))
        # GPU availability
        usable = []
        if snap["ok"]:
            for idx, g in snap["gpus"].items():
                free = not g["foreign"] and (g["mem"] - g["our_mem"]) < MEM_IDLE_MIB and not snap["foreign_servers"]
                if free:
                    free_since.setdefault(idx, time.time())
                else:
                    free_since.pop(idx, None)
                if free and time.time() - free_since[idx] >= a.settle_min * 60:
                    usable.append(idx)
        busy = {j.gpu_index for j in jobs if j.proc}
        stop = (STATE / "STOP").exists()
        if (STATE / "ABORT").exists():
            for j in jobs:
                if j.proc:
                    os.killpg(j.proc.pid, signal.SIGTERM); log(f"ABORT sent SIGTERM to our job {j.id} (pgid {j.proc.pid})")
            (STATE / "ABORT").unlink(); stop = True
        disk_ok = shutil.disk_usage(HOME).free / 1e9 > MIN_FREE_DISK_GB
        # launch
        for idx in sorted(usable):
            if idx in busy or stop or not disk_ok:
                continue
            ready = [j for j in jobs if j.state == "pending" and j.gpu and all(by_id[d].state == "done" for d in j.deps)]
            if not ready:
                break
            j = ready[0]
            if j.kind == "gen" and j.kw["split"] == "test" and not j.id.startswith("smoke"):
                j.kw["limit"] = pick_test_limit(j, jobs, timing, max(len(usable), 1))
            cmd = command(j, c)
            if a.dry_run:
                log(f"DRY would launch {j.id} on GPU {idx}: {' '.join(cmd)}"); busy.add(idx); continue
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(idx),
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="8", HF_HUB_DISABLE_TELEMETRY="1")
            logf = open(STATE / "logs" / f"{j.id.replace(':', '_')}.log", "a")
            logf.write(f"\n=== {now():%Y-%m-%d %H:%M:%S}Z GPU {idx}: {' '.join(cmd)}\n"); logf.flush()
            j.proc = subprocess.Popen(cmd, cwd=PIPE, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            j.state, j.gpu_index, j.started = "running", idx, time.time(); j.tries += 1; busy.add(idx)
            log(f"LAUNCH {j.id} on GPU {idx} (pid {j.proc.pid}, try {j.tries}){' limit ' + str(j.kw['limit']) if 'limit' in j.kw else ''}")
        # persist state
        (done_file).write_text(json.dumps([j.id for j in jobs if j.state == "done"]))
        status = {"updated": f"{now():%Y-%m-%d %H:%M:%S}Z", "dry_run": a.dry_run, "stop_requested": stop, "disk_ok": disk_ok,
                  "gpus": {i: {"used_mib": g["mem"], "ours_mib": g["our_mem"], "foreign": g["foreign"],
                               "free_for_min": round((time.time() - free_since[i]) / 60, 1) if i in free_since else 0,
                               "usable": i in usable} for i, g in snap.get("gpus", {}).items()},
                  "waiting_on": snap.get("foreign_servers", [])[:5],
                  "running": {j.id: {"gpu": j.gpu_index, "min": round((time.time() - j.started) / 60)} for j in jobs if j.proc},
                  "counts": {s: sum(j.state == s for j in jobs) for s in ("pending", "running", "done", "failed", "skipped")},
                  "gates": {j.model: j.note for j in jobs if j.kind == "gate" and j.note},
                  "seconds_per_question": timing,
                  "prefetch": None if prefetch is None else ("running" if prefetch.poll() is None else f"exit {prefetch.returncode}")}
        tmp = STATE / ".status.json"; tmp.write_text(json.dumps(status, indent=1)); os.replace(tmp, STATE / "status.json")
        if all(j.state in ("done", "failed", "skipped") for j in jobs):
            refresh_report(jobs); log(f"FINISHED {status['counts']}"); return 0
        if stop and not any(j.proc for j in jobs):
            log("STOPPED on request; rerun campaign.py to resume (finished jobs are remembered)"); return 0
        if a.polls and polls >= a.polls:
            log(f"POLL LIMIT reached ({polls}); gpus {status['gpus']}; waiting_on {status['waiting_on'][:2]}"); return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
