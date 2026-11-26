"""Score decision-model requests through OpenRouter's /api/alpha/decisions.

Runs on the host with network access (ada), not on the GPU box. Reads the
prompts.jsonl built by make_openrouter_prompts.py, writes one result per uid:

    {"uid": ..., "model": ..., "probs": {"A": 0.7, ...}}

Retries 429/5xx with jittered backoff, resumes by uid. The API key is read from
$OPENROUTER_API_KEY; it is never written to the output.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openrouter.ai/api/alpha/decisions"


def _context() -> ssl.SSLContext:
    """Verified by default. The lab firewall MITMs HTTPS, so the GPU-side runner
    sets OPENROUTER_INSECURE=1 explicitly; that is a documented lab workaround,
    not the default. OPENROUTER_CA_BUNDLE can point at the firewall's CA."""
    if os.environ.get("OPENROUTER_INSECURE") == "1":
        print("WARNING: TLS verification disabled (OPENROUTER_INSECURE=1)", flush=True)
        return ssl._create_unverified_context()
    cafile = os.environ.get("OPENROUTER_CA_BUNDLE")
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


CTX = _context()


def one(req: dict, key: str, model: str, tries: int = 6) -> dict | None:
    body = json.dumps({"model": model, "state": req["state"],
                       "questions": req["questions"]}).encode()
    for attempt in range(tries):
        try:
            t0 = time.time()
            r = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=120, context=CTX) as resp:
                d = json.loads(resp.read())
            latency = round(time.time() - t0, 3)
            ans = d["answers"]
            out = {"uid": req["uid"], "model": req.get("model", "control"),
                   "latency_s": latency, "api_model": d.get("model"),
                   "api_id": d.get("id"), "usage": d.get("usage")}
            if "idx" in req:
                out["idx"] = req["idx"]
            if "best" in ans:
                out["probs"] = ans["best"]["probabilities"]
            elif "correct" in ans:
                out["noul"] = ans["correct"]["noul"]
            else:
                choice = next(v for v in ans.values() if v.get("type") == "choice")
                out["probs"] = choice["probabilities"]
            return out
        except Exception as e:
            msg = str(e)
            wait = min(60, (2 ** attempt) + random.random())
            print(f"retry {attempt+1} {req['uid']} {type(e).__name__} {msg[:120]}", flush=True)
            if attempt == tries - 1:
                return {"uid": req["uid"], "model": req.get("model", "control"), "error": msg[:300]}
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="perplexity/pplx-decider-flash-v1.3:free")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    reqs = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    reqs = reqs[: a.limit] if a.limit else reqs
    def req_key(r: dict) -> tuple:
        return (r.get("model", "control"), r["uid"], r.get("idx"))

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "error" not in r:          # an error is not a completed request
                    done.add(req_key(r))
    todo = [r for r in reqs if req_key(r) not in done]
    print(f"{len(done)} done, {len(todo)} to go, workers={a.workers}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "a", encoding="utf-8") as f:
        for k, res in enumerate(ex.map(lambda r: one(r, key, a.model), todo), 1):
            if res is not None:
                f.write(json.dumps(res) + "\n")
                f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
