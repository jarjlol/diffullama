"""Probe the locally served generator and reviewer BEFORE the full run. Must PASS.

    python3 ideation/local/probe_models.py [env-file]     # default ~/.ideation-local-env

Calls the servers directly and writes nothing to the LLM cache. Checks:

  generator  answers a real archived method-generation prompt; output parses as
             'Method:' + 'Rationale:'; reports latency.
  reviewer   answers up to 12 real archived method-review prompts; every output parses;
             agreement with the ratings the original run's reviewer (Ling-3.0-flash-sante)
             gave the same prompts; rating spread (a reviewer giving everything 4-5 is
             useless for ranking -- the paper's own prompt warns against exactly that).

Verdict PASS requires: generator parses; reviewer parses 100%; >=70% of reviewer ratings
within 1 point of the original reviewer; ratings not all >= 4. Results are written to
ideation/data/local_probe.json so the model choice is recorded with its evidence.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.ideation-local-env"))
for k, v in re.findall(r'^export\s+(\w+)=["\']?([^"\'\n]*)["\']?\s*$', ENV_FILE.read_text(encoding="utf-8"), re.M):
    os.environ[k] = v                       # before importing llm, which reads LLM_TIMEOUT at import

sys.path.insert(0, str(ROOT / "ideation/scripts"))
from llm import _openai_once  # noqa: E402
from research_agent import ParseError, parse_artifact, parse_review  # noqa: E402

ARCHIVE = ROOT / "ideation/data/llm/archive/openrouter-2026-10"
OUT = ROOT / "ideation/data/local_probe.json"


def env(role, key, default=None):
    p = "REVIEWER_" if role == "reviewer" else "LLM_"
    return os.environ.get(p + key) or os.environ.get("LLM_" + key) or default


def served_root(role):
    """The real HF model id behind the served name, for the record."""
    try:
        req = urllib.request.Request(env(role, "BASE_URL").rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {env(role, 'API_KEY', 'local')}"})
        data = json.load(urllib.request.urlopen(req, timeout=15))["data"]
        return [m.get("root") or m["id"] for m in data if m["id"] == env(role, "MODEL")][0]
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"


def call(role, system, user):
    extra = json.loads(env(role, "EXTRA_BODY")) if env(role, "EXTRA_BODY") else None
    t0 = time.time()
    out = _openai_once(env(role, "BASE_URL").rstrip("/"), env(role, "API_KEY", "local"), env(role, "MODEL"),
                       float(env(role, "TEMPERATURE", "0.7")), system, user, extra)
    return out, time.time() - t0


def split(path):
    s, u = Path(path).read_text(encoding="utf-8").split("# SYSTEM\n\n", 1)[1].split("\n\n# USER\n\n", 1)
    return s.strip(), u.strip()


def main() -> int:
    if not ARCHIVE.exists():
        print(f"archive missing: {ARCHIVE}"); return 2
    res = lambda f: Path(str(f).replace("/requests/", "/responses/"))  # noqa: E731
    reqs = sorted(glob.glob(str(ARCHIVE / "requests/*.md")))
    tag = lambda f: os.path.basename(f).rsplit("-", 1)[0]  # noqa: E731
    counts = collections.Counter(map(tag, reqs))

    report = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "env_file": str(ENV_FILE),
              "generator_model": served_root("generator"), "reviewer_model": served_root("reviewer")}
    print(f"generator: {report['generator_model']}\nreviewer : {report['reviewer_model']}\n")

    # ---- generator
    gen = next(f for f in reqs if "-r0-gen-" in f and f.split("/")[-1].startswith("method-") and res(f).exists())
    try:
        text, dt = call("generator", *split(gen))
        body, rat = parse_artifact(text, "Method", "probe")
        g = {"ok": True, "seconds": round(dt, 1), "method_chars": len(body), "rationale_chars": len(rat)}
    except ParseError as e:
        g = {"ok": False, "error": f"ParseError: {e}"}
    except Exception as e:  # noqa: BLE001
        g = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    report["generator"] = g
    print("generator:", g)

    # ---- reviewer: single-chain archived reviews, stratified across the original ratings
    by_rating = collections.defaultdict(list)
    for f in reqs:
        if "-rev-" in f and counts[tag(f)] == 1 and res(f).exists():
            try:
                by_rating[parse_review(res(f).read_text(encoding="utf-8"), "x")["rating"]].append(f)
            except ParseError:
                pass
    picks = []                                  # round-robin across original ratings
    while len(picks) < 12 and any(by_rating.values()):
        for r in sorted(by_rating):
            if by_rating[r] and len(picks) < 12:
                picks.append((by_rating[r].pop(0), r))
    rows = []
    for f, orig in picks:
        try:
            text, dt = call("reviewer", *split(f))
            rows.append({"prompt": os.path.basename(f), "orig": orig, "new": parse_review(text, "probe")["rating"],
                         "seconds": round(dt, 1)})
        except ParseError as e:
            rows.append({"prompt": os.path.basename(f), "orig": orig, "new": None, "error": "ParseError"})
        except Exception as e:  # noqa: BLE001
            rows.append({"prompt": os.path.basename(f), "orig": orig, "new": None, "error": f"{type(e).__name__}"})
    ok = [r for r in rows if r["new"] is not None]
    within1 = sum(abs(r["orig"] - r["new"]) <= 1 for r in ok) / len(ok) if ok else 0
    rv = {"n": len(rows), "parsed": len(ok),
          "mean_abs_diff": round(sum(abs(r["orig"] - r["new"]) for r in ok) / len(ok), 2) if ok else None,
          "within_1": round(within1, 2),
          "ratings": dict(collections.Counter(r["new"] for r in ok)),
          "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 1) if ok else None,
          "rows": rows}
    report["reviewer"] = rv
    print(f"reviewer : parsed {rv['parsed']}/{rv['n']}, within-1 of original {rv['within_1']:.0%}, "
          f"mean |diff| {rv['mean_abs_diff']}, ratings {rv['ratings']}, ~{rv['mean_seconds']} s/review")
    for r in rows:
        print(f"   {r['orig']} -> {r['new'] if r['new'] is not None else 'x (' + r.get('error', '') + ')'}   {r['prompt'][:70]}")

    checks = {
        "generator output parses": g.get("ok", False),
        "reviewer output parses (100%)": bool(rows) and len(ok) == len(rows),
        "reviewer within 1 point of original on >=70%": within1 >= 0.70,
        "reviewer ratings not all 4-5": bool(ok) and min(r["new"] for r in ok) < 4,
    }
    report["checks"] = checks
    report["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nverdict: {report['verdict']}   (recorded in {OUT.relative_to(ROOT)})")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
