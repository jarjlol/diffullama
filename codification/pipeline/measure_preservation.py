"""Step 5 -- per-layer CKA between each diffusion LM and its own AR base.

    python3 measure_preservation.py                  # every model with a base
    python3 measure_preservation.py --model diffullama

Runs extract_hidden.py twice per model in SEPARATE processes (dlm, then base), then
computes linear CKA block by block. Writes work/preservation.json:

    {model: {"per_layer": [...], "mean": m, "base": hf_id}}

Models with no AR base (LLaDA, trained from scratch) are recorded as the control,
with no score. Use --python to point each role at the right virtualenv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from common import HERE, WORK, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*")
    ap.add_argument("--python", default=sys.executable, help="interpreter for extraction (a model venv)")
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()
    import torch
    from cka import cka_torch
    c = cfg(); out = WORK / "preservation.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for name in (a.model or list(c["models"])):
        spec = c["models"][name]
        if not spec["base"]:
            res[name] = {"base": None, "mean": None, "per_layer": None, "note": "trained from scratch: control, no AR base"}
            continue
        if not a.skip_extract:
            for role in ("dlm", "base"):
                subprocess.run([a.python, str(HERE / "extract_hidden.py"), "--model", name, "--role", role], check=True)
        d = torch.load(WORK / "hidden" / name / "dlm.pt"); b = torch.load(WORK / "hidden" / name / "base.pt")
        assert d["layers"].shape[:2] == b["layers"].shape[:2], f"{name}: layer/segment mismatch {d['layers'].shape} vs {b['layers'].shape}"
        per = [cka_torch(d["layers"][i], b["layers"][i]) for i in range(d["layers"].shape[0])]
        res[name] = {"base": spec["base"], "per_layer": per, "mean": sum(per) / len(per)}
        print(f"{name:<12} mean CKA {res[name]['mean']:.4f}  (min {min(per):.3f} at block {per.index(min(per))})")
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
