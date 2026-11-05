"""Print the latest Arbor review prompt, and whether it has been answered."""
import json, sys
ev = [json.loads(l) for l in open(sys.argv[1])]
idx = [i for i, e in enumerate(ev) if e["type"] == "user.await"]
if not idx:
    sys.exit("no review prompt yet")
i = idx[-1]
d = ev[i]["data"]
answered = any(e["type"] == "user.input_received" for e in ev[i + 1:])
print(f"[{d['kind']}] answered={answered} node={d.get('node_id')}")
print(d["prompt"])
