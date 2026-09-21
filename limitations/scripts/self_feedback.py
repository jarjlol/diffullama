"""Stage 5 -- Self-Feedback Agent (arXiv:2601.11578 sec 3.1).

Agents whose Judge score falls below 8/10 are regenerated with the Judge's written
feedback appended to their original prompt.  The paper caps this at two retries.
This script assembles the regeneration prompts and merges the regenerated outputs
into the final agent set that the Master Agent consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_json, read_text, write_json, write_text  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "limitations" / "data"
REQ = DATA / "agent_requests"
MAX_RETRIES = 2

REGEN_HEADER = """\
# REGENERATION (Self-Feedback round {round} of {max_retries})

Your previous output scored {score}/100 from the Judge Agent, below the 8/10
threshold. Judge feedback:

> {feedback}

Regenerate your full output addressing this feedback. Keep what was strong.
Your previous output is shown at the end for reference.

---

"""


def main() -> int:
    verdict = read_json(DATA / "judge_verdict.json")
    round1 = read_json(DATA / "agent_outputs.json")
    routed = verdict["routed_to_self_feedback"]

    if len(routed) == 0:
        print("no agent below threshold; Self-Feedback not triggered")
    for agent in routed:
        r = verdict["results"][agent]
        prompt = REGEN_HEADER.format(
            round=1, max_retries=MAX_RETRIES,
            score=r["total_100"], feedback=r["feedback"],
        ) + read_text(REQ / f"{agent}.md")
        write_text(REQ / f"{agent}_regen.md", prompt)
        print(f"regeneration prompt written: {agent}_regen.md "
              f"(was {r['total_100']}/100, {r['n_limitations']} limitations)")

    # merge: regenerated agents override round 1, others carry forward unchanged
    regen_path = DATA / "agent_outputs_round2.json"
    if regen_path.exists():
        regen = read_json(regen_path)
        final = dict(round1)
        for agent in routed:
            if agent in regen:
                final[agent] = regen[agent]
        final["_rounds"] = {a: (2 if a in routed else 1)
                            for a in ("extractor", "analyzer", "reviewer", "citation")}
        write_json(DATA / "agent_outputs_final.json", final)
        print("\nmerged -> agent_outputs_final.json")
        for a in ("extractor", "analyzer", "reviewer", "citation"):
            print(f"  {a:<11} round {final['_rounds'][a]}  "
                  f"{len(final[a]['limitations'])} limitations")
    else:
        print(f"\n[waiting] {regen_path.name} not yet present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
