"""Stage 4 -- Judge Agent (arXiv:2601.11578 sec 3.1, 3.3).

Reference-free quality assessment of each worker agent's output on four
dimensions, each scored 0-10 and combined by the paper's fixed weights:

    Depth           0.2   criticality and insight into study-design flaws
    Originality     0.2   novel versus generic insight
    Actionability   0.3   feasibility for a researcher to address
    Topic Coverage  0.3   breadth across methodology, scope, ethics

    total = 100 * sum(w_d * score_d) / 10        -> 0-100

Any agent whose weighted mean falls below 8/10 is routed to the Self-Feedback
Agent for regeneration (paper: threshold 8/10, at most two retries).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_json, write_json  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "limitations" / "data"

WEIGHTS = {"depth": 0.2, "originality": 0.2, "actionability": 0.3, "topic_coverage": 0.3}
THRESHOLD_10 = 8.0
MAX_RETRIES = 2

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "paper's judge weights must sum to 1.0"


def score_agent(dims: dict[str, int]) -> tuple[float, float]:
    """Return (weighted mean on 0-10, total on 0-100)."""
    missing = set(WEIGHTS) - set(dims)
    if missing:
        raise ValueError(f"judge did not score dimensions: {sorted(missing)}")
    mean10 = sum(WEIGHTS[d] * dims[d] for d in WEIGHTS)
    return mean10, round(mean10 * 10, 1)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    rubric_name = argv[0] if len(argv) > 0 else "judge_scores.json"
    out_name = argv[1] if len(argv) > 1 else "agent_outputs.json"
    verdict_name = argv[2] if len(argv) > 2 else "judge_verdict.json"
    rubric = read_json(DATA / rubric_name)
    outputs = read_json(DATA / out_name)
    print(f"[judge] rubric={rubric_name} outputs={out_name}\n")

    results, needs_feedback = {}, []
    for agent, dims in rubric["dimension_scores"].items():
        mean10, total100 = score_agent(dims)
        below = mean10 < THRESHOLD_10
        results[agent] = {
            "dimensions": dims,
            "weighted_mean_10": round(mean10, 2),
            "total_100": total100,
            "n_limitations": len(outputs[agent]["limitations"]),
            "below_threshold": below,
            "feedback": rubric["feedback"].get(agent, ""),
        }
        if below:
            needs_feedback.append(agent)

    write_json(DATA / verdict_name, {
        "weights": WEIGHTS,
        "threshold_10": THRESHOLD_10,
        "max_retries": MAX_RETRIES,
        "results": results,
        "routed_to_self_feedback": needs_feedback,
    })

    hdr = f"{'agent':<11}{'depth':>6}{'orig':>6}{'action':>8}{'cover':>7}{'/10':>7}{'/100':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for a, r in results.items():
        d = r["dimensions"]
        verdict = "REGENERATE" if r["below_threshold"] else "pass"
        print(f"{a:<11}{d['depth']:>6}{d['originality']:>6}{d['actionability']:>8}"
              f"{d['topic_coverage']:>7}{r['weighted_mean_10']:>7.2f}{r['total_100']:>7.1f}  {verdict}")
    print(f"\nrouted to Self-Feedback Agent: {needs_feedback or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
