"""Stage 6 -- rank each problem's ideas and render the assignment deliverable.

Ranking key: the mean of the ten ReviewingAgent ratings on the final drafts -- five for
the method (Table 14) and five for the experiment (Table 15). This is the paper's own
evaluation: it reports each stage's five-criterion average (sec 4.3, Fig. 2).

The assignment asks for ranking on criteria "such as novelty, feasibility, clarity,
impact". These are reported as columns, mapped onto the paper's criteria rather than
scored by an extra, invented judge:

    Novelty     <- method Innovativeness
    Feasibility <- experiment Feasibility
    Clarity     <- mean(method Clarity, experiment Clarity)
    Impact      <- method Generalizability  (problem-level Significance is shared by every
                   idea for a problem, so it cannot discriminate between them; it is
                   reported once, with the problem)

Ties on the overall mean break on the weakest single rating (an idea with no 2s beats
one with a 2), then on experiment Feasibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, IDEATION, OUT, read_json, write_json, write_text  # noqa: E402
from templates import CRITERIA  # noqa: E402


def score(idea: dict) -> dict:
    m = idea["method"]["final"]["reviews"]
    e = idea["experiment"]["final"]["reviews"]
    ratings = [r["rating"] for r in m.values()] + [r["rating"] for r in e.values()]
    return {
        "overall": round(sum(ratings) / len(ratings), 2),
        "min": min(ratings),
        "novelty": m["Innovativeness"]["rating"],
        "feasibility": e["Feasibility"]["rating"],
        "clarity": round((m["Clarity"]["rating"] + e["Clarity"]["rating"]) / 2, 1),
        "impact": m["Generalizability"]["rating"],
        "method": {k: v["rating"] for k, v in m.items()},
        "experiment": {k: v["rating"] for k, v in e.items()},
    }


def main() -> int:
    cfg = read_json(IDEATION / "config.json")
    problems = {p["key"]: p for p in read_json(DATA / "problems.json")}
    ideas = read_json(DATA / "ideas.json")
    sel = read_json(DATA / "selected_problems.json")
    k = cfg["top_k_ideas"]

    ranking, doc = {}, [
        "# Research Problems and Ranked Research Ideas\n",
        "**Base paper.** Gong et al., *Scaling Diffusion Language Models via Adaptation from "
        "Autoregressive Models*, ICLR 2025.\n",
        "**Method.** An implementation of **ResearchAgent** (Baek et al., "
        "[arXiv:2404.07738](https://arxiv.org/abs/2404.07738)). Problems, methods and experiment designs "
        "are generated with the paper's own prompts (Tables 6-8) over the base paper, the works that cite it, "
        "and entities retrieved from a knowledge store (Eq. 2). Each is reviewed by ReviewingAgents on "
        "five criteria per stage with the paper's human-induced rubrics (Tables 9-15), and refined from "
        f"the reviews for {cfg['refinement_rounds']} rounds. Deviations: [`REPORT.md`](../REPORT.md) §3.\n",
        f"**Selection.** Problems chosen by: {sel.get('selected_by', '?')}. {sel.get('reason', '')}\n",
        "**Novelty.** Not yet checked for any idea below. Every idea must be checked against recent "
        "literature, starting with the base paper's own authors' later work, before it is pursued.\n",
        "---\n"]

    for n, pk in enumerate(sel["selected"], 1):
        p = problems[pk]["final"]
        scored = sorted(({**i, "score": score(i)} for i in ideas[pk]),
                        key=lambda i: (-i["score"]["overall"], -i["score"]["min"], -i["score"]["feasibility"]))
        ranking[pk] = [{"rank": r, "key": i["key"], **i["score"]} for r, i in enumerate(scored, 1)]
        pscores = ", ".join(f"{m} {p['reviews'][m]['rating']}/5" for m in CRITERIA["problem"])

        doc += [f"## Research problem {n} ({pk})\n", f"{p['text']}\n",
                f"**Rationale.** {p['rationale']}\n", f"*ReviewingAgent scores: {pscores}.*\n",
                f"### All {len(scored)} ideas, ranked\n",
                "| Rank | Idea | Overall /5 | Novelty | Feasibility | Clarity | Impact | Weakest |",
                "|---|---|---|---|---|---|---|---|"]
        for r, i in enumerate(scored, 1):
            s = i["score"]
            mark = "**" if r <= k else ""
            doc.append(f"| {mark}{r}{mark} | {i['key']} | {mark}{s['overall']}{mark} | {s['novelty']} | "
                       f"{s['feasibility']} | {s['clarity']} | {s['impact']} | {s['min']} |")
        doc.append(f"\nTop {k} submitted; the rest are listed so the cut is visible.\n")

        for r, i in enumerate(scored[:k], 1):
            m, e, s = i["method"]["final"], i["experiment"]["final"], i["score"]
            weak = min(list(m["reviews"].items()) + list(e["reviews"].items()), key=lambda x: x[1]["rating"])
            doc += [f"### Idea {n}.{r} — {i['key']} (overall {s['overall']}/5)\n",
                    f"**Method.**\n\n{m['text']}\n", f"**Why this method.** {m['rationale']}\n",
                    f"**Experiment design.**\n\n{e['text']}\n", f"**Why this design.** {e['rationale']}\n",
                    "| Method | " + " | ".join(CRITERIA["method"]) + " |",
                    "|---|" + "---|" * 5,
                    "| rating | " + " | ".join(str(s["method"][c]) for c in CRITERIA["method"]) + " |",
                    "",
                    "| Experiment | " + " | ".join(CRITERIA["experiment"]) + " |",
                    "|---|" + "---|" * 5,
                    "| rating | " + " | ".join(str(s["experiment"][c]) for c in CRITERIA["experiment"]) + " |",
                    f"\n**Weakest point per the reviewers ({weak[0]}, {weak[1]['rating']}/5).** {weak[1]['feedback']}\n"]
        doc.append("---\n")

    write_json(OUT / "ranking.json", ranking)
    write_text(OUT / "research_problems_and_ideas.md", "\n".join(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
