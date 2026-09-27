"""Render the justified-dismissal analysis for L1-L12."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import read_json, write_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "limitations" / "data", ROOT / "limitations" / "output"
DIMS = ("liveness", "feasibility", "claim", "specificity")
BAR, FLOOR = 70.0, 4


def main() -> int:
    lims = {l["id"]: l for l in read_json(DATA / "master_merged.json")["consolidated"]}
    sc = read_json(DATA / "dismissal_scores.json")
    mapping = {m["id"]: m for m in read_json(DATA / "litreview_mapping.json")}

    rows, detail = [], []
    for lid in [f"L{i}" for i in range(1, 13)]:
        s = sc["scores"][lid]
        total = round(sum(s[d] for d in DIMS) / len(DIMS) * 10, 1)
        floor_fail = [d for d in DIMS if s[d] < FLOOR]
        passes = total >= BAR and not floor_fail
        assert not passes, f"{lid} passes the bar -- the dismissal argument does not hold"
        m = mapping[lid]
        rows.append((lid, s, total, floor_fail, m))

        refs = "\n".join(
            f"  - *{r['title']}* ({r['year']}), LLM relevance {r['llm_relevance']}, match {r['score']}"
            for r in m["top_references"][:3])
        detail.append(
            f"### {lid} — {lims[lid]['title']}\n\n"
            f"**Score {total}/100** — liveness {s['liveness']}, feasibility {s['feasibility']}, "
            f"claim {s['claim']}, anchor-specificity {s['specificity']}. "
            f"Fails the floor on: {', '.join(floor_fail) if floor_fail else 'none'}.\n\n"
            f"**Why we decline it.** {s['note']}\n\n"
            f"**Mapped onto our generated literature review.** "
            f"{m['pool_engagement']} of {m['pool_size']} retrieved references engage this topic; "
            f"the survey discusses it chiefly under *{m['survey_sections'][0]['heading']}*. "
            f"Closest retrieved work:\n{refs}\n")

    table = "\n".join(
        f"| {lid} | {s['liveness']} | {s['feasibility']} | {s['claim']} | {s['specificity']} | "
        f"**{tot}** | {m['pool_engagement']}/48 | {s['reason']} |"
        for lid, s, tot, _, m in rows)

    infeasible = [r[0] for r in rows if r[1]["feasibility"] <= 3]
    weak = [r[0] for r in rows if r[1]["feasibility"] >= 4]

    doc = f"""# Why we decline all twelve generated research gaps

**Anchor paper.** Gong et al., *Scaling Diffusion Language Models via Adaptation from
Autoregressive Models*, ICLR 2025.

**Where the twelve came from.** A faithful implementation of Al Azher, Guo & Alhoori,
*Multi-Agent LLMs for Generating Research Limitations*
([arXiv:2601.11578](https://arxiv.org/abs/2601.11578)) — one of the limitation-generation
frameworks provided for this task. Four worker agents (Extractor, Analyzer, Reviewer,
Citation) over the anchor's full text plus a retrieval-augmented corpus, then a Judge, a
Self-Feedback loop and a Master consolidation. 33 raw limitations from four agents,
consolidated to twelve. Pipeline and methodology: [`REPORT.md`](../REPORT.md). The twelve
themselves: [`limitations_and_research_problem.md`](limitations_and_research_problem.md).

**What this document is.** We are declining all twelve. This is the argument for that,
scored against an explicitly anchored rubric and mapped onto the literature review this
project generated in the prior assignment using QUAL-SG.

---

## The rubric

| Dimension | Anchoring |
|---|---|
| **Liveness** | {sc['_rubric']['liveness']} |
| **Feasibility** | {sc['_rubric']['feasibility']} |
| **Claim strength** | {sc['_rubric']['claim']} |
| **Anchor specificity** | {sc['_rubric']['specificity']} |

**Bar:** {sc['_bar']}

---

## Scorecard

| Gap | Live | Feas | Claim | Spec | Total | In our review | Reason declined |
|---|---|---|---|---|---|---|---|
{table}

---

## The structural finding

The twelve do not fail individually and for unrelated reasons. They partition into two
groups, and **nothing survives both filters**:

- **Training-bound, strong claims** — {', '.join(infeasible)}. These include the best science in
  the set. L1 is a verified open gap: nobody has tested DiffuLLaMA's actual annealing at 7B for
  full-attention diffusion. L4 is the sharpest criticism available, since L1, L2 and L3 are each
  an instance of it. All are unreachable without adaptation training we cannot run.
- **Inference-reachable, weak claims** — {', '.join(weak)}. Every gap we can actually execute is
  either already closed, already crowded, or too thin to carry a contribution.

**This is a property of the anchor paper, not a failure of the generation method.**
DiffuLLaMA's substantive weaknesses live in its *training recipe* — which components were kept,
which were dropped for implementation convenience, and which were never ablated at the scale
deployed. A frozen released checkpoint cannot answer any of those questions, because the
counterfactual was never trained. What remains observable at inference is a thin surface.

We therefore decline all twelve rather than select the least-bad one, and note explicitly that
L1 and L4 are declined **on resources, not on merit**. Presenting them as scientifically weak
would be indefensible; they are good questions we cannot afford to ask.

---

## Per-gap detail

{chr(10).join(detail)}
---

## Method note

Reference matching uses the same hybrid BM25 + TF-IDF retrieval as the limitation pipeline
(`scripts/map_to_litreview.py`), run over the 48-reference pool and the section structure of the
generated survey. Engagement counts references scoring above a fixed similarity threshold.

**Low engagement is not itself grounds for dismissal** — an untouched gap may simply be novel, and
L5 and L8 both have low engagement while being declined for unrelated reasons. Coverage is one
input to the rubric, never the verdict.

**Correction recorded.** L8 as originally generated asserted that the anchor's HumanEval infilling
figure belonged to a separately trained Diffu-CodeLLaMA. This is false — Table 1 assigns
DiffuLLaMA 7B its own Code score of 15.5 pass@1, while Diffu-CodeLLaMA's 0.76 appears in Table 8,
a different finetune. The error originated in the Reviewer agent and was corrected before
submission. It is recorded rather than quietly fixed, because it is a real failure mode of the
generation pipeline.
"""
    write_text(OUT / "gap_dismissal_analysis.md", doc)
    print(f"wrote {OUT/'gap_dismissal_analysis.md'}")
    print(f"  all 12 verified below the bar ({BAR}/100 with no dimension under {FLOOR})")
    print(f"  training-bound : {infeasible}")
    print(f"  feasible/thin  : {weak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
