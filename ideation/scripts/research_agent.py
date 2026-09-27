"""ResearchAgent and ReviewingAgents (Baek et al., arXiv:2404.07738, sec 3.2-3.3).

    problem     p = LLM(T_p(L, Ret(L;K)))              Table 6
    method      m = LLM(T_m(p, L, Ret(L;K)))           Table 7
    experiment  d = LLM(T_e(p, m, L, Ret(L;K)))        Table 8

Each artifact is reviewed by ReviewingAgents, ONE call PER CRITERION (the {metric}
slot in Tables 9-11), and regenerated from the reviews for R rounds (the paper
reports gains saturate after three). An artifact therefore costs (R+1) generations
+ 5(R+1) reviews = 24 calls at R = 3.

Every request is content-addressed: its id hashes the full prompt, so a change to
any upstream input produces fresh requests rather than a stale cache hit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import templates as T  # noqa: E402
from common import DATA, IDEATION, ROOT, read_json, stable_id  # noqa: E402

KINDS = {
    "problem":    ("Problem", T.PROBLEM_SYSTEM, T.PROBLEM_USER, T.REVIEW_PROBLEM_SYSTEM, T.REVIEW_PROBLEM_USER),
    "method":     ("Method", T.METHOD_SYSTEM, T.METHOD_USER, T.REVIEW_METHOD_SYSTEM, T.REVIEW_METHOD_USER),
    "experiment": ("Experiment", T.EXPERIMENT_SYSTEM, T.EXPERIMENT_USER, T.REVIEW_EXPERIMENT_SYSTEM, T.REVIEW_EXPERIMENT_USER),
}


class ParseError(Exception):
    pass


# ------------------------------------------------------------------ output parsing

def _field(text: str, name: str, nxt: str | None) -> str | None:
    lab = rf"\**\s*{name}\s*\**\s*:?\s*\**"
    stop = rf"(?=^\s*\**\s*{nxt}\s*\**\s*:)" if nxt else r"\Z"
    m = re.search(rf"^\s*{lab}(.*?){stop}", text, re.S | re.M | re.I)
    return m.group(1).strip() if m else None


def parse_artifact(text: str, label: str, request_id: str) -> tuple[str, str]:
    body, rationale = _field(text, label, "Rationale"), _field(text, "Rationale", None)
    if not body or not rationale:
        raise ParseError(f"{request_id}: expected '{label}:' and 'Rationale:' sections")
    return body, rationale


def parse_review(text: str, request_id: str) -> dict:
    m = re.search(r"Rating\s*\(1-5\)[\s*:]*([1-5])\b", text, re.I)
    if not m:
        raise ParseError(f"{request_id}: no 'Rating (1-5): <1-5>' found")
    return {"review": _field(text, "Review", "Feedback") or "",
            "feedback": _field(text, "Feedback", r"Rating\s*\(1-5\)") or "",
            "rating": int(m.group(1))}


# ------------------------------------------------------------------------- context

def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))


class Context:
    """Everything the templates need, loaded once."""

    def __init__(self):
        self.cfg = read_json(IDEATION / "config.json")
        lit = read_json(DATA / "literature.json")
        ents = read_json(DATA / "retrieved_entities.json")
        t, rel = lit["target"], lit["related"]
        self.base = {
            "paper_title": t["title"], "paper_abstract": t["abstract"],
            "ref_titles": _numbered([r["title"] for r in rel]),
            "ref_abstracts": _numbered([r["abstract"] for r in rel]),
            "entities": ", ".join(e["entity"] for e in ents["retrieved"]),
            "n_refs": len(rel), "n_ents": len(ents["retrieved"]),
        }
        self.gaps = self._gaps() if self.cfg["use_gaps"] else ""
        self.constraints = (T.CONSTRAINTS_BLOCK.format(constraints=_numbered(self.cfg["constraints"]))
                            if self.cfg["use_constraints"] else "")

    @staticmethod
    def _gaps() -> str:
        """D1: the prior assignment's twelve gaps, each with its assessed reason."""
        merged = read_json(ROOT / "limitations/data/master_merged.json")["consolidated"]
        verdict = read_json(ROOT / "limitations/data/dismissal_scores.json")["scores"]
        lines = []
        for g in sorted(merged, key=lambda x: x["rank"]):
            v = verdict[g["id"]]
            first = re.split(r"(?<=\.)\s", g["statement"], maxsplit=2)
            lines.append(f"{g['id']}. {g['title']}. {' '.join(first[:2])} "
                         f"[Assessment: {v['reason']}. {v['note']}]")
        return T.GAPS_BLOCK.format(gaps="\n".join(lines))


# ---------------------------------------------------------------------- the agent

class ResearchAgent:
    def __init__(self, llm, ctx: Context):
        self.llm, self.ctx, self.R = llm, ctx, ctx.cfg["refinement_rounds"]

    def _gen_prompt(self, kind, slots, diversity, refine):
        label, sys_t, user_t, _, _ = KINDS[kind]
        add = ""
        if kind == "problem":
            add += self.ctx.gaps
        add += self.ctx.constraints
        add += diversity + refine
        fill = {**self.ctx.base, **slots, "addenda": add}
        return sys_t.format(**fill), user_t.format(**fill)

    def _review_prompt(self, kind, metric, slots):
        _, _, _, sys_t, user_t = KINDS[kind]
        add = self.ctx.constraints if metric == "Feasibility" else ""   # D2
        fill = {**self.ctx.base, **slots, "metric": metric,
                "criteria": T.criteria_text(kind, metric), "addenda": add}
        return sys_t.format(**fill), user_t.format(**fill)

    def _call(self, tag, system, user, role):
        rid = f"{tag}-{stable_id(role, system, user)}"
        return rid, self.llm(rid, system, user, role=role)

    def develop(self, kind: str, key: str, upstream: dict, previous: list[str]) -> dict | None:
        """Generate one artifact and refine it R times. Returns None while pending."""
        label = KINDS[kind][0]
        field = {"problem": "problem", "method": "method", "experiment": "experiment"}[kind]
        diversity = (T.DIVERSITY_BLOCK.format(kind=kind, previous=_numbered(previous)) if previous else "")
        history, refine = [], ""
        for rnd in range(self.R + 1):
            s, u = self._gen_prompt(kind, upstream, diversity, refine)
            rid, text = self._call(f"{kind}-{key}-r{rnd}-gen", s, u, "generator")
            if text is None:
                return None
            body, rationale = parse_artifact(text, label, rid)
            slots = {**upstream, field: body, f"{field}_rationale": rationale}
            reviews, waiting = {}, False
            for metric in T.CRITERIA[kind]:
                s2, u2 = self._review_prompt(kind, metric, slots)
                rrid, rtext = self._call(f"{kind}-{key}-r{rnd}-rev-{metric}", s2, u2, "reviewer")
                if rtext is None:
                    waiting = True
                    continue
                reviews[metric] = parse_review(rtext, rrid)
            if waiting:
                return None
            history.append({"round": rnd, "text": body, "rationale": rationale, "reviews": reviews,
                            "mean": round(sum(r["rating"] for r in reviews.values()) / len(reviews), 3)})
            if rnd < self.R:                                               # D4
                refine = T.REFINE_BLOCK.format(
                    kind=kind, draft=f"{label}: {body}\nRationale: {rationale}",
                    reviews="\n".join(f"- {m} ({r['rating']}/5): {r['review']} Feedback: {r['feedback']}"
                                      for m, r in reviews.items()))
        return {"key": key, "kind": kind, "final": history[-1], "history": history}
