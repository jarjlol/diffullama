"""Run the ResearchAgent pipeline end to end. Resumable: re-run after supplying responses.

    python3 ideation/scripts/run_pipeline.py                     # manual backend (default)
    python3 ideation/scripts/run_pipeline.py --backend openai    # needs LLM_API_KEY
    python3 ideation/scripts/run_pipeline.py --estimate          # call budget only

Stages
  1  literature      target + related (citing-side) papers           build_literature.py
  2  entities        knowledge store K + Eq. 2 retrieval             build_entity_store.py
  3  problems        N candidate problems, each reviewed + refined   Tables 6, 9, 13
  4  SELECTION GATE  the team picks 1-2 problems -> data/selected_problems.json
  5  ideas           per problem: N methods, each with an experiment Tables 7, 8, 10, 11, 14, 15
  6  rank + render   top-k per problem -> output/research_problems_and_ideas.md

Exit codes: 0 done, 2 waiting on LLM responses, 3 waiting on the selection gate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, IDEATION, OUT, read_json, write_json, write_text  # noqa: E402
from llm import LLM  # noqa: E402
from research_agent import Context, ResearchAgent  # noqa: E402

HERE = Path(__file__).parent


def develop_many(agent, kind, keys, upstream_of, stage_note):
    """Round-0 drafts serially (each sees the earlier ones for diversity), then refine all."""
    r0 = {}
    for key in keys:
        prev = [r0[k] for k in r0]
        s, u = agent._gen_prompt(kind, upstream_of(key), _div(kind, prev), "")
        rid, text = agent._call(f"{kind}-{key}-r0-gen", s, u, "generator")
        if text is None:
            print(f"  [{stage_note}] waiting on round-0 draft {key}")
            return None, r0
        from research_agent import KINDS, parse_artifact
        r0[key] = parse_artifact(text, KINDS[kind][0], rid)[0]
    results = {}
    keys_list = list(r0)
    for i, key in enumerate(keys_list):
        results[key] = agent.develop(kind, key, upstream_of(key), [r0[k] for k in keys_list[:i]])
    return results, r0


def _div(kind, prev):
    import templates as T
    from research_agent import _numbered
    return T.DIVERSITY_BLOCK.format(kind=kind, previous=_numbered(prev)) if prev else ""


def estimate(cfg) -> int:
    per = (cfg["refinement_rounds"] + 1) * 6
    probs = cfg["n_problem_candidates"] * per
    try:
        n_prob = len(read_json(DATA / "selected_problems.json")["selected"])
    except (OSError, ValueError, KeyError):
        n_prob = 2
    ideas = n_prob * cfg["n_ideas_per_problem"] * 2 * per
    print(f"calls per artifact : {per}  ((R+1) generations + 5(R+1) reviews, R={cfg['refinement_rounds']})")
    print(f"problem stage      : {probs}")
    print(f"idea stage ({n_prob} prob): {ideas}  ({cfg['n_ideas_per_problem']} methods + experiments per problem)")
    print(f"total              : {probs + ideas}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="manual", choices=["manual", "openai", "gemini", "mock"])
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--auto-select", type=int, default=0,
                    help="select the top-N problems automatically (testing only; the real gate is a team decision)")
    args = ap.parse_args()
    cfg = read_json(IDEATION / "config.json")
    if args.estimate:
        return estimate(cfg)

    for script in ("build_literature.py", "build_entity_store.py"):
        if subprocess.run([sys.executable, str(HERE / script)], capture_output=True).returncode:
            raise SystemExit(f"{script} failed")
    print("stages 1-2  literature + entity store: done")

    llm = LLM(args.backend)
    agent = ResearchAgent(llm, Context())

    # ---- stage 3: problems ----------------------------------------------------------
    pkeys = [f"P{i}" for i in range(1, cfg["n_problem_candidates"] + 1)]
    problems, _ = develop_many(agent, "problem", pkeys, lambda k: {}, "problems")
    if problems is None or any(v is None for v in problems.values()):
        return _pending(llm, "problem stage")
    ranked = sorted(problems.values(), key=lambda p: -p["final"]["mean"])
    write_json(DATA / "problems.json", ranked)
    _write_problem_menu(ranked)
    print(f"stage 3     problems: {len(ranked)} candidates -> output/problem_candidates.md")

    # ---- stage 4: selection gate ----------------------------------------------------
    sel_path = DATA / "selected_problems.json"
    if args.auto_select and not sel_path.exists():
        write_json(sel_path, {"selected": [p["key"] for p in ranked[:args.auto_select]],
                              "selected_by": "auto (--auto-select, testing only)", "reason": ""})
    if not sel_path.exists():
        print("\nSELECTION GATE: the team picks 1-2 problems. Read output/problem_candidates.md, then write")
        shown = sel_path.relative_to(IDEATION.parent) if sel_path.is_relative_to(IDEATION.parent) else sel_path
        print(f'  {shown}  as  {{"selected": ["P2"], "selected_by": "...", "reason": "..."}}')
        return 3
    selected = read_json(sel_path)["selected"]
    by_key = {p["key"]: p for p in ranked}
    assert 1 <= len(selected) <= 4 and all(s in by_key for s in selected), f"bad selection: {selected}"

    # ---- stage 5: ideas ------------------------------------------------------------
    all_ideas, waiting = {}, False
    for pk in selected:
        p = by_key[pk]["final"]
        up = {"problem": p["text"], "problem_rationale": p["rationale"]}
        mkeys = [f"{pk}-M{j}" for j in range(1, cfg["n_ideas_per_problem"] + 1)]
        methods, _ = develop_many(agent, "method", mkeys, lambda k: up, f"{pk} methods")
        if methods is None:
            waiting = True
            continue
        ideas = []
        for mk, m in methods.items():
            if m is None:
                waiting = True
                continue
            exp = agent.develop("experiment", mk.replace("-M", "-E"),
                                {**up, "method": m["final"]["text"], "method_rationale": m["final"]["rationale"]}, [])
            if exp is None:
                waiting = True
                continue
            ideas.append({"key": mk, "method": m, "experiment": exp})
        all_ideas[pk] = ideas
    if waiting:
        return _pending(llm, "idea stage")

    # ---- stage 6: rank + render ----------------------------------------------------
    write_json(DATA / "ideas.json", all_ideas)
    import render
    render.main()
    print(f"stage 6     ranked + rendered -> output/research_problems_and_ideas.md  ({llm.calls} new model calls)")
    return 0


def _pending(llm, where) -> int:
    ids = sorted(set(llm.pending))
    write_text(DATA / "llm" / "PENDING.txt", "\n".join(ids) + "\n")
    print(f"\n[{where}] {len(ids)} LLM response(s) pending.")
    print("  prompts   : ideation/data/llm/requests/<id>.md")
    print("  answers   : ideation/data/llm/responses/<id>.md   (same id; list in data/llm/PENDING.txt)")
    print("  then re-run this script. Completed work is cached and never re-requested.")
    return 2


def _write_problem_menu(ranked):
    from templates import CRITERIA
    lines = ["# Candidate research problems\n",
             "Generated by ResearchAgent's problem-identification stage (Table 6), each reviewed on five",
             "criteria (Tables 9, 13) and refined. **The team selects 1-2** in `data/selected_problems.json`.\n"]
    for p in ranked:
        f = p["final"]
        scores = " | ".join(f"{m} {f['reviews'][m]['rating']}" for m in CRITERIA["problem"])
        lines += [f"## {p['key']} — mean {f['mean']:.2f}/5", f"*{scores}*\n",
                  f"**Problem.** {f['text']}\n", f"**Rationale.** {f['rationale']}\n"]
    write_text(OUT / "problem_candidates.md", "\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
