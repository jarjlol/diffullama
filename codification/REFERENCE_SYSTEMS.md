# Which reference paper has a complete implementation?

Assessment of the four papers the codification brief offers as references, for **whether the code
behind each paper is publicly available and actually implements the paper's method**. Checked
2026-11-01 against each paper's full text (arXiv HTML) and its GitHub repository (API metadata, file
trees, and a scan of every `.py` file for the mechanisms the paper names).

## Verdict

**Only Arbor is a complete, open implementation of its paper.** The other three either never released
the system (AutoSOTA) or released a product whose code no longer contains the paper's method
(DeepScientist, EvoScientist).

| Paper | Code | Paper's method in the code? | Licence | Fit for us |
|---|---|---|---|---|
| **Arbor** — *Toward Generalist Autonomous Research via Hypothesis-Tree Refinement* ([2606.11926](https://arxiv.org/abs/2606.11926)) | [RUC-NLPIR/Arbor](https://github.com/RUC-NLPIR/Arbor), v0.1.4, active | ✅ **Yes** — `src/coordinator/`, executor/coordinator agents, hypothesis-tree export, dev/held-out evaluation, merge threshold | Apache-2.0 | ✅ `pip install arbor-agent`; local **vLLM** via LiteLLM; needs only an eval script + dev/test split + clean git repo |
| **DeepScientist** ([2509.26603](https://arxiv.org/abs/2509.26603)) | [ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist), v1.6.0 | ❌ **No** — the paper's Bayesian-optimisation loop over a Findings Memory (surrogate model, UCB/acquisition) appears in **none** of 162 `.py` files; only in prompt reference notes | Apache-2.0 | ⚠️ A different, later product ("local-first research studio") that delegates work to external coding agents (Codex, Claude Code, Kimi, OpenCode). Earliest tag is v1.5.11; the paper-era system is not tagged |
| **EvoScientist** ([2603.08127](https://arxiv.org/abs/2603.08127)) | [EvoScientist/EvoScientist](https://github.com/EvoScientist/EvoScientist), v0.4.0 | ❌ **No** — the paper's Researcher / Engineer / Evolution-Manager agents and its ideation + experimentation memories appear in **none** of 328 `.py` files at HEAD, nor in the 42 files of the earliest tag (`v0.0.1-alpha.1`) | Apache-2.0 | ⚠️ A general research-assistant product (TUI, scheduled tasks, generic memory) built on hosted model providers |
| **AutoSOTA** ([2604.05550](https://arxiv.org/abs/2604.05550)) | [tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA) — **results only** | ❌ **Not released.** The repo holds the 105 per-paper optimisation archives (2.3 GB). The system ships separately as a packaged npm CLI (`.tgz`), not as source | MIT (archives) | ❌ The CLI drives **Claude Code** as a subprocess, needs a Claude or OpenRouter API key, and requires `autosota login` to accept terms of service |

## Evidence

- **Arbor.** All 169 `.py` files scanned: `coordinator` in 70 files, `executor` in 49, `held-out` in
  14, a hypothesis-tree module (`src/experience.py`), and LiteLLM in 8. The README and the paper
  describe the same coordinator + executor + Idea Tree design, with git-worktree isolation per
  executor and merge-on-held-out-gain. Runnable example: `examples/algotune_knn/`.
- **DeepScientist.** The paper names "Findings Memory" 12 times, "Bayesian" 8, "surrogate" 4 and "UCB" 3.
  The code has none of them. The paper's abstract promised to "open-source all experimental logs and
  system code"; what is published is v1.5+ of an evolved product.
- **EvoScientist.** The paper names its Researcher Agent and Engineer Agent 14 times each, and its
  Evolution Manager, ideation memory and experimentation memory 12 each. None appears in the code at
  any tag checked.
- **AutoSOTA.** Top-level repository contents are `AutoSOTA.pdf`, `CVPR-2026/` and `ICML-2026/`.
  Releases attach only `autosota-X.Y.Z.tgz`. Its `cli_guide.md` documents install via
  `npm install -g ./autosota-X.Y.Z.tgz`, a Claude Code backend, and a terms-acceptance login.

**Limits of this check.** The scan is name-based. A codebase could implement an idea under different
names, so "not found" means the paper's architecture is not *identifiable* in the code, not proof
that no part of it survives. Arbor's positive result is stronger evidence: it was cross-checked
against the README and the paper's own description.

## Why Arbor fits this assignment

The brief asks us to codify the best idea, find and fix its specification gaps or infeasibilities,
and report results against baselines or the state of the art. Arbor's loop is close to that brief:

- **Refine-on-failure:** hypotheses that fail are recorded with their failure modes in the Idea Tree,
  and later ideas start from them — the brief's "identify issues and refine the idea by giving
  feedback to the model".
- **Baseline discipline:** executors iterate on a dev split and an improvement is merged only if it
  beats the baseline on a **held-out** split by a set margin. That is the "does it beat the baseline"
  question, with a guard against overfitting.
- **Runs on our hardware:** an OpenAI-compatible local backend means the vLLM setup already written
  for the ideation run (`ideation/local/`) can drive it, with no paid API.

**Practical constraint to plan around.** On the workstation, Arbor's LLM server and the experiments it
launches (DiffuLLaMA inference, ~13.5 GB in bf16) share one 96 GB card. The model pair used for
ideation (~64 GB) plus DiffuLLaMA fits, but with little headroom for the KV cache. A smaller reviewer,
or a single model for Arbor, may be needed.

Ideas from the other three papers can still be borrowed as described in their papers (the brief
allows that) — for example DeepScientist's tiered "hypothesize → verify → analyze" promotion of
promising findings — but they would have to be implemented by us, not reused.
