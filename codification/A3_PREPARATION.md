# Preparing Arbor iteration 2

Owner: Aryan, with Aaditya operating the GPU queues. Tracker: issue #4, latest ownership
correction `7047475382`. This is preparation, not a completed Arbor run. Do not start a
server or claim idle GPUs while prerequisites are missing.

`prepare_arbor_iter2.py` checks inputs and exports a new private task repository. It does
not change the original task pack, launch Arbor, or read the final split. Only the approved
Dream/LLaDA models enter this export; this model scope and the evaluation patch need a
recorded team agreement before export.

## Inputs and owners

| Input | Owner | Required evidence |
|---|---|---|
| `memory/RED_LINES.md`, `IDEA_LIBRARY.md`, `eval_order_check.patch`; `rerank_task/plugins/rerank_guard.yaml` | Aalhad, T1 | Rule audit; passing selector and failing stateful-selector checks |
| `memory/FINDINGS.md`, `FINDINGS.jsonl`, `EXPERIENCE_CURATED.md` | Neel, T3 | Curated findings; banned cross-model cache recorded as a pitfall |
| `memory/STRATEGIES.md` | Aalhad, T4 | Failures and fixes with sources |
| `ANALYSIS_ITER2.md`, frozen tuned-majority `solution.py` | Neel, N4 | Dev-selected variant, dev/gate results and a `Failure classes for A3` section containing counts/labels/examples, no gold answers |
| Cut AR scores and DLM self-score on dev/gate items | Aaditya, A2 | `ar.qwen7b-cut.nll_sum` and `dlm.selfscore` present for every candidate |
| Fidelity aggregate JSON | Aaditya/Aryan | `configs.dream-sampler` and `configs.llada-published`, each with n and unique_uids at least 100; source digests retained |
| Member-authored agreement receipt | Team | JSON fields `author`, `url`, `body`, `models` = `["dream-base", "llada-base"]`, `eval_patch_sha256`; preserve actual agreement, do not fabricate it |
| `memory/RESEARCH_REPORT.md` | Neel, N3 | Verified published mechanisms; optional only with explicit `--allow-no-research-report` |

The checker validates presence, dataset shape, feature availability and hashes; it cannot
prove a review occurred or an agreement was authored by the named person. Read the linked
member comment, review T1's patch and plugin, and inspect the frozen seed before export.
Do not pass the full test500 task as `--task-dir`: test must already be the 200-question gate.
Final300 stays outside this task. New step-budget candidates are excluded from this export.

## Check and prepare

CPU only. Use paths to the completed inputs, not placeholder values:

```bash
python3 codification/prepare_arbor_iter2.py \
  --task-dir /path/to/assembled-dev200-gate200-task \
  --seed /path/to/N4-frozen-solution.py \
  --fidelity /path/to/fidelity_snapshot.json \
  --agreement /path/to/member-agreement.json \
  --json /tmp/a3-readiness.json
```

Exit 2 means not ready; no export occurs. The JSON lists missing inputs and source hashes.
When ready, repeat with `--dest /home/antpc/arbor-rerank-iter2`. Existing destinations
are refused. A failed export is left for inspection and must not be launched. The exporter
copies only the evaluator, seed, plugin, dev and gate data; applies T1's patch to the copy;
injects curated memory and the N4 failure block in `INSTRUCTION.md`; evaluates the seed;
and commits the clean task with Aryan's verified Git identity. It never uses the upstream
exporter's fallback author `codify@localhost`.

The instruction makes the margin a reviewer rule: at least 3 net gate questions on each
non-floor model relative to frozen tuned majority, an ablation before merging, paired final
comparison and frozen transfer before a progress claim. It does not present Arbor's
`merge_threshold` as an automatic enforcement mechanism.

## Verify native plugin loading before a GPU request

Arbor on Blackwell was independently checked on 21 Nov: `arbor-agent` version **0.1.4**.
Its installed `arbor/plugins/base.py` exposes `load_plugin(name, search_dirs, strict=True)`.
From the prepared task, use the installed runtime without loading an LLM:

```bash
~/venvs/arbor/bin/python - <<'PY'
from pathlib import Path
from arbor.plugins.base import load_plugin
p = load_plugin('rerank_guard', [Path('plugins')], strict=True)
assert p.schema_version == 1
assert p.eval_contract['metric_direction'] == 'maximize'
assert p.eval_contract['eval_cmd'] == 'bash {cwd}/eval.sh dev'
assert {'task.py', 'eval.py', 'eval.sh', 'baselines.py', 'data/**'} <= set(p.protected_paths)
assert all(getattr(p, key).strip() for key in (
    'meta_init_inject', 'meta_ideate_inject', 'meta_decide_inject',
    'meta_preamble_inject', 'sub_workflow_inject', 'sub_preamble_inject'))
assert p.convergence['min_experiments'] == 2
assert p.convergence['warn_after'] == 2
assert p.convergence['force_after'] == 3
assert p.convergence['parent_exhaustion_count'] == 2
print('native rerank_guard plugin contract passed')
PY
```

`arbor config show` describes user config in this installed version; do not treat it alone
as proof that the project plugin loaded. Also inspect the effective run metadata when
Arbor starts. Run T1's order-dependence regression on the prepared evaluator, verify source
and frozen manifest hashes, and keep evaluator/data/config fixed throughout the run.

## Launch and review handoff

Only after the above checks and T1/T3/T4/N4/A2 are ready, send Aaditya the exact launch
command as an `ask` on #4 and arrange a staffed review window. The request is:

```bash
cd /home/antpc/arbor-rerank-iter2
HF_HUB_OFFLINE=1 OPENAI_API_KEY=dummy ~/venvs/arbor/bin/arbor run \
  "$(cat INSTRUCTION.md)" --yes --yes-cwd /home/antpc/arbor-rerank-iter2 \
  --run-name iter2
```

This consumes the vLLM service only when Aaditya has started it with its consumer.
Review replies must use `/approve`, `/skip`, `/edit ...` or `/answer ...`; log them
verbatim. Never allow the unattended review timeout to approve a proposal. Stop the
service after the consumer ends, through its owner. Do not kill another user's process.
Tier 1 jobs keep priority. Paper results freeze: **22 Nov 2026 18:00 IST**.

After the run, freeze the selector and use the held-out final split once; report paired
statistics, the ablation and extra compute. A second dataset remains a team choice.
Do not select a transfer dataset or claim a positive result without the recorded evidence.
