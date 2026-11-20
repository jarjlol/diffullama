# Operator fallback assets

These files support A3 while Neel/Aalhad's T1/T3/T4/N4 contributions are pending.
They are operator-authored and are not attributed to those members. Reviewed peer
assets should replace them through an explicit new task export, before search.

- `assets/memory`: R1–R8,17 audited ideas,83 evidence-linked findings, curated
  experience with the forbidden leverage removed, and operational strategies.
- `build_support_memory.py`: reproduce findings and curated experience from repo
  root; checks baseline numeric pointers and evidence hashes. No surrogate LLM/UCB
  run is claimed. GPU priority continues to use the authoritative tiers.
- `tune_majority.py`: six protocol-specified variants; pick on dev, freeze, then gate.
  Operator results and frozen selector: `results/iter2/aryan/support-N4/`.
  The full N4 semantic taxonomy and leave-one-model-out analysis remain Neel's work.
- `EXTRACTION_REVIEW.md`: agent review of20 dev candidates, with limitations. Do not
  inject answer-bearing audit details into the search instructions.
- `order_check.py`: external gold-free regression with fresh forward/reverse and
  isolated/repeated calls. Official evaluator unchanged. Not a purity proof/sandbox.
- `refresh_pack.py`: reassemble only approved dev/gate membership into a new copy.
- `operator_start_guard.py`: duplicate-consumer guard; no GPU access or signals.

`prepare_arbor_iter2.py --assets-dir codification/a3_support/assets
--preserve-evaluator --allow-no-research-report` uses this explicitly selected
fallback. It still rejects incomplete features, wrong split sizes, duplicate items,
missing fidelity, missing seed, or an existing export destination. It CPU-evaluates
the seed and runs the external order regression. Without `--preserve-evaluator`,
the exact evaluator-patch hash still requires a preserved actual team agreement.
The optional N3 omission makes no novelty claim. No fake agreement receipt is needed
or created when leaving the protected evaluator unchanged.

Native Arbor0.1.4 plugin loading passed on Blackwell21Nov02:52. Nine preparation
boundary tests, three order regressions, the singleton regression and an end-to-end
CPU/mock export/reuse test passed locally. Preparation and order checks also passed
on Blackwell; the real frozen seed passes order checks on400 dev and400 gate items.
Passing preparation is not an Arbor search, a final result, or permission to stop
Tier1 early. vLLM starts only with its ready consumer and staffed review gates.
