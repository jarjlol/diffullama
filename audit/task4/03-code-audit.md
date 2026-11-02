# 03 — Code audit of `codification/`

Scope: every file in `codification/pipeline/`, `codification/rerank_task/` and the new
`codification/campaign/`. Method: line-by-line reading, then checks against the upstream sampler
(`model.py`, `attention_patch.py`), the LLaDA `generate.py` at the pinned commit, the Hugging Face
model configs, and two live probes on Blackwell's CPU. Severity: **high** = wrong result or lost
run, **medium** = biased result or wasted time, **low** = cosmetic or defensive.

## Fixed on 2026-11-02

| # | Severity | Where | Problem | Evidence | Fix |
|---|---|---|---|---|---|
| C-1 | high | `pipeline/config.json`, `generate_candidates.py` | DiffuGPT-S has `max_length` 512 (anchor App. B: trained at 512); a 4-shot prompt is 581 GPT-2 tokens at the median, so `Anchor.sample` raises on question 1 | real GSM8K prompts tokenised with GPT-2: 1319/1319 overflow | per-model `n_shots`/`gen_tokens`; DiffuGPT-S 1-shot, 192 tokens (0/1319 overflow) — commit 42c8866 |
| C-2 | medium | `config.json` | DiffuLLaMA's CKA base `meta-llama/Llama-2-7b-hf` is gated | HF API `gated: manual` | `NousResearch/Llama-2-7b-hf`, identical SHA-256 for both shards and `tokenizer.model` — 42c8866 |
| C-3 | high (schedule) | `config.json` | Dream/LLaDA at 256 steps ≈ 45 GPU-h each on the subset | FLOP estimate | 64 steps — 1530aa8 |
| C-4 | **high** | `pipeline/extract_hidden.py:94-96` (before fix) | DiffuGPT run with `attention_mask=None`. `model.py:43` makes GPT-2 bidirectional with `attn.bias.fill_(True)`, but that buffer is only read by *eager* attention. DiffuGPT loads as `GPT2SdpaAttention`, and the patched forward (`attention_patch.py:68-80`) turns a `None` mask into `is_causal=True`. DiffuGPT's CKA vs GPT-2 would have been computed **causally**, inflating "preservation" | CPU probe on Blackwell: changing the last token moves earlier hidden states by **0** with `None`, by **4.36** with the all-zero 4D mask | always pass the all-zero 4D mask (what `generate_samples` does), plus `check_attention()`, which aborts unless a DLM is bidirectional and a base is causal — 36dbff2 |
| C-5 | medium | `generate_candidates.py`, `score_candidates.py` | idea 1.1 asks for T_gen / T_score; nothing was recorded per question | — | `gen_seconds`, `score_seconds` per row — 36dbff2 |
| C-6 | low | `pipeline/selftest.py` on Windows | `bash` resolves to WSL's System32 `bash.exe` | `execvpe(/bin/bash) failed` | not a pipeline bug; run on Linux or put Git Bash first |

## Open findings

| # | Severity | Where | Problem | Recommendation |
|---|---|---|---|---|
| C-7 | medium | `rerank_task/research_config.yaml:18`, `eval.sh` | Arbor gates every merge on `eval.sh test`. Over 12 cycles the "held-out" score becomes a selection target, so the final test number is optimistic. With 5 × 500 test items the SE of macro accuracy is ≈ 0.01 absolute, while the merge threshold (1% relative ≈ 0.004) sits *below* noise | three-way split: iterate on dev-A, gate on dev-B (the other half of dev), report test **once** at the end. No extra generation. Also set the threshold to about 2 SE |
| C-8 | medium | `research_config.yaml` `ui.interaction_mode: auto` | contradicts the README and the brief's feedback requirement | `review` |
| C-9 | ✅ fixed 97ebc8d | `extract_hidden.py` pooling `out.float().mean(1)` | mean over all 128 positions includes position 0, where causal LMs park a very large "attention-sink" activation even without BOS. Linear CKA is dominated by such outlier directions, so base-vs-DLM CKA can reflect the sink, not preservation | also store the mean over positions 1…127, and report CKA both ways. It must happen before the hidden-state jobs run (they are last in the campaign queue) |
| C-10 | ✅ fixed 97ebc8d | `measure_preservation.py` / `cka.py` | biased linear CKA with n = 2000 samples and d up to 4096 dimensions. The biased estimator is inflated when n is small relative to d, and differently so for d = 768, 1024, 3584 and 4096, which differ across models | add debiased CKA (unbiased HSIC, Song et al. 2012) next to the biased one. CPU-only, runs on the stored activations |
| C-11 | low | `generate_candidates.py:87` | `[bos] + tok.encode(prompt)` gives two BOS tokens for LLaMA tokenisers (encode adds one) | matches upstream `inf_diffullama.py:63`, so keep it for fidelity, but record it |
| C-12 | low | `score_candidates.py:71-74` | context and answer tokenised separately (`" " + t`), so the boundary token can differ from joint tokenisation | harmless for ranking, since every candidate is treated the same; note only |
| C-13 | low | `rerank_task/task.py:19,60` | `trim` only cuts at a *newline* + `Question:`; the last-number fallback extracts some number from almost any text, so the extraction rate overstates well-formedness | report the `####`-marker rate separately in the gate |
| C-14 | medium | `pipeline/analyze.py:72`, `baselines.py:80` | headroom is `None` when oracle = first. A model with ~0% accuracy is silently dropped from the correlation, which can leave n < 3 → no result | report n explicitly in the iteration-1 text. Mitigation is more models (`07` P1-1) |
| C-15 | low | `solution.py:30` | seed ties broken by index (fine) | none |
| C-16 | low | `campaign.py` | the `vllm` substring rule is deliberately conservative: an unrelated process with `vllm` in its arguments (e.g. `tail -f vllm.log`) blocks launches. The block is visible in `status.json` `waiting_on` | acceptable; it fails safe |

## What was checked and is correct

- **Anchor sampler offset** (`generate_candidates.py`): the shift drops canvas position 0, and the
  run-time asserts that the returned prompt region equals the prompt catch an off-by-one.
- **LLaDA**: `generate()` at f2683fd takes `steps, gen_length, block_length, temperature, cfg_scale,
  remasking`. With `block_length = gen_length` there is one block, and `steps % num_blocks == 0`
  holds for 64. The single-batch assert applies only to "iLLaDA" model names, not to `LLaDA-8B-Base`.
- **Dream**: `alg="entropy"` is enforced, and `temperature > 0` is asserted.
- **Scorer causality**: `HFScorer._check_causal` aborts if the attention patch leaked into the scorer.
- **Leakage**: `eval.py` statically bans file and OS access in `solution.py`; `assemble.py` asserts no
  gold in items; `make_arbor_repo.sh` refuses a pack with a gold field.
- **Reproducibility**: `prepare_data.py` on Blackwell regenerated `gold.jsonl` byte-for-byte (`git status` clean).
- **Selftest** passes after every change today (CPU, mock models).

## Update — P0-4 done (2026-11-02, commits 97ebc8d, 4bf0551, a5f2c02 on `assignment/codification`)

- **C-9 / C-10 fixed.** `extract_hidden.py` saves `layers` (all positions, as specified) and
  `layers_nosink` (positions 1..end). `cka.py` adds `cka_debiased_torch` (unbiased HSIC) with a
  pure-Python reference. `measure_preservation.py` keeps the specified score as `mean` and adds three
  variants. `analyze.py` reports Spearman and permutation p under each.
- **Synthetic demonstration (Blackwell CPU, scorer venv):** 2000 × 768 activations, DLM = 0.6·base +
  noise, plus one large outlier direction with independent magnitudes. Results: plain CKA, all
  positions **0.0007**; plain, no outlier **0.537**; debiased, no outlier **0.360**. One outlier
  direction is enough to decide linear CKA, and debiasing removes about a third of the chance
  similarity at this n/d.
- **C-17 (new, fixed 4bf0551).** `selftest.py` built the torch cross-check tensors as float32
  (`torch.tensor(list)` default). The plain-CKA comparison failed at 1e-8 relative. It had never
  run with torch installed. Now float64; the full selftest passes on Blackwell with torch 2.7.1.
- Synced to the Blackwell clone before any hidden-state job ran (the campaign had launched nothing).
