# Established facts

Verified findings, each with its source and **how it was checked**. This is the durable output of the
audit on `audit/opus-review` — it is written here so the findings survive that branch's deletion.

**Confidence key:**
- 🟢 **PRIMARY** — checked directly against the paper text, the code in this repo, or an official config file
- 🟡 **SECONDARY** — verified by an automated research pass with a working link, not re-read personally
- 🔴 **WEAK** — inferred, estimated, or an absence-of-evidence finding

---

## About the anchor paper (DiffuLLaMA)

### F-1 🟢 Attention mask annealing was dropped at 7B, validated only at 124M/355M
Quoted verbatim from the paper:
> "For efficient implementation we enable flash-attention 2 and directly use bi-directional attention
> **without attention mask annealing**."

> "For DD loss, removing attention mask annealing and shift operations both degrade performance...
> **The mask annealing has minimal impact, so we choose to omit it for 7B adaptation** to simplify
> implementation using flash-attention 2."

**The ablation (Table 3):**

| | GPT2-S (124M) | GPT2-M (355M) |
|---|---|---|
| DD w/o anneal | 43.3 | 47.2 |
| DD (full) | 45.4 | 49.7 |
| **annealing gain** | **+2.1** | **+2.5** |

**The gain grows with scale**, yet the authors call it "minimal" and extrapolate ~20× beyond their largest
test. *Checked by reading the paper's own extracted full text in `litreview/data/anchor_fulltext.txt`.*

⚠️ **Common misreading to avoid:** several secondary sources state annealing "has minimal impact **on the
7B model**." The paper says no such thing — annealing was **never tested at 7B**.

### F-22 🟢 The only 7B annealing test targets a *different architecture* and a *modified* annealing
**NBDiff** — *From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs*,
[arXiv:2512.06776](https://arxiv.org/abs/2512.06776), submitted 7 Dec 2025, revised 30 Jan 2026, 13 authors.
This is the closest known work to Option B. **Read in full, not from a summary.** Four findings:

1. **It ablates annealing only for block-diffusion adaptation.** There is **no full-sequence-diffusion
   comparison anywhere in the paper** — the target architecture is Block-Diffusion, which is not what
   DiffuLLaMA does.
2. **Its "Annealed Attention Mask" baseline is not DiffuLLaMA's annealing.** They restructured it for
   parallel block training: *"we 'chain' the randomness of M_BD and M_OBC such that each token will not
   view each position twice in attention."* So the 7B number does **not** cleanly test the original method.
3. **Their critique is theoretical, not empirical:** *"its transition is not 'natural.' In practice,
   training sees unknown future corpora; sporadically granting early tokens access to a random subset of
   future tokens yields incomplete and potentially misleading context."*
4. **They never discuss DiffuLLaMA's 7B omission.** DiffuLLaMA is cited only as the source of the
   annealing concept.

Setup: Qwen3-4B-Base, Qwen3-8B-Base, openPangu-7B; 4000 iterations / ~30B tokens; GSM8K, MATH, HumanEval,
MBPP. On openPangu-7B, annealing scores **44.34** average vs their method **54.94**.

⚠️ *Number discrepancy to resolve:* a v1 fetch reported their comparison figure as 48.95; a v2 fetch
reported 54.94. Possibly different table rows or a v1→v2 revision. **Confirm against the current PDF
before citing.**

**Consequence for Option B: the gap is cleaner than first assessed.** The only existing 7B annealing
result uses a *modified* annealing, on a *different* adaptation target, trained for only 4000 iterations,
and justified theoretically rather than by an architecture-matched ablation. Nobody has tested
DiffuLLaMA's actual annealing at 7B for full-attention diffusion.

### F-2 🟢 The `[MASK]` token is a reused vocabulary word, inconsistently
> "In theory, we should expand the original vocabulary... However, considering practical issues on
> implementation, we can alternatively **select an existing word from the vocabulary** to serve as the
> [MASK] token... For DiffuGPT-S we use tokenid=10541 and for **DiffuGPT-M we set a new [MASK] token with
> tokenid=50257**. For DiffuLLaMA, we set tokenid=811."

Confirmed in code: `DiffuLLaMA-training/train.py:304` defaults `--mask_token 811`;
`LLaMA-Factory/src/llamafactory/model/loader.py:145,152` hardcodes 10541 / 811.

**Three things nobody checked:** no evidence token 811 or 10541 is actually low-frequency (asserted, never
measured); reuse forces the model to disambiguate mask-vs-literal-word; and **DiffuGPT-M is both the only
model with a proper new token and the better performer** — confounded with size, never disentangled.

---

## About this repository's code

### F-3 🟢 The shift invariant — `canvas_position = returned_index + 1`
Proven by `audit/test_shift_offset.py` (all checks pass). Derived from `model.py` L110→L115→L120→L124→L156.
Exact and tokenizer-independent.

**Consequence:** mapping a token seen at `returned[k]` to `src_mask[k]` is **off by one** and silently masks
the wrong token. The test demonstrates this concretely — targeting `returned[2]` masks `'b'` when `'c'` was
intended.

### F-4 🟢 `x0_scores` is the *sampled-token* log-prob, not max probability
`model.py:116` gathers the log-prob of the sampled token; the max-based form at L113 is commented out.
It measures sampling luck, not model certainty — and it is **never read and never returned** (`model.py:158`).

**Consequence:** a confidence baseline built on it would be unfairly weak, biasing any comparison in favour
of a proposed alternative. Use max softmax probability instead.

### F-5 🟢 Targeted remasking saves zero compute
`model.py:139` forwards the entire sequence every denoising step regardless of how many positions are
masked. No KV cache, no sparsity benefit. Repairing 5 tokens costs what regenerating 500 costs at equal
step count. **The only available win is quality, never efficiency** — unless steps-to-convergence differs,
which nobody has measured.

### F-6 🟢 `src_mask` is a genuine freeze/remask primitive
`model.py:102` (`maskable_mask = ~src_mask`) permits arbitrary non-contiguous remasking with no sampler
changes. Verified in `audit/test_shift_offset.py` test 3. Worked example at `inf_diffullama.py:65`.

### F-7 🟢 The inference attention mask is a no-op that blocks flash attention
With `attn_mask_ratio=1.0`, `get_anneal_attn_mask` returns an **all-zeros** additive mask — mathematically
identical to no mask. But it is a dense 4-D float tensor, and `inf_diffullama.py:24` defaults
`_attn_implementation="eager"`. Passing `None` with flash attention instead should give a solid speedup.

### F-21 🟢 The AST→token chain survives the real tokenizer, but bleeds indentation
Tested against the real `diffusionfamily/diffullama` tokenizer (`audit/test_real_tokenizer.py`), 26
statement nodes across 4 Python programs:

| Measure | Result |
|---|---|
| Mapped tokens cover the node's source | **26/26** ✅ |
| First mapped token starts *before* the node | **22/26 (85%)** |
| ...where the bled text is whitespace | 22 |
| ...where the bled text is real code | **0** ✅ |

**Cause:** SentencePiece glues the final space of an indent onto the following identifier. A 4-space
indent tokenizes as `▁▁▁` (3 spaces) + `▁disc` (space + text). So a statement at `col_offset=4` has its
first character inside a token that *also* covers the 4th indent space.

**Verdict: not fatal, but must be handled explicitly.** No code corruption — the bleed is always
whitespace. But Python indentation is semantic, so masking a statement necessarily masks part of its own
indent, and the model must regenerate the exact indent or the program breaks.

Three options: (a) snap masks to line boundaries and regenerate whole lines; (b) exclude the
leading-whitespace token and freeze the statement's first sub-token; (c) strip the indent, tokenize,
re-attach. **The naive "mask every token overlapping the AST span" rule silently does (a) without saying
so** — whichever is chosen must be stated in the design doc.

*Only relevant if Option C is chosen.*

---

## About the competitive landscape

### F-8 🟢 CDC preempts structure-guided remasking
[arXiv:2605.16829](https://arxiv.org/abs/2605.16829) *Constrained Code Generation with Discrete Diffusion*
(Shao, Cardei, Xie, Fioretto, Wang), 16 May 2026. **v1 preprint, unrefereed, no venue.**

Its MDFI operator builds a partial Code Property Graph mid-denoising → identifies an offending node → takes
its AST/dataflow neighbourhood → lifts to token spans → applies a budget → remasks. Paper's own words:
*"the localization set anchored on a witness node rather than on a heuristic score."*

### F-9 🟢 CDC Fig. 8(b) — there is an interior optimum for neighbourhood scope

| Scope | func-sec@1 |
|---|---|
| **Parent+Leaf (deployed)** | **34.3** |
| Use–Def Slice | 26.9 |
| Token-Window | 24.1 |

*Read directly from the paper.* **Two things matter here:**
1. It is a **middle optimum** — Parent+Leaf beats both the looser slice and the tighter window. Not
   "tighter is better."
2. The metric is `func-sec@1` on **CWEval, a security benchmark**. Whether the same optimum holds for
   *functional* bugs is **genuinely open** — CDC routes functional correctness through a different
   mechanism entirely.

### F-10 🟢 Dream ships with random-order decoding and greedy sampling
`Dream-org/Dream-v0-Instruct-7B`'s `generation_config.json`, fetched directly:
`"alg": "origin"` (the **random** branch), `"temperature": 0.0`, `"steps": 512`.

**Two consequences:**
1. Any Dream baseline run without explicitly setting `alg="entropy"` **silently runs random-order** — a
   silent-failure mode in the same class as the off-by-one.
2. At temperature 0, a remasked span deterministically reproduces the same tokens when surrounding context
   is unchanged. Any repair-loop design needs an explicit per-backbone temperature policy.

### F-11 🟡 LLaDA unmasks *highest*-confidence-first
`generate.py` does `torch.topk(confidence, ...)`. The config name `remasking='low_confidence'` refers to
which tokens *stay masked*, not which get unmasked. Default is also semi-autoregressive (`block_length=32`).

### F-12 🟢 Three novelty claims that must not be made
Deleted from any write-up, per F-8/F-9:
1. ~~first to use program structure to select remasking positions~~
2. ~~first non-contiguous structure-derived remask set~~
3. ~~structure beats fixed-window/random remasking~~ (as a novel *finding*)

### F-13 🟡 What CDC does **not** cover — the remaining gaps
- **No confidence-based remasking baseline at all**, at any budget. That comparison exists nowhere.
- **Purely static** — no execution feedback, no traceback, no failing-test localization anywhere.
- **Constrained generation, not repair** — no iterate-until-pass loop, and **zero repair-benchmark numbers**
  (no HumanEvalFix, DebugBench, QuixBugs, Defects4J).
- Never cites the remasking-policy literature and does not present itself as a remasking policy.

### F-14 🟡 Baselines with canonical published forms
Any "instability" baseline must cite and faithfully implement these rather than inventing a strawman:
- **STaRR** [2601.04205](https://arxiv.org/abs/2601.04205) — temporal variance + spatial deviance
- **RCR** (in MDPO) [2508.13148](https://arxiv.org/abs/2508.13148) — running confidence across steps
- **ReMDM** [2503.00307](https://arxiv.org/abs/2503.00307) — principled remasking sampler
- **Re-evaluating Confidence Remasking** [2606.12232](https://arxiv.org/abs/2606.12232) — finds WINO gives
  little benefit over plain confidence unmasking; strong motivating citation

*Note: STaRR's metrics are derived from token-confidence dynamics, i.e. internal model signals. It is an
instance of the probabilistic/spatial class, **not** a counterexample to it.*

### F-15 🟡 Generation-order falsification is already published, at scale
- [2601.15593](https://arxiv.org/abs/2601.15593) — order statistics across **58 benchmarks, 8 MDLMs up to
  100B**; concludes MDLMs do not realize arbitrary-order generation in practice
- [2608.05687](https://arxiv.org/abs/2608.05687) — the causal commitment-order intervention, on LLaDA/Dream
- [2601.13228](https://arxiv.org/abs/2601.13228) — AR models rival diffusion at any-order generation

**But every one studies from-scratch models. None studies an *adapted* model** — that gap is open.

---

## Corrected numbers (the design doc had these wrong)

### F-16 🟡
| Claim | Actual |
|---|---|
| EvalPlus "~560 problems" | **542** (HumanEval+ 164 + MBPP+ 378); 560 ≈ the deprecated pre-v0.2.0 count |
| Dream-7B "~14 GB" | **7.6B, 15.2 GB bf16** (Qwen2.5-7B backbone) |
| Dream HF IDs | Doc's IDs do not exist. Real: `Dream-org/Dream-v0-Instruct-7B`, `Dream-org/Dream-Coder-v0-Instruct-7B` |
| DiffuGPT-small "127M" | Hub artifact `diffusionfamily/diffugpt-s` is **124M** (127M is the paper's figure) |
| DiffuLLaMA "~14 GB" | 6.74B, **13.5 GB** |
| MBPP "~500" | Correct only for the test split (task_ids 11–510); full set 974, sanitized 427 |

Verified as written: LLaDA-8B-Instruct (8.0B, 16.03 GB), HumanEval = 164, MBPP's 3-asserts-per-problem
(exactly `{3: 974}`), EvalPlus's `base_input`/`plus_input` split.

⚠️ EvalPlus stores test **inputs** evaluated differentially against `canonical_solution`, not literal
assert strings — any harness must render them into asserts itself.

### F-17 🔴 Compute estimate (reasoning, not measurement)
The design doc's ~0.4 s/repair is likely optimistic by 2–4× (eager attention, dense 4-D mask blocking flash
attention, T=32–64 full-sequence forwards). Grid nearer 30–80 GPU-hours than 10–20. **This is FLOPs
arithmetic, not a benchmark** — one hour of measurement would settle it.

---

## Methodological lessons recorded

### F-18 Verify award/venue claims against the official page, not search snippets
An early search claimed "InspAIred" was the NAACL 2025 Best Paper. The official NAACL awards page showed the
real winner is "The BiGGen Bench" — InspAIred is a co-located workshop paper.

### F-19 Read full text, not abstracts, before committing to a gap
An automated pass flagged Depth Anything V2's "transparent surfaces" as an open failure mode. The paper
itself reports 83.6%/91.2% accuracy there. The real remaining gap was narrower (mirrors specifically).

### F-20 Count-check any hand-built index before use
An LLM-relevance scoring array was built with 222 entries for 218 papers. Caught by a length assertion
before use; rebuilt as an explicit `{index: score}` map that fails safe. The same off-by-N class of bug is
documented once already in the project's own REPORT.md §4.
