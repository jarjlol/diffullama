# Upstream model code — triggered detail

> Load only when the task matches Triggers in `AGENTS.md`:
> inference, sampling, remasking, attention, tokenizer, shift, mask token.

Covers the vendored DiffuLLaMA inference path: `inf_diffullama.py` → `model.py` →
`attention_patch.py`. `README_SIMPLIFIED.md` is a from-zero walkthrough of the same path
and is the better starting point if you have not read this code before.

These traps are verified against the code in this repo (file:line references below), not
inherited from the upstream paper's prose.

## Must do

- **Account for the shift when mapping source positions to token positions.** The model
  inherits an autoregressive shift: position *i* predicts the token at *i+1*
  (`model.py:120`, and the final `x0 = x0[:,1:]` at `model.py:156`). The invariant is
  `canvas_position = returned_index + 1`. Getting this wrong targets the token *next to* the
  one you meant — it runs end to end, raises nothing, and quietly degrades results.
- **Write an invariant test before building anything that depends on that mapping.**
  `audit/test_real_tokenizer.py` is the existing one; it needs `transformers` only, no torch
  and no GPU.
- **Use `src_mask` for freezing.** `model.py:102` (`maskable_mask = ~src_mask`) already
  supports arbitrary non-contiguous remasking with no sampler changes: pass the current draft
  as `input_ids` and set `src_mask = 1` everywhere you want frozen.
  Worked example: `inf_diffullama.py:65`.
- **Set the unmasking policy explicitly per backbone.** These models do not share one.
  DiffuLLaMA unmasks in *random* order (`model.py:133`), while other diffusion LMs in this
  space default to confidence-order or ship `alg="origin"` (random) plus `temperature: 0.0`.
  Comparing across backbones without fixing this confounds your result with sampler
  differences.

## Must not

- Do not build a confidence baseline on `x0_scores`. It is the log-prob of the *sampled*
  token (`model.py:116`), i.e. sampling luck rather than model certainty — the max-based
  form is commented out at `model.py:113`, and the value is never returned
  (`model.py:158`). A baseline built on it is unfairly weak, which biases comparisons toward
  whatever you are proposing.
- Do not assume targeted remasking saves compute. `model.py:139` forwards the entire
  sequence every denoising step regardless of how many positions are masked — there is no
  KV cache. Repairing 5 tokens costs what regenerating 500 costs at equal step count. The
  available win is quality, not efficiency.
- Do not assume the inference attention mask does anything. With `attn_mask_ratio=1.0`,
  `get_anneal_attn_mask` returns an all-zeros additive mask — mathematically identical to no
  mask — but as a dense 4-D float tensor that forces the eager attention path
  (`inf_diffullama.py:24` defaults to `"eager"`).

## Example

```python
# Freeze everything except the positions you intend to regenerate.
src_mask = torch.ones_like(input_ids)
src_mask[0, target_positions] = 0        # 0 = maskable, 1 = frozen
inputs = {"input_ids": input_ids, "src_mask": src_mask}
# remember: target_positions are canvas positions, i.e. returned_index + 1
```
