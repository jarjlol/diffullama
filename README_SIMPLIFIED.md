# DiffuLLaMA Inference, Explained From Zero

This is a companion to the main [`README.md`](README.md), written for someone who has
**never seen this repository before** and wants to understand, precisely, what happens
when you run:

```bash
python inf_diffullama.py --model_name diffusionfamily/diffullama --flash_attn flash_attention_2
```

It walks the *exact* call path through `inf_diffullama.py` → `model.py` →
`attention_patch.py`, names the real functions involved, and illustrates each stage
with a hand-drawn diagram. No prior diffusion-model or DiffuLLaMA knowledge assumed.

> This document only covers **inference / sampling** (generating text from a trained
> checkpoint). Training the diffusion adaptation itself is a separate process, covered
> in `DiffuLLaMA-training/` and `LLaMA-Factory/src/llamafactory/train/ddm/` — it is
> mentioned briefly in the FAQ at the bottom for orientation, not in depth.

---

## 1. The idea, in one paragraph

A normal large language model (like LLaMA-2) is **autoregressive**: it generates text
left to right, one token at a time, and each token can only "see" the tokens before it
(*causal attention*). DiffuLLaMA takes that exact same pretrained LLaMA-2 and reuses it
as something different: a **discrete diffusion denoiser**. Generation starts from a
sequence that is almost entirely a special `[MASK]` token (like a blank page), and the
model is called repeatedly — dozens of times — each time looking at the *whole*
sequence at once (bidirectional attention, not causal) and filling in a few more
`[MASK]` tokens with real words, until nothing is masked. No new architecture, no new
weights format: the same LLaMA-2 layers are reused, only *how they are called* changes.

---

## 2. File map

| File | Role |
|---|---|
| [`inf_diffullama.py`](inf_diffullama.py) | Entry point / driver script. Parses CLI args, loads the checkpoint, builds the starting (masked) input, calls the sampler, prints the decoded text. |
| [`model.py`](model.py) | Defines `DiscreteDiffusionModel` (the denoiser wrapper around LLaMA), `generate_samples()` (the reverse-diffusion sampling loop — the heart of this document), and small helpers: `get_anneal_attn_mask()`, `top_p_logits()`, `LinearNoise`. |
| [`attention_patch.py`](attention_patch.py) | Monkey-patches HuggingFace's `LlamaModel.forward`, `LlamaFlashAttention2.forward` and `GPT2Model.forward` so a custom, **non-causal** `attention_mask` can be passed straight through instead of being silently overwritten with a causal one. |

---

## 3. Glossary (read this once, refer back as needed)

| Term | Meaning |
|---|---|
| AR (autoregressive) model | The original LLaMA-2 — predicts each token only from the tokens before it (causal attention), one at a time. |
| Discrete diffusion | A generative process that starts from a fully "noised" (masked) sequence and iteratively denoises *all* positions in parallel until real text remains. |
| `[MASK]` / `tokenizer.mask_token_id` | The "noise": DiffuLLaMA's noise is *replace the token with a special mask id*, not Gaussian pixel noise like image diffusion. |
| `x` | The ground-truth / given tokens: the prompt (if any) plus placeholder zeros for the part to be generated. |
| `src_mask` | `1` where a token is given (the prompt) and must never be overwritten; `0` where it must be generated. |
| `maskable_mask` | The complement of `src_mask` (`NOT src_mask`). Shrinks every iteration as more positions are permanently revealed. |
| `xt` | The current, partially-denoised sequence at diffusion step `t` — a mix of real tokens and `[MASK]`. |
| `x0` | The model's current best full guess of the *clean* (noise-free) sequence — resampled on every forward pass. |
| `diffusion_steps` (`T`) | Number of denoising passes, default `64`. More steps ≈ slower generation, usually higher quality. |
| `logits_temp` / `topp_temp` | Sampling temperature and nucleus (top-p) cutoff applied before every token guess. |
| `shift` | An alignment fix, required because the reused LLaMA head was trained to predict "the *next* token," not "this token." Keep it `True` for these checkpoints. |
| bidirectional attention | Every position can attend to every other position, past *and* future — enabled by `attention_patch.py`. A normal LLaMA forward pass is strictly causal (past only). |

---

## 4. Diagram 1 — End-to-end pipeline

What `inf_diffullama.py` does, top to bottom, before any text comes out.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 886" width="100%" role="img" aria-label="End-to-end inference pipeline for DiffuLLaMA">
<rect x="0" y="0" width="860" height="886" fill="#f6f8fa"/>

<rect x="40" y="20" width="780" height="100" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="46" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">1 &#183; Patch HF attention (once, at import)</text>
<text x="60" y="68" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12.5" fill="#0550ae">attention_patch.replace_attention_mask()  &#8212;  attention_patch.py:459</text>
<text x="60" y="88" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Overwrites LlamaModel.forward / LlamaFlashAttention2.forward / GPT2Model.forward</text>
<text x="60" y="106" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">to accept a ready-made attention_mask instead of rebuilding a causal one.</text>

<line x1="430" y1="120" x2="430" y2="150" stroke="#57606a" stroke-width="2"/>
<polygon points="424,150 436,150 430,160" fill="#57606a"/>

<rect x="40" y="160" width="780" height="100" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="186" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">2 &#183; Load the frozen autoregressive checkpoint</text>
<text x="60" y="208" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12.5" fill="#0550ae">LlamaForCausalLM.from_pretrained(model_name)  &#8212;  inf_diffullama.py:30-37</text>
<text x="60" y="228" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Plain HuggingFace load &#8212; same weights and architecture as a normal LLaMA-2 model.</text>
<text x="60" y="246" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Nothing diffusion-specific has happened yet.</text>

<line x1="430" y1="260" x2="430" y2="290" stroke="#57606a" stroke-width="2"/>
<polygon points="424,290 436,290 430,300" fill="#57606a"/>

<rect x="40" y="300" width="780" height="118" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="326" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">3 &#183; Wrap the checkpoint as a denoiser</text>
<text x="60" y="348" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12.5" fill="#0550ae">DiscreteDiffusionModel(model, config, tokenizer, device)  &#8212;  model.py:15,25-57</text>
<text x="60" y="368" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Splits LLaMA into embed_tokens, denoise_model (the transformer body) and lm_head,</text>
<text x="60" y="386" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">then discards the original CausalLM wrapper.</text>
<text x="60" y="404" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">No weights change here &#8212; only how the pieces get called changes.</text>

<line x1="430" y1="418" x2="430" y2="448" stroke="#57606a" stroke-width="2"/>
<polygon points="424,448 436,448 430,458" fill="#57606a"/>

<rect x="40" y="458" width="780" height="118" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="484" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">4 &#183; Build the starting noised sequence x_T</text>
<text x="60" y="506" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12.5" fill="#0550ae">inf_diffullama.py:52-74</text>
<text x="60" y="526" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Unconditional: x = gen_len zero-ids, src_mask = all 0 (nothing given).</text>
<text x="60" y="544" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Conditional: x = prompt ids + zero padding, src_mask = 1 on the prompt,</text>
<text x="60" y="562" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">0 on the part still to be generated.</text>

<line x1="430" y1="576" x2="430" y2="606" stroke="#57606a" stroke-width="2"/>
<polygon points="424,606 436,606 430,616" fill="#57606a"/>

<rect x="40" y="616" width="780" height="120" rx="10" fill="#ddf4ff" stroke="#0969da" stroke-width="1.5"/>
<text x="60" y="642" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#0550ae">5 &#183; Reverse-diffusion sampling (full detail in Diagram 4)</text>
<text x="60" y="664" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12.5" fill="#0550ae">generate_samples(model, args, tokenizer, inputs)  &#8212;  model.py:80-158</text>
<text x="60" y="684" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#0550ae">Runs diffusion_steps (T, default 64) denoising passes over the whole sequence</text>
<text x="60" y="702" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#0550ae">at once, gradually turning [MASK] positions into real tokens.</text>

<line x1="430" y1="736" x2="430" y2="766" stroke="#57606a" stroke-width="2"/>
<polygon points="424,766 436,766 430,776" fill="#57606a"/>

<rect x="40" y="776" width="780" height="90" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="802" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">6 &#183; Decode tokens to text</text>
<text x="60" y="824" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12.5" fill="#0550ae">tokenizer.decode(res.tolist()[0])  &#8212;  inf_diffullama.py:58,73</text>
<text x="60" y="844" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">The finished integer id sequence becomes human-readable text.</text>
</svg>

---

## 5. Diagram 2 — What `xt` physically looks like over time

Before the mechanics, the intuition: `xt` is a row of token ids the same length as your
target output. A few are fixed (the prompt, if any). The rest start as `[MASK]` and get
replaced with real, sampled tokens a little at a time, over `T` steps, never reverting
once revealed.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" role="img" aria-label="Snapshots of xt across diffusion steps, showing mask tokens being progressively revealed">
<rect x="0" y="0" width="400" height="300" fill="#f6f8fa"/>

<text x="15" y="38" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#1f2328">t = T (start)</text>
<rect x="110" y="26" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="132" y="26" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="154" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="176" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="198" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="220" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="242" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="264" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="286" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="308" y="26" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>

<text x="15" y="88" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#1f2328">t &#8776; T/2</text>
<rect x="110" y="76" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="132" y="76" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="154" y="76" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="176" y="76" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="198" y="76" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="220" y="76" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="242" y="76" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="264" y="76" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="286" y="76" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="308" y="76" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>

<text x="15" y="138" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#1f2328">t = 2</text>
<rect x="110" y="126" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="132" y="126" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="154" y="126" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="176" y="126" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="198" y="126" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="220" y="126" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="242" y="126" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="264" y="126" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="286" y="126" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>
<rect x="308" y="126" width="18" height="18" rx="3" fill="#d0d7de" stroke="#57606a"/>

<text x="15" y="188" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#1f2328">t = 0 (done)</text>
<rect x="110" y="176" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="132" y="176" width="18" height="18" rx="3" fill="#8250df"/>
<rect x="154" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="176" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="198" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="220" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="242" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="264" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="286" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>
<rect x="308" y="176" width="18" height="18" rx="3" fill="#1a7f37"/>

<rect x="20" y="222" width="16" height="16" rx="3" fill="#8250df"/>
<text x="44" y="235" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">prompt token (from src_mask, fixed &#8212; never changes)</text>
<rect x="20" y="246" width="16" height="16" rx="3" fill="#1a7f37"/>
<text x="44" y="259" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">revealed token (sampled, then locked in for good)</text>
<rect x="20" y="270" width="16" height="16" rx="3" fill="#d0d7de" stroke="#57606a"/>
<text x="44" y="283" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">[MASK] &#8212; still unknown, resampled every step</text>
</svg>

---

## 6. Diagram 3 — Inside a single denoising pass

Every one of the `T` forward passes runs exactly this: `DiscreteDiffusionModel.forward()`.
This is a *bidirectional* pass, unlike a normal LLaMA call — the bottom box is why.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 410" width="100%" role="img" aria-label="Internals of one DiscreteDiffusionModel forward pass">
<rect x="0" y="0" width="1200" height="410" fill="#f6f8fa"/>

<rect x="20" y="80" width="160" height="120" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="40" y="104" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">xt</text>
<text x="40" y="126" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">token ids</text>
<text x="40" y="144" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">[batch, seq_len]</text>
<text x="40" y="162" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">(current noisy sequence)</text>

<line x1="180" y1="140" x2="210" y2="140" stroke="#57606a" stroke-width="2"/>
<polygon points="210,134 210,146 220,140" fill="#57606a"/>
<text x="200" y="128" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#57606a" text-anchor="middle">embed()</text>

<rect x="220" y="80" width="200" height="120" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="240" y="104" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">embed_tokens()</text>
<text x="240" y="126" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">model.py:65-66</text>
<text x="240" y="144" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">= model.model.embed_tokens</text>
<text x="240" y="162" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">token id &#8594; embedding vector</text>

<line x1="420" y1="140" x2="450" y2="140" stroke="#57606a" stroke-width="2"/>
<polygon points="450,134 450,146 460,140" fill="#57606a"/>
<text x="440" y="128" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#57606a" text-anchor="middle">x_embed</text>

<rect x="460" y="80" width="240" height="120" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="480" y="104" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">denoise_model()</text>
<text x="480" y="126" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">model.py:74</text>
<text x="480" y="144" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">bidirectional LLaMA transformer</text>
<text x="480" y="162" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">(patched attention &#8212; see below)</text>

<line x1="700" y1="140" x2="730" y2="140" stroke="#57606a" stroke-width="2"/>
<polygon points="730,134 730,146 740,140" fill="#57606a"/>
<text x="720" y="128" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#57606a" text-anchor="middle">hidden</text>

<rect x="740" y="80" width="200" height="120" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="760" y="104" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">lm_head()</text>
<text x="760" y="126" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">= get_logits()</text>
<text x="760" y="144" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">model.py:59-60,76</text>
<text x="760" y="162" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">hidden vector &#8594; vocab scores</text>

<line x1="940" y1="140" x2="970" y2="140" stroke="#57606a" stroke-width="2"/>
<polygon points="970,134 970,146 980,140" fill="#57606a"/>

<rect x="980" y="80" width="200" height="120" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="1000" y="104" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">logits</text>
<text x="1000" y="126" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">[batch, seq_len, vocab_size]</text>
<text x="1000" y="144" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">one full distribution</text>
<text x="1000" y="162" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">per position, every step</text>

<line x1="580" y1="250" x2="580" y2="210" stroke="#57606a" stroke-width="2"/>
<polygon points="574,210 586,210 580,200" fill="#57606a"/>

<rect x="420" y="250" width="320" height="140" rx="10" fill="#fff8e1" stroke="#9a6700" stroke-width="1.5" stroke-dasharray="6,4"/>
<text x="440" y="274" font-family="Helvetica,Arial,sans-serif" font-size="14" font-weight="700" fill="#7d4e00">attention_mask</text>
<text x="440" y="294" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11.5" fill="#7d4e00">get_anneal_attn_mask(...)  &#8212;  model.py:100,175-187</text>
<text x="440" y="314" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">attn_mask_ratio=1.0 &#8594; fully bidirectional:</text>
<text x="440" y="332" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">every position may attend to every other</text>
<text x="440" y="350" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">position &#8212; unlike a normal LLaMA forward.</text>
</svg>

---

## 7. Diagram 4 — The reverse-diffusion loop, step by step

This is `generate_samples()` in full: the actual algorithm that turns a fully-masked
sequence into finished text, one denoising pass at a time.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1478" width="100%" role="img" aria-label="Full step by step flow of the generate_samples reverse diffusion loop">
<rect x="0" y="0" width="900" height="1478" fill="#f6f8fa"/>

<rect x="40" y="20" width="820" height="94" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="44" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">Initialize xt at t = T</text>
<text x="60" y="66" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)  &#8212;  model.py:105</text>
<text x="60" y="86" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">maskable_mask = NOT src_mask: every position without a given prompt token starts as [MASK].</text>

<line x1="450" y1="114" x2="450" y2="144" stroke="#57606a" stroke-width="2"/>
<polygon points="444,144 456,144 450,154" fill="#57606a"/>

<rect x="40" y="154" width="820" height="94" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="178" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">First forward pass (t = T)</text>
<text x="60" y="200" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">logits = model(xt, attention_mask)  &#8212;  model.py:110</text>
<text x="60" y="220" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">One denoising pass over the WHOLE sequence at once (see Diagram 3).</text>

<line x1="450" y1="248" x2="450" y2="278" stroke="#57606a" stroke-width="2"/>
<polygon points="444,278 456,278 450,288" fill="#57606a"/>

<rect x="40" y="288" width="820" height="94" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="312" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">Sample a full-sequence guess x0</text>
<text x="60" y="334" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">top_p_logits(...) &#8594; log_softmax &#8594; Categorical(scores).sample()  &#8212;  model.py:111-115</text>
<text x="60" y="354" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Nucleus (top-p) filtering + temperature, then one stochastic token guess per position.</text>

<line x1="450" y1="382" x2="450" y2="412" stroke="#57606a" stroke-width="2"/>
<polygon points="444,412 456,412 450,422" fill="#57606a"/>

<rect x="40" y="422" width="820" height="112" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="446" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">Shift alignment (if shift=True, default)</text>
<text x="60" y="468" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">x0 = cat([x[:, 0:1], x0[:, :-1]], dim=1)  &#8212;  model.py:118-121</text>
<text x="60" y="488" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">LLaMA's head at position i predicts token i+1, so guesses are shifted right one slot</text>
<text x="60" y="506" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">to line each guess up with the position it actually describes.</text>

<line x1="450" y1="534" x2="450" y2="564" stroke="#57606a" stroke-width="2"/>
<polygon points="444,564 456,564 450,574" fill="#57606a"/>

<rect x="40" y="574" width="820" height="112" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="60" y="598" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#1f2328">Combine with what's already known</text>
<text x="60" y="620" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])  &#8212;  model.py:124</text>
<text x="60" y="640" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Positions already revealed or given as prompt keep their xt value untouched;</text>
<text x="60" y="658" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">only still-[MASK] positions take the new guess.</text>

<line x1="450" y1="686" x2="450" y2="716" stroke="#57606a" stroke-width="2"/>
<polygon points="444,716 456,716 450,726" fill="#57606a"/>

<rect x="40" y="726" width="820" height="560" rx="10" fill="#fff8e1" stroke="#9a6700" stroke-width="1.5" stroke-dasharray="6,4"/>
<text x="60" y="756" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#7d4e00">LOOP &#8212; for t = T-1 down to 1  (T-1 more denoising passes)  &#183; model.py:128-153</text>

<rect x="70" y="782" width="760" height="90" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="90" y="806" font-family="Helvetica,Arial,sans-serif" font-size="14" font-weight="700" fill="#1f2328">Pick a slice of [MASK] positions to reveal</text>
<text x="90" y="826" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11.5" fill="#0550ae">p = 1/(t+1);  masked_to_x0 = maskable_mask AND (rand() &lt; p)  &#8212;  model.py:131-133</text>
<text x="90" y="846" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">As t shrinks toward 1, p grows &#8212; more of what's left gets locked in each step.</text>

<line x1="450" y1="872" x2="450" y2="886" stroke="#57606a" stroke-width="2"/>
<polygon points="444,886 456,886 450,896" fill="#57606a"/>

<rect x="70" y="896" width="760" height="90" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="90" y="920" font-family="Helvetica,Arial,sans-serif" font-size="14" font-weight="700" fill="#1f2328">Commit those tokens into xt, permanently</text>
<text x="90" y="940" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11.5" fill="#0550ae">xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])  &#8212;  model.py:134</text>
<text x="90" y="960" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">maskable_mask = maskable_mask AND NOT masked_to_x0 &#8212; revealed tokens are never touched again.</text>

<line x1="450" y1="986" x2="450" y2="1000" stroke="#57606a" stroke-width="2"/>
<polygon points="444,1000 456,1000 450,1010" fill="#57606a"/>

<rect x="70" y="1010" width="760" height="90" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="90" y="1034" font-family="Helvetica,Arial,sans-serif" font-size="14" font-weight="700" fill="#1f2328">Re-run the forward pass on the more-revealed xt</text>
<text x="90" y="1054" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11.5" fill="#0550ae">logits = model(xt, attention_mask)  &#8212;  model.py:139</text>
<text x="90" y="1074" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">xt now carries more real context, so this guess is better-informed than the last one.</text>

<line x1="450" y1="1100" x2="450" y2="1114" stroke="#57606a" stroke-width="2"/>
<polygon points="444,1114 456,1114 450,1124" fill="#57606a"/>

<rect x="70" y="1124" width="760" height="90" rx="10" fill="#ffffff" stroke="#57606a" stroke-width="1.5"/>
<text x="90" y="1148" font-family="Helvetica,Arial,sans-serif" font-size="14" font-weight="700" fill="#1f2328">Sample + shift + combine again</text>
<text x="90" y="1168" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11.5" fill="#0550ae">same steps as boxes 3-5 above, applied to the fresh logits  &#8212;  model.py:140-151</text>
<text x="90" y="1188" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#57606a">Produces an updated full-sequence guess x0, used at the top of the next iteration.</text>

<path d="M830,1169 L850,1169 L850,827 L842,827" fill="none" stroke="#9a6700" stroke-width="2"/>
<polygon points="830,827 842,820 842,834" fill="#9a6700"/>
<text x="845" y="998" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#7d4e00" text-anchor="middle" transform="rotate(90 845 998)">repeat &#183; t -= 1</text>

<line x1="450" y1="1286" x2="450" y2="1316" stroke="#57606a" stroke-width="2"/>
<polygon points="444,1316 456,1316 450,1326" fill="#57606a"/>

<rect x="40" y="1326" width="820" height="132" rx="10" fill="#ddf4ff" stroke="#0969da" stroke-width="1.5"/>
<text x="60" y="1350" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#0550ae">Loop ends (t reached 1) &#8212; return the result</text>
<text x="60" y="1372" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" fill="#0550ae">if shift: x0 = x0[:, 1:]   then   return x0  &#8212;  model.py:155-158</text>
<text x="60" y="1392" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#0550ae">x0 already equals xt with any still-[MASK] leftovers filled by the model's final guess</text>
<text x="60" y="1410" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#0550ae">(that's what the masked_scatter in step 5 does) &#8212; nothing is left as [MASK].</text>
<text x="60" y="1430" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#0550ae">Back in inf_diffullama.py this becomes text via tokenizer.decode() &#8212; Diagram 1, step 6.</text>
</svg>

---

## 8. Quickstart (for reference)

```bash
# unconditional + a prefix-conditioned example, printed to stdout
python inf_diffullama.py --model_name diffusionfamily/diffullama --flash_attn flash_attention_2
```

Flags that change the sampling behavior described above (`inf_diffullama.py:17-24`):

| Flag | Default | Effect |
|---|---|---|
| `--diffusion_steps` | `64` | `T` in the loop above &#8212; how many denoising passes run. |
| `--logits_temp` | `0.9` | Temperature dividing the logits before filtering/sampling. |
| `--topp_temp` | `0.9` | Nucleus (top-p) cutoff used by `top_p_logits()`. |
| `--shift` | `True` | Leave this on for these checkpoints &#8212; see "Shift alignment" above. |
| `--flash_attn` | `eager` | `eager`, `sdpa`, or `flash_attention_2` &#8212; which attention kernel backs `denoise_model`. |
| `--verbose` | `False` | Prints the partially-decoded `xt` at every step, so you can literally watch Diagram 2 happen. |

---

## 9. FAQ

**Why does the sampler call the model `T` times instead of once?**
Because this is diffusion, not autoregression: the model never commits to an output in
a single pass. Each of the `T` calls looks at the *whole* sequence as it currently
stands and refines the guess everywhere at once — that's what makes bidirectional
attention (Diagram 3) necessary in the first place.

**Why the "shift"?**
`DiscreteDiffusionModel` reuses the *unmodified* LLaMA `lm_head`, which was trained
as a next-token predictor: the logits produced at position `i` are trained to predict
the token at position `i+1`, not the token at `i`. `generate_samples()` corrects for
this by shifting the sampled guesses one slot to the right (`model.py:118-121`) before
using them. This is a property of reusing an AR checkpoint's head as-is, not something
diffusion models need in general.

**Is `LinearNoise` (`model.py:160-173`) used anywhere in this document's flow?**
No. It's defined in `model.py` to describe the abstract noise schedule (α_t) used
conceptually for absorbing-state diffusion and during training. The actual inference
loop above doesn't call it — it hardcodes its own reveal-probability schedule directly
as `p = 1/(t+1)` (`model.py:131`).

**Does training follow this same loop?**
No — training optimizes the denoiser's weights against a diffusion loss over the
adaptation corpus; this document only covers using an *already-trained* checkpoint to
generate text. Training code lives in `DiffuLLaMA-training/` (custom LLaMA-2 adaptation)
and `LLaMA-Factory/src/llamafactory/train/ddm/` (the `ddm` / `ddm-sft` stages mentioned
in the main [`README.md`](README.md)).

**What is `get_anneal_attn_mask` actually building?**
A same-shape mask as a normal causal mask, but blended with a random mask controlled by
`attn_mask_ratio` (`model.py:175-187`) — during training this ratio is annealed from
causal (`0.0`) toward fully bidirectional (`1.0`) over the course of adaptation. At
inference time it is always called with `attn_mask_ratio=1.0` (`model.py:100`), i.e.
always fully bidirectional, matching the fully-adapted checkpoint.

---

*This file is a supplement to [`README.md`](README.md) and does not replace it — see
the main README for installation, training, finetuning and evaluation instructions.*
