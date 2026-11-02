"""Per-layer, mean-pooled hidden states of ONE model on a fixed corpus. One model per process.

    python3 extract_hidden.py --model diffullama --role dlm     # the diffusion LM
    python3 extract_hidden.py --model diffullama --role base    # its autoregressive origin

Separate processes are mandatory: attention_patch (needed by DiffuLLaMA) replaces
LlamaModel/GPT2Model.forward process-wide and would make the AR base bidirectional.

Representation = output of each transformer block, captured with forward hooks and
mean-pooled over the segment's tokens. Hooks (not output_hidden_states) because the
anchor's wrapper ignores that flag, and because HF applies the final norm to the last
hidden state -- hooks give block i of the DLM and block i of the base the same treatment.

Inputs are identical token ids for both roles: the dlm run writes them, the base run
reads them back and asserts the tokenizer reproduces them.
  * DLM input: the clean segment, bidirectional attention (the fully-unmasked final state).
  * base input: the clean segment, its native causal attention.
"""
from __future__ import annotations

import argparse
import sys

from common import ROOT, WORK, cfg

N_SEG_DEFAULT = None


def corpus_segments(tok, n, seg_len):
    """WikiText-103 validation (needs the `datasets` package in this venv)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    buf, segs = [], []
    for row in ds:
        t = row["text"].strip()
        if not t or t.startswith("="):
            continue
        buf.extend(tok(" " + t, add_special_tokens=False).input_ids)
        while len(buf) >= seg_len and len(segs) < n:
            segs.append(buf[:seg_len]); buf = buf[seg_len:]
        if len(segs) >= n:
            break
    assert len(segs) == n, f"corpus too small: {len(segs)} < {n}"
    return segs


def find_blocks(model):
    """The ModuleList of transformer blocks, found structurally (works across model families)."""
    import torch.nn as nn
    best = None
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 4 and any(k in type(m[0]).__name__ for k in ("Layer", "Block")):
            if best is None or len(m) > len(best[1]):
                best = (name, m)
    assert best, "could not locate transformer blocks"
    return best


def check_attention(run, ids, bidirectional):
    """Abort unless the model attends the way its role requires: a DLM must see future
    tokens, an AR base must not. Changing the last token must (DLM) or must not (base)
    change the outputs at earlier positions."""
    import torch
    x = torch.tensor([ids], device="cuda"); alt = x.clone(); alt[0, -1] = (alt[0, -1] + 7) % 30000
    with torch.no_grad():
        d = (run(x)[0][0, :-1].float() - run(alt)[0][0, :-1].float()).abs().max().item()
    seen = d > 1e-2
    assert seen == bidirectional, (f"expected {'bidirectional' if bidirectional else 'causal'} attention, "
                                   f"but earlier outputs changed by {d:.3g} when the last token changed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--role", choices=["dlm", "base"], required=True)
    a = ap.parse_args()
    c = cfg(); spec = c["models"][a.model]; k = c["cka"]
    import torch
    out_dir = WORK / "hidden" / a.model; out_dir.mkdir(parents=True, exist_ok=True)
    hf = spec["hf"] if a.role == "dlm" else spec["base"]
    assert hf, f"{a.model} has no AR base (trained from scratch) -- nothing to extract"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    ids_path = out_dir / "segments.pt"
    if a.role == "dlm":
        segs = corpus_segments(tok, k["n_segments"], k["seg_tokens"]); torch.save(segs, ids_path)
    else:
        segs = torch.load(ids_path)
        check = tok.convert_ids_to_tokens(segs[0][:20])
        assert None not in check, "base tokenizer cannot represent the DLM's token ids"

    if a.role == "dlm" and spec["backend"] in ("diffugpt", "diffullama"):
        sys.path.insert(0, str(ROOT))
        if spec["backend"] == "diffullama":
            from attention_patch import replace_attention_mask
            replace_attention_mask()
        from transformers import AutoConfig, LlamaForCausalLM
        from model import DiscreteDiffusionModel, get_anneal_attn_mask
        conf = AutoConfig.from_pretrained(hf)
        if spec["backend"] == "diffullama":
            base = LlamaForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, _attn_implementation="eager")
            m = DiscreteDiffusionModel(model=base, config=conf, tokenizer=tok, device="cuda").to("cuda").eval()
        else:
            m = DiscreteDiffusionModel.from_pretrained(hf, model=spec["base"].split("/")[-1], config=conf,
                                                       tokenizer=tok, device="cuda").to("cuda").eval()
        def run(x):
            # The all-zero 4D mask is what generate_samples passes. It is needed for DiffuGPT too:
            # under SDPA the patched GPT-2 forward turns attention_mask=None into is_causal=True,
            # and the bias.fill_(True) in model.py only affects eager attention (checked 2026-11-02).
            att = get_anneal_attn_mask(x.shape[1], x.shape[0], torch.bfloat16, x.device, 1.0)
            return m.denoise_model(inputs_embeds=m.get_embeds(x), attention_mask=att, return_dict=False)
        net = m.denoise_model
    else:
        from transformers import AutoModel
        net = AutoModel.from_pretrained(hf, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        def run(x):
            return net(input_ids=x)

    check_attention(run, segs[0][:16], bidirectional=(a.role == "dlm"))
    name, blocks = find_blocks(net)
    pooled = [[] for _ in blocks]
    hooks = [b.register_forward_hook(lambda mod, inp, out, i=i: pooled[i].append(
        (out[0] if isinstance(out, tuple) else out).float().mean(1).cpu())) for i, b in enumerate(blocks)]
    with torch.no_grad():
        for s in range(0, len(segs), 8):
            run(torch.tensor(segs[s:s + 8], device="cuda"))
    for h in hooks:
        h.remove()
    layers = torch.stack([torch.cat(p) for p in pooled]).half()            # L x n x d
    torch.save({"hf": hf, "blocks": name, "layers": layers}, out_dir / f"{a.role}.pt")
    print(f"{a.model}/{a.role}: {hf}  {layers.shape[0]} blocks x {layers.shape[1]} segments x {layers.shape[2]} dims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
