"""
Off-by-one / shift invariant test for structure-guided remasking.

This is the test the design doc (Sec 5.3, Risk 2) says must be written BEFORE the selector.
It exists because the failure mode is silent: a mapping that ignores DiffuLLaMA's inherited
autoregressive shift masks the WRONG tokens, runs end to end without error, and quietly
underperforms.

Runs on pure stdlib -- no torch, no transformers, no GPU, no model download. The shift is
integer index arithmetic, so it can be verified exactly without any of that.

WHAT THIS COVERS
  1. Shift semantics of generate_samples() replicated faithfully from model.py -> proves the
     exact canvas/returned index relation.
  2. The concrete off-by-one bug, demonstrated: naive mapping masks the wrong token.
  3. src_mask freeze semantics (frozen positions are never altered).
  4. AST source-span -> character-offset mapping (stdlib `ast`, exact).
  5. Full AST -> char -> token -> canvas chain against a STUB tokenizer.

WHAT THIS DOES NOT COVER -- still needs the real tokenizer, flagged loudly at the end:
  * SentencePiece/BPE quirks: the LLaMA tokenizer glues leading whitespace into tokens and
    uses U+2581 prefix markers. Python indentation is SEMANTIC. Step 5 uses a stub tokenizer
    with clean offsets, so it validates the LOGIC but not the real tokenizer's behaviour.
    Re-run step 5 with `AutoTokenizer.from_pretrained('diffusionfamily/diffullama')` and
    return_offsets_mapping=True before trusting any of this on real code.

Usage:  python audit/test_shift_offset.py
"""
import ast
import sys

MASK = "[M]"          # stand-in for tokenizer.mask_token_id
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Faithful replication of model.py's shift behaviour.
#
# model.py, with diff_args.shift=True:
#   L110  logits = model(xt)                        logits[i] predicts position i+1
#   L115  x0     = sample(logits)                   x0[i] = prediction FOR position i+1
#   L120  x0     = cat([x[:,0:1], x0[:,:-1]])       right-shift -> x0 now canvas-aligned
#   L124  x0     = xt.masked_scatter(maskable, ...) fill only the masked positions
#   L156  x0     = x0[:,1:]                         DROP canvas position 0
# ---------------------------------------------------------------------------
def generate_samples_shift_sim(canvas, src_mask, oracle):
    """Pure-python simulation of generate_samples(..., shift=True).

    `oracle(i)` stands in for the model: it returns the token the model predicts for
    canvas position i+1 (i.e. raw pre-shift logits semantics, exactly as model.py sees them).
    Returns the sequence generate_samples() would return.
    """
    L = len(canvas)
    maskable = [not m for m in src_mask]

    # L105: xt = x.masked_fill(maskable_mask, mask_token_id)
    xt = [MASK if maskable[i] else canvas[i] for i in range(L)]

    # L115: raw model output, pre-shift. x0_raw[i] is the prediction for position i+1.
    x0_raw = [oracle(i) for i in range(L)]

    # L120: right-shift. x0[0] = x[0]; x0[i] = x0_raw[i-1] = prediction FOR position i.
    x0 = [canvas[0]] + x0_raw[:-1]

    # L124: only masked positions get the model's prediction; frozen keep xt.
    x0 = [x0[i] if maskable[i] else xt[i] for i in range(L)]

    # L156: drop canvas position 0.
    return x0[1:]


def test_1_shift_relation():
    print("\n[1] Shift semantics: what canvas position does each returned index hold?")
    canvas = [f"c{i}" for i in range(8)]
    src_mask = [0] * 8                     # everything maskable
    # oracle: prediction for position i+1 is the sentinel "PRED@<i+1>"
    out = generate_samples_shift_sim(canvas, src_mask, lambda i: f"PRED@{i+1}")

    check("returned length is L-1",
          len(out) == len(canvas) - 1,
          f"got {len(out)}, expected {len(canvas) - 1}")

    # out[k] should be the prediction for canvas position k+1
    ok = all(out[k] == f"PRED@{k+1}" for k in range(len(out)))
    check("returned[k] holds the prediction for canvas position k+1",
          ok,
          f"out={out}")

    print("\n  => INVARIANT:  canvas_position = returned_index + 1")
    print("     equivalently: returned_index = canvas_position - 1")


def test_2_the_offbyone_bug():
    print("\n[2] The bug, demonstrated: naive mapping masks the WRONG token.")
    # Canvas: [bos, a, b, c, d, e]. Say we decoded the returned sequence, found the token
    # we want to repair at RETURNED index 2, and naively wrote src_mask[2] = maskable.
    canvas = ["bos", "a", "b", "c", "d", "e"]
    target_returned_index = 2

    # What token is actually at that returned index? (from test 1: canvas[k+1])
    correct_canvas_pos = target_returned_index + 1
    naive_canvas_pos = target_returned_index          # the bug: forgetting the shift

    check("naive mapping picks a different canvas position than the correct one",
          naive_canvas_pos != correct_canvas_pos,
          "if these matched, there would be no bug to catch")

    print(f"        token the user saw at returned[{target_returned_index}] "
          f"= canvas[{correct_canvas_pos}] = {canvas[correct_canvas_pos]!r}")
    print(f"        naive src_mask[{naive_canvas_pos}] would instead mask "
          f"{canvas[naive_canvas_pos]!r}  <-- off by one, silently wrong")

    # Prove it end-to-end: mask the naive position, see which token actually changes.
    src_mask = [1] * len(canvas)
    src_mask[naive_canvas_pos] = 0                    # naive: mark this one maskable
    out = generate_samples_shift_sim(canvas, src_mask, lambda i: "REGEN")

    changed = [k for k in range(len(out)) if out[k] != canvas[k + 1]]
    check("naive masking regenerates a token the user did not intend",
          changed == [naive_canvas_pos - 1] and naive_canvas_pos - 1 != target_returned_index,
          f"changed returned indices = {changed}, user wanted {target_returned_index}")

    # And the correct version hits the intended token.
    src_mask2 = [1] * len(canvas)
    src_mask2[correct_canvas_pos] = 0
    out2 = generate_samples_shift_sim(canvas, src_mask2, lambda i: "REGEN")
    changed2 = [k for k in range(len(out2)) if out2[k] != canvas[k + 1]]
    check("correct (+1) mapping regenerates exactly the intended token",
          changed2 == [target_returned_index],
          f"changed returned indices = {changed2}, wanted [{target_returned_index}]")


def test_3_freeze_semantics():
    print("\n[3] src_mask freeze: frozen positions must never change.")
    # Mirrors inf_diffullama.py:65-66 -- prefix frozen, rest generated.
    prefix = ["bos", "Today", "is", "a"]
    gen_len = 9
    canvas = prefix + ["pad"] * (gen_len - len(prefix))
    src_mask = [1] * len(prefix) + [0] * (gen_len - len(prefix))

    out = generate_samples_shift_sim(canvas, src_mask, lambda i: f"GEN@{i+1}")

    # canvas[0] is dropped, so the prefix occupies returned[0 .. len(prefix)-2]
    prefix_in_returned = out[: len(prefix) - 1]
    check("frozen prefix survives unchanged in the returned sequence",
          prefix_in_returned == prefix[1:],
          f"got {prefix_in_returned}, expected {prefix[1:]}")

    # Generated region: canvas positions len(prefix).. -> returned indices len(prefix)-1..
    gen_region = out[len(prefix) - 1:]
    check("every generated position was actually regenerated",
          all(t.startswith("GEN@") for t in gen_region),
          f"got {gen_region}")

    print(f"        prefix occupies canvas[0..{len(prefix)-1}] "
          f"-> returned[0..{len(prefix)-2}]")
    print(f"        generation starts at canvas[{len(prefix)}] "
          f"-> returned[{len(prefix)-1}]")


# ---------------------------------------------------------------------------
# AST -> char span -> token -> canvas
# ---------------------------------------------------------------------------
def line_start_offsets(src):
    """Prefix sums of line start positions, for (lineno, col_offset) -> char index."""
    offsets, total = [0], 0
    for line in src.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def node_char_span(node, starts):
    """AST node -> (start_char, end_char). ast linenos are 1-based, cols are 0-based bytes."""
    if not hasattr(node, "lineno") or node.lineno is None:
        return None
    if getattr(node, "end_lineno", None) is None:
        return None
    return (starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset)


SRC = '''def price(qty):
    discount = 0.10
    subtotal = qty * 5
    final = subtotal - discount * qty
    return final
'''


def test_4_ast_char_spans():
    print("\n[4] AST -> character span mapping (stdlib ast, exact).")
    starts = line_start_offsets(SRC)
    tree = ast.parse(SRC)

    n_checked, n_ok = 0, 0
    for node in ast.walk(tree):
        span = node_char_span(node, starts)
        if span is None:
            continue
        s, e = span
        recovered = SRC[s:e]
        try:
            expected = ast.get_source_segment(SRC, node)
        except Exception:
            continue
        if expected is None:
            continue
        n_checked += 1
        if recovered == expected:
            n_ok += 1
        elif n_ok == n_checked - 1:      # report only the first mismatch
            print(f"        MISMATCH on {type(node).__name__}: "
                  f"recovered={recovered!r} expected={expected!r}")

    check(f"all {n_checked} AST node spans recover their exact source text",
          n_checked > 0 and n_ok == n_checked,
          f"{n_ok}/{n_checked} matched")

    # The doc's motivating example: the `discount` definition is far from the failure site.
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    disc = [n for n in assigns
            if any(getattr(t, "id", None) == "discount" for t in n.targets)]
    check("can locate the 'discount' definition statement (the doc's example)",
          len(disc) == 1,
          f"found {len(disc)}")
    if disc:
        s, e = node_char_span(disc[0], starts)
        print(f"        discount def at chars [{s},{e}) = {SRC[s:e]!r}")


class StubTokenizer:
    """Whitespace/punctuation tokenizer exposing offset_mapping, like a HF fast tokenizer.

    Deliberately clean: every token's offsets are exact and nothing is glued to whitespace.
    The REAL LLaMA tokenizer does not behave this way -- see the warning at the end.
    """

    def encode_with_offsets(self, text):
        toks, offs, i = [], [], 0
        while i < len(text):
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c.isalnum() or c == "_":
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == "_" or text[j] == "."):
                    j += 1
            else:
                j = i + 1
            toks.append(text[i:j])
            offs.append((i, j))
            i = j
        return toks, offs


def test_5_full_chain_stub():
    print("\n[5] Full chain AST -> char -> token -> canvas (STUB tokenizer).")
    tok = StubTokenizer()
    toks, offs = tok.encode_with_offsets(SRC)
    starts = line_start_offsets(SRC)
    tree = ast.parse(SRC)

    disc = [n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "discount" for t in n.targets)][0]
    s, e = node_char_span(disc, starts)

    # char span -> token indices (any token overlapping the span)
    tok_idx = [i for i, (a, b) in enumerate(offs) if a < e and b > s]
    check("AST span maps to a non-empty, contiguous token index set",
          len(tok_idx) > 0 and tok_idx == list(range(tok_idx[0], tok_idx[-1] + 1)),
          f"tok_idx={tok_idx}")

    recovered = " ".join(toks[i] for i in tok_idx)
    check("decoded tokens cover the AST node's source text",
          all(w in recovered for w in ["discount", "0.10"]),
          f"recovered={recovered!r}")
    print(f"        node source  = {SRC[s:e]!r}")
    print(f"        token idx    = {tok_idx} -> {[toks[i] for i in tok_idx]}")

    # token index (in the RETURNED/decoded sequence) -> canvas position, per test 1
    PROMPT_LEN = 5                       # arbitrary non-zero prompt, to exercise the offset
    canvas_pos = [PROMPT_LEN + i + 1 for i in tok_idx]     # +1 is the shift
    naive_pos = [PROMPT_LEN + i for i in tok_idx]          # the bug
    check("canvas positions differ from the naive (no-shift) mapping",
          canvas_pos != naive_pos)
    print(f"        canvas pos   = {canvas_pos}   (correct, includes +1 shift)")
    print(f"        naive pos    = {naive_pos}   (WRONG -- masks one token early)")


def main():
    print("=" * 78)
    print("Shift / offset-mapping invariant test  --  design doc Sec 5.3, Risk 2")
    print("=" * 78)

    test_1_shift_relation()
    test_2_the_offbyone_bug()
    test_3_freeze_semantics()
    test_4_ast_char_spans()
    test_5_full_chain_stub()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    else:
        print("All checks passed.")
    print("=" * 78)
    print("""
STILL UNVERIFIED -- do not skip this before building the selector:

  Step 5 uses a STUB tokenizer with clean, whitespace-free offsets. The real LLaMA
  tokenizer is SentencePiece-based: it glues leading whitespace into tokens and marks
  word starts with U+2581. Python indentation is SEMANTIC, so an indented statement's
  first token may carry its own indentation, and the char->token mapping can behave
  differently from the stub in exactly the cases that matter.

  Re-run step 5 with the real thing before trusting any of this:

      from transformers import AutoTokenizer
      tk = AutoTokenizer.from_pretrained('diffusionfamily/diffullama', use_fast=True)
      enc = tk(SRC, return_offsets_mapping=True, add_special_tokens=False)
      # then assert, for EVERY AST node in a corpus of programs, that decoding the
      # mapped token indices recovers text covering that node's source segment.

  The canvas relation proven in test 1 (canvas_position = returned_index + 1) is exact
  and tokenizer-independent -- it is pure index arithmetic from model.py and does not
  need re-testing with the real tokenizer.
""")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
