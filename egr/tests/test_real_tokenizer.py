"""
Q-6: does the AST -> char -> token -> canvas chain survive the REAL LLaMA tokenizer?

`test_shift_offset.py` proved the shift invariant (tokenizer-independent) and validated the mapping
logic against a STUB tokenizer with clean offsets. This file runs the same chain against the actual
`diffusionfamily/diffullama` tokenizer, which is SentencePiece-based.

Needs `transformers` only -- no torch, no GPU, no model weights (tokenizer files are ~2 MB).

    pip install transformers
    PYTHONIOENCODING=utf-8 python audit/test_real_tokenizer.py

WHY THIS MATTERS
Python indentation is semantic. If a token that covers an AST node's first character ALSO covers
part of the preceding indentation, then remasking that node necessarily destroys indentation the
model must then regenerate correctly. That is a silent correctness hazard on top of the shift.
"""
import ast
import sys

MODEL = "diffusionfamily/diffullama"

CORPUS = [
    # flat body
    '''def price(qty):
    discount = 0.10
    subtotal = qty * 5
    final = subtotal - discount * qty
    return final
''',
    # one level of nesting
    '''def count_evens(nums):
    total = 0
    for n in nums:
        if n % 2 == 0:
            total += 1
    return total
''',
    # deeper nesting + multi-line statement
    '''def classify(items):
    out = []
    for it in items:
        if it > 0:
            out.append(
                "positive"
            )
        else:
            out.append("other")
    return out
''',
    # HumanEval-ish
    '''def has_close_elements(numbers, threshold):
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
''',
]


def line_start_offsets(src):
    offsets, total = [0], 0
    for line in src.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def node_char_span(node, starts):
    if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
        return None
    return (starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset)


def main():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("FATAL: pip install transformers")
        return 2

    tk = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if not tk.is_fast:
        print("FATAL: need a fast tokenizer for offset_mapping")
        return 2
    print(f"tokenizer: {type(tk).__name__}  fast={tk.is_fast}  vocab={tk.vocab_size}\n")

    total_nodes = 0
    covered = 0
    left_bleed_nodes = 0
    bleed_is_whitespace = 0
    bleed_is_code = 0
    examples = []

    for src_i, src in enumerate(CORPUS):
        starts = line_start_offsets(src)
        enc = tk(src, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        tree = ast.parse(src)

        for node in ast.walk(tree):
            # statements only -- these are what a structural selector would remask
            if not isinstance(node, ast.stmt):
                continue
            span = node_char_span(node, starts)
            if span is None:
                continue
            s, e = span
            seg = ast.get_source_segment(src, node)
            if not seg:
                continue

            tok_idx = [i for i, (a, b) in enumerate(offs) if a < e and b > s]
            if not tok_idx:
                continue

            total_nodes += 1

            # does the mapped token span cover the node's source text?
            cov_start, cov_end = offs[tok_idx[0]][0], offs[tok_idx[-1]][1]
            if cov_start <= s and cov_end >= e:
                covered += 1

            # LEFT BLEED: the first mapped token starts before the node does
            if cov_start < s:
                left_bleed_nodes += 1
                bled = src[cov_start:s]
                if bled.strip() == "":
                    bleed_is_whitespace += 1
                else:
                    bleed_is_code += 1
                if len(examples) < 6:
                    examples.append((src_i, type(node).__name__,
                                     seg.splitlines()[0][:44], bled, cov_start, s))

    print(f"Statement nodes examined: {total_nodes}")
    print(f"  mapped tokens cover the node's source : {covered}/{total_nodes}")
    print(f"  first token starts BEFORE the node    : {left_bleed_nodes}/{total_nodes} "
          f"({100*left_bleed_nodes/max(total_nodes,1):.0f}%)")
    print(f"      of which the bled text is whitespace : {bleed_is_whitespace}")
    print(f"      of which the bled text is real code  : {bleed_is_code}")

    print("\nExamples of left-bleed (masking this node also masks the bled text):")
    for src_i, kind, first_line, bled, cov_start, s in examples:
        print(f"  [corpus {src_i}] {kind:<10} {first_line!r}")
        print(f"      node starts at char {s}, first token starts at {cov_start} "
              f"-> bleeds {bled!r}")

    print("\n" + "=" * 78)
    if left_bleed_nodes == 0:
        print("RESULT: no left bleed. The stub-tokenizer logic transfers cleanly.")
        verdict = 0
    elif bleed_is_code > 0:
        print("RESULT: left bleed includes REAL CODE. Remasking a node would destroy")
        print("        neighbouring code. The mapping needs a token-boundary snap rule.")
        verdict = 1
    else:
        print("RESULT: left bleed is whitespace-only -- and in Python that whitespace is")
        print("        INDENTATION, which is semantic. Remasking a statement necessarily")
        print("        remasks part of its own indentation, so the model must regenerate")
        print("        the exact indent or the program breaks.")
        print("")
        print("        Not fatal, but it must be handled explicitly. Options:")
        print("          (a) snap the mask to line boundaries and regenerate whole lines")
        print("          (b) exclude the leading-whitespace token from the mask set and")
        print("              accept that the first sub-token of the statement is frozen")
        print("          (c) re-tokenize with the indent stripped, then re-attach")
        print("")
        print("        Whichever is chosen must be stated in the design doc, because the")
        print("        naive 'mask every token overlapping the AST span' rule silently")
        print("        does (a) WITHOUT saying so.")
        verdict = 1
    print("=" * 78)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
