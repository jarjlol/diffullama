"""
Regenerate the SVG diagrams used by the EGR design documents.

    python egr/docs/assets/build_diagrams.py

Stdlib only -- no torch, no GPU, no third-party packages. The palette matches
`assets/simplified/*.svg` so the two document sets look like one repository:
GitHub light-mode surfaces, dark ink, muted grey for secondary text.

Every diagram is written as a real `.svg` file and committed. Documents reference
them with plain Markdown image syntax, so they render on GitHub, in editors, and
in anything that exports the docs elsewhere.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

BG = "#f6f8fa"
CARD = "#ffffff"
LINE = "#d0d7de"
INK = "#1f2328"
MUTED = "#57606a"
ACCENT = "#0969da"
GOOD = "#1a7f37"
WARN = "#9a6700"
BAD = "#cf222e"

SANS = "Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=13, fill=INK, weight="400", family=SANS, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>')


def box(x, y, w, h, stroke=LINE, fill=CARD, width=1.5, rx=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def path(d, stroke=MUTED, width=1.8, marker=True, dash=None):
    m = ' marker-end="url(#arrow)"' if marker else ""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{m}{ds}/>'


def svg(w, h, title, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img">\n'
        f'<title>{esc(title)}</title>\n'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>\n'
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


def header(title, subtitle, x=40):
    return [text(x, 34, title, size=15, weight="700"),
            text(x, 54, subtitle, size=12.5, fill=MUTED)]


def stage(x, y, w, h, num, name, detail, code=None, accent=LINE):
    """A numbered card: bold title line, muted detail line, optional mono line."""
    out = [box(x, y, w, h, stroke=accent),
           text(x + 18, y + 26, f"{num}   {name}", size=13.5, weight="700"),
           text(x + 18, y + 45, detail, size=11.5, fill=MUTED)]
    if code:
        out.append(text(x + 18, y + 62, code, size=11, fill=ACCENT, family=MONO))
    return out


# ----------------------------------------------------------------------------
# 01 -- the recursive repair loop
# ----------------------------------------------------------------------------
def diagram_loop():
    W, H = 940, 600
    b = header("The recursive repair loop",
               "One iteration. Left column establishes failure; right column converts it into a targeted edit.")

    b += [text(60, 92, "VERIFY  --  is it wrong, and how?", size=11.5, fill=MUTED, weight="700"),
          text(520, 92, "REPAIR  --  where, and regenerate", size=11.5, fill=MUTED, weight="700")]

    LW, RW, BH = 360, 360, 74
    ys = [110, 204, 298, 392]

    # left column, top to bottom
    b += stage(60, ys[0], LW, BH, "1", "Seed candidate",
               "Benchmark prompt, or a given buggy program.", "program: str")
    b += stage(60, ys[1], LW, BH, "2", "Syntax gate",
               "Parse before running anything.", "ast.parse(program)")
    b += stage(60, ys[2], LW, BH, "3", "Sandboxed, traced test run",
               "Subprocess, timeout, no network; line-event tracer.", "verdict, spectra, trace")
    b += stage(60, ys[3], LW, BH, "4", "Verdict",
               "PASS / FAIL / ERROR / TIMEOUT / HARNESS_ERROR", "typed, never a bare bool")

    # right column, bottom to top (flow returns upward)
    b += stage(520, ys[3], RW, BH, "5", "Evidence",
               "Syntax | Assertion | Exception | Timeout.", "one dataclass per failure kind")
    b += stage(520, ys[2], RW, BH, "6", "Localize",
               "Ochiai over spectra + trace -> witness line -> AST node.", "witness: ast.stmt")
    b += stage(520, ys[1], RW, BH, "7", "Remask",
               "AST neighbourhood -> lines -> chars -> tokens -> canvas.", "src_mask[i] = 0")
    b += stage(520, ys[0], RW, BH, "8", "Infill",
               "Frozen context, masked span only.", "generate_samples(ids, src_mask)")

    # vertical arrows, left column
    for a, c in zip(ys, ys[1:]):
        b.append(path(f"M 240 {a + BH} L 240 {c - 6}"))
    # vertical arrows, right column (upward)
    for a, c in zip(ys, ys[1:]):
        b.append(path(f"M 700 {c} L 700 {a + BH + 6}"))

    # 4 -> 5 crossover
    b.append(path(f"M {60 + LW} {ys[3] + 37} L 514 {ys[3] + 37}", stroke=BAD))
    b.append(text(444, ys[3] + 30, "FAIL", size=11.5, fill=BAD, weight="700", anchor="middle"))

    # 8 -> 2 loop back
    b.append(path(f"M 520 {ys[0] + 37} L 476 {ys[0] + 37} L 476 {ys[1] + 37} L 426 {ys[1] + 37}",
                  stroke=ACCENT))
    b.append(text(473, ys[0] + 22, "depth += 1", size=11, fill=ACCENT, weight="700", anchor="middle"))

    # PASS exit
    b.append(path(f"M 240 {ys[3] + BH} L 240 505", stroke=GOOD))
    b.append(box(60, 505, 360, 52, stroke=GOOD))
    b.append(text(78, 528, "PASS  ->  return repaired program", size=13, fill=GOOD, weight="700"))
    b.append(text(78, 546, "Record depth reached and every intermediate attempt.", size=11, fill=MUTED))

    # stop conditions
    b.append(box(520, 480, 360, 92, stroke=WARN, dash="5 4"))
    b.append(text(538, 502, "Stop without a pass when:", size=12.5, fill=WARN, weight="700"))
    b.append(text(538, 521, "depth == 5  (the recursion cap)", size=11.5, fill=MUTED, family=MONO))
    b.append(text(538, 538, "evidence signature repeats and scope cannot widen", size=11.5, fill=MUTED, family=MONO))
    b.append(text(538, 555, "no token span maps to the witness node", size=11.5, fill=MUTED, family=MONO))

    return svg(W, H, "The recursive repair loop", b)


# ----------------------------------------------------------------------------
# 02 -- the localization chain and its invariants
# ----------------------------------------------------------------------------
def diagram_chain():
    W, H = 940, 712
    b = header("From a failing test to a set of masked token positions",
               "Seven links. Each one is separately testable, and two of them have known silent-failure modes.")

    b += [text(60, 92, "LINK", size=11.5, fill=MUTED, weight="700"),
          text(560, 92, "WHAT MUST HOLD", size=11.5, fill=MUTED, weight="700")]

    rows = [
        ("Failing test + trace", "spectra: {line -> (n_pass, n_fail)}",
         "A crash in the harness is never recorded as a test failure.", LINE),
        ("Ochiai score per line", "score = f / sqrt((f + p) * total_failing)",
         "Ties broken deterministically, by line number.", LINE),
        ("Witness line", "line: int  (1-based, like ast)",
         "Line is inside the function under repair, not the test.", LINE),
        ("Witness AST node", "node: ast.stmt covering that line",
         "Smallest statement node containing the line.", LINE),
        ("Neighbourhood", "Leaf | Parent+Leaf | Enclosing function",
         "Scope is an explicit parameter, never a constant in the code.", ACCENT),
        ("Character span", "(start, end) offsets into the source",
         "Snapped OUT to whole lines -- see the indentation hazard.", WARN),
        ("Token span -> canvas", "canvas_index = returned_index + 1",
         "The shift invariant. Off by one masks the wrong token, silently.", BAD),
    ]

    y = 110
    for i, (name, code, rule, accent) in enumerate(rows, start=1):
        b.append(box(60, y, 430, 62, stroke=accent))
        b.append(text(78, y + 25, f"{i}   {name}", size=13, weight="700"))
        b.append(text(78, y + 45, code, size=11, fill=ACCENT, family=MONO))
        b.append(box(560, y, 340, 62, stroke=LINE))
        # wrap the rule onto two lines if long
        words, lines, cur = rule.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 44:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        for j, ln in enumerate(lines[:2]):
            b.append(text(578, y + 27 + j * 17, ln, size=11.5, fill=MUTED))
        b.append(path(f"M 490 {y + 31} L 554 {y + 31}"))
        if i < len(rows):
            b.append(path(f"M 275 {y + 62} L 275 {y + 74}"))
        y += 76

    b.append(box(60, y + 4, 840, 44, stroke=GOOD))
    b.append(text(78, y + 31,
                  "Output: a src_mask where 0 marks every position the model may rewrite, and 1 freezes everything else.",
                  size=12.5, fill=GOOD, weight="700"))
    return svg(W, H, "Localization chain from failing test to masked tokens", b)


# ----------------------------------------------------------------------------
# 03 -- module architecture
# ----------------------------------------------------------------------------
def diagram_modules():
    W, H = 940, 620
    b = header("Module architecture",
               "Five protocols, one loop. Everything that varies between experiments is a swappable implementation.")

    # the loop, centre
    b.append(box(330, 268, 280, 78, stroke=ACCENT, width=2.2))
    b.append(text(470, 296, "egr/loop.py", size=14, weight="700", anchor="middle", family=MONO))
    b.append(text(470, 316, "repair_loop(task, cfg) -> RunRecord", size=11, fill=MUTED,
                  anchor="middle", family=MONO))
    b.append(text(470, 334, "The only stateful code. ~120 lines.", size=11, fill=MUTED, anchor="middle"))

    # satellites: (x, y, w, h, module, protocol, impls)
    sats = [
        (60, 100, 300, 96, "egr/benchmarks/", "Benchmark -> Task",
         "humaneval_plus | mbpp_plus | humanevalfix | mutants"),
        (580, 100, 300, 96, "egr/backend.py", "DiffusionBackend",
         "DiffuLLaMA | Dream-Coder | Mock (CPU, no GPU)"),
        (60, 240, 240, 134, "egr/verify.py", "Verifier -> Verdict",
         "syntax gate, sandbox,\nline tracer, spectra"),
        (640, 240, 240, 134, "egr/policy.py", "RemaskPolicy",
         "ours | static | confidence\n| random | resample"),
        (60, 418, 300, 96, "egr/localize.py", "Evidence -> witness node",
         "Ochiai + traceback + AST neighbourhood"),
        (580, 418, 300, 96, "egr/canvas.py", "Canvas",
         "the single shift-aware span -> index mapping"),
    ]
    for x, y, w, h, mod, proto, impls in sats:
        b.append(box(x, y, w, h))
        b.append(text(x + 18, y + 26, mod, size=13, weight="700", family=MONO))
        b.append(text(x + 18, y + 46, proto, size=11.5, fill=ACCENT, weight="700"))
        for j, ln in enumerate(impls.split("\n")):
            b.append(text(x + 18, y + 68 + j * 16, ln, size=11, fill=MUTED))

    # arrows into the loop
    b.append(path("M 300 148 L 470 148 L 470 262", dash="4 4"))
    b.append(path("M 640 148 L 470 148", dash="4 4", marker=False))
    b.append(path("M 300 300 L 324 300"))
    b.append(path("M 640 300 L 616 300"))
    b.append(path("M 360 466 L 470 466 L 470 352", dash="4 4"))
    b.append(path("M 580 466 L 470 466", dash="4 4", marker=False))

    b.append(box(60, 542, 820, 52, stroke=GOOD, dash="5 4"))
    b.append(text(78, 566, "Adding an experiment arm never touches the loop:",
                  size=12.5, fill=GOOD, weight="700"))
    b.append(text(78, 584,
                  "new policy = one class + one registry entry   |   new benchmark = one adapter returning Task   |   new backbone = one DiffusionBackend",
                  size=11, fill=MUTED))
    return svg(W, H, "EGR module architecture", b)


def main():
    for name, fn in [("01-loop.svg", diagram_loop),
                     ("02-chain.svg", diagram_chain),
                     ("03-modules.svg", diagram_modules)]:
        (OUT / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
