"""
Minimal hand-written Markdown -> PDF converter with real clickable link
annotations, built entirely on the Python standard library (no reportlab/
pandoc/weasyprint/LaTeX available in this environment -- see REPORT.md).

Supports exactly what generated_survey.md needs: H1/H2/H3 headings,
italic preamble paragraphs, **bold** spans, plain paragraphs, and a
References section of the form "[N] text [label](url)" -- the [label](url)
becomes a real PDF link annotation, not just blue text.
"""
import re
import sys

PAGE_W, PAGE_H = 612, 792  # US Letter, points
MARGIN_L, MARGIN_R, MARGIN_TOP, MARGIN_BOTTOM = 60, 60, 60, 60
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R

# Real Adobe Helvetica / Helvetica-Bold AFM advance widths (1/1000 em),
# standard base-14 metrics -- an approximate width table caused visible
# glyph overlap/word-gluing in an earlier version of this script, since
# these widths drive BOTH line-wrap decisions and actual glyph x-advance.
_HELV = {
    ' ':278,'!':278,'"':355,'#':556,'$':556,'%':889,'&':667,"'":191,'(':333,')':333,
    '*':389,'+':584,',':278,'-':333,'.':278,'/':278,'0':556,'1':556,'2':556,'3':556,
    '4':556,'5':556,'6':556,'7':556,'8':556,'9':556,':':278,';':278,'<':584,'=':584,
    '>':584,'?':556,'@':1015,'A':667,'B':667,'C':722,'D':722,'E':667,'F':611,'G':778,
    'H':722,'I':278,'J':500,'K':667,'L':556,'M':833,'N':722,'O':778,'P':667,'Q':778,
    'R':722,'S':667,'T':611,'U':722,'V':667,'W':944,'X':667,'Y':667,'Z':611,'[':278,
    '\\':278,']':278,'^':469,'_':556,'`':333,'a':556,'b':556,'c':500,'d':556,'e':556,
    'f':278,'g':556,'h':556,'i':222,'j':222,'k':500,'l':222,'m':833,'n':556,'o':556,
    'p':556,'q':556,'r':333,'s':500,'t':278,'u':556,'v':500,'w':722,'x':500,'y':500,
    'z':500,'{':334,'|':260,'}':334,'~':584,
    '§':556, 'é':556, 'ü':556, '—':1000, '‐':333,
}
_HELV_BOLD = {
    ' ':278,'!':333,'"':474,'#':556,'$':556,'%':889,'&':722,"'":238,'(':333,')':333,
    '*':389,'+':584,',':278,'-':333,'.':278,'/':278,'0':556,'1':556,'2':556,'3':556,
    '4':556,'5':556,'6':556,'7':556,'8':556,'9':556,':':333,';':333,'<':584,'=':584,
    '>':584,'?':611,'@':975,'A':722,'B':722,'C':722,'D':722,'E':667,'F':611,'G':778,
    'H':722,'I':278,'J':556,'K':722,'L':611,'M':833,'N':722,'O':778,'P':667,'Q':778,
    'R':722,'S':667,'T':611,'U':722,'V':667,'W':944,'X':667,'Y':667,'Z':611,'[':333,
    '\\':278,']':333,'^':584,'_':556,'`':333,'a':556,'b':611,'c':556,'d':611,'e':556,
    'f':333,'g':611,'h':611,'i':278,'j':278,'k':556,'l':278,'m':889,'n':611,'o':611,
    'p':611,'q':611,'r':389,'s':556,'t':333,'u':611,'v':556,'w':778,'x':556,'y':556,
    'z':500,'{':389,'|':280,'}':389,'~':584,
    '§':556, 'é':611, 'ü':611, '—':1000, '‐':333,
}


def char_width(c, style=False):
    """style: False=normal, True/'B'=bold, 'I'=italic (Oblique metrics == regular
    Helvetica metrics for the base-14 fonts), 'C'=Courier (fixed 600/1000 em)."""
    if style == "C":
        return 600
    table = _HELV_BOLD if style in (True, "B") else _HELV
    return table.get(c, 556)


def text_width(s, size, style=False):
    return sum(char_width(c, style) for c in s) * size / 1000.0


# WinAnsiEncoding (~cp1252) covers section-sign/accented letters/em-dash;
# the one character in this document outside it (U+2010, a "hyphen" that
# is not plain ASCII '-') is normalized to ASCII '-' before encoding.
_CP1252_FALLBACK = {'‐': '-'}
def to_pdf_bytes(s):
    s = ''.join(_CP1252_FALLBACK.get(c, c) for c in s)
    return s.encode('cp1252', errors='replace')


def wrap(s, size, max_w, bold=False):
    words = s.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_width(trial, size, bold) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def pdf_escape(s):
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class Page:
    def __init__(self):
        self.ops = []  # list of ('text', x, y, size, bold, italic, code, text, color)
        self.links = []  # list of (x1, y1, x2, y2, url)


class PDFDoc:
    def __init__(self):
        self.pages = [Page()]
        self.y = PAGE_H - MARGIN_TOP
        self.x = MARGIN_L

    def cur(self):
        return self.pages[-1]

    def newpage(self):
        self.pages.append(Page())
        self.y = PAGE_H - MARGIN_TOP

    def ensure_space(self, needed):
        if self.y - needed < MARGIN_BOTTOM:
            self.newpage()

    def draw_line(self, text, size, bold=False, italic=False, color=(0, 0, 0), indent=0, link=None):
        self.ensure_space(size * 1.4)
        x = MARGIN_L + indent
        self.cur().ops.append(("text", x, self.y, size, bold, italic, False, text, color))
        if link:
            w = text_width(text, size, bold)
            self.cur().links.append((x, self.y - size * 0.2, x + w, self.y + size * 0.85, link))
        self.y -= size * 1.4

    def paragraph(self, runs, size=10.5, leading=1.42, indent=0, gap_after=8, italic_all=False):
        """runs: list of (text, style) tuples forming one logical paragraph, wrapped
        together. style: False=normal, True/'B'=bold, 'I'=italic, 'C'=code (Courier)."""
        # flatten into a word stream carrying style, then wrap greedily
        words = []
        for text, style in runs:
            eff_style = "I" if (italic_all and style is False) else style
            for w in text.split(" "):
                if w:
                    words.append((w, eff_style))
        max_w = TEXT_W - indent
        lines = []
        cur, cur_w = [], 0.0
        for w, style in words:
            piece_w = text_width(w, size, style) + text_width(" ", size, style)
            if cur and cur_w + piece_w > max_w:
                lines.append(cur)
                cur, cur_w = [(w, style)], piece_w
            else:
                cur.append((w, style))
                cur_w += piece_w
        if cur:
            lines.append(cur)

        for line in lines:
            self.ensure_space(size * leading)
            x = MARGIN_L + indent
            for w, style in line:
                bold = style in (True, "B")
                italic = style == "I"
                code = style == "C"
                self.cur().ops.append(("text", x, self.y, size, bold, italic, code, w, (0, 0, 0)))
                x += text_width(w + " ", size, style)
            self.y -= size * leading
        self.y -= gap_after

    def reference_line(self, num, prefix, link_label, url, size=9.5, leading=1.35):
        """prefix text wraps normally; link_label is rendered at the end and
        made clickable, possibly wrapping onto its own continuation."""
        full = f"[{num}] {prefix} "
        words = [(w, False) for w in full.split(" ") if w]
        words.append((link_label, "LINK"))
        max_w = TEXT_W
        lines, cur, cur_w = [], [], 0.0
        for w, bold in words:
            piece_w = text_width(w, size) + text_width(" ", size)
            if cur and cur_w + piece_w > max_w:
                lines.append(cur)
                cur, cur_w = [(w, bold)], piece_w
            else:
                cur.append((w, bold))
                cur_w += piece_w
        if cur:
            lines.append(cur)

        for line in lines:
            self.ensure_space(size * leading)
            x = MARGIN_L
            for w, bold in line:
                if bold == "LINK":
                    color = (0.06, 0.30, 0.65)
                    self.cur().ops.append(("text", x, self.y, size, False, False, False, w, color))
                    lw = text_width(w, size)
                    self.cur().links.append((x, self.y - size * 0.2, x + lw, self.y + size * 0.85, url))
                    x += lw + text_width(" ", size)
                else:
                    self.cur().ops.append(("text", x, self.y, size, False, False, False, w, (0, 0, 0)))
                    x += text_width(w + " ", size)
            self.y -= size * leading
        self.y -= 6

    def hr_gap(self, pts):
        self.y -= pts

    # ---- serialization ----
    def build(self, title="Document"):
        objs = []  # list of strings, 1-indexed by position+1

        def add(s):
            objs.append(s)
            return len(objs)

        font_ids = {}
        for name in ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier"]:
            fid = add(f"<< /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding /WinAnsiEncoding >>")
            font_ids[name] = fid

        page_ids = []
        content_ids = []
        annot_id_lists = []

        pages_obj_id = None  # placeholder, will fix after we know page ids

        for page in self.pages:
            stream_parts = []
            for op in page.ops:
                _, x, y, size, bold, italic, code, text, color = op
                if code:
                    fname = "Courier"
                elif bold and italic:
                    fname = "Helvetica-BoldOblique"
                elif bold:
                    fname = "Helvetica-Bold"
                elif italic:
                    fname = "Helvetica-Oblique"
                else:
                    fname = "Helvetica"
                r, g, b = color
                stream_parts.append(
                    f"BT /F{font_ids[fname]} {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
                    f"{x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET"
                )
            stream = "\n".join(stream_parts)
            stream_bytes = to_pdf_bytes(stream)
            cid = add(f"<< /Length {len(stream_bytes)} >>\nstream\n" + stream + "\nendstream")
            content_ids.append(cid)

            annot_ids = []
            for (x1, y1, x2, y2, url) in page.links:
                aid = add(
                    "<< /Type /Annot /Subtype /Link "
                    f"/Rect [{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}] /Border [0 0 0] "
                    f"/A << /Type /Action /S /URI /URI ({pdf_escape(url)}) >> >>"
                )
                annot_ids.append(aid)
            annot_id_lists.append(annot_ids)

        # reserve page object slots (filled after Pages parent id is known)
        pages_index = add("PLACEHOLDER_PAGES")
        for i, page in enumerate(self.pages):
            font_res = " ".join(f"/F{fid} {fid} 0 R" for fid in font_ids.values())
            annots = " ".join(f"{aid} 0 R" for aid in annot_id_lists[i])
            annots_field = f" /Annots [{annots}]" if annots else ""
            pid = add(
                f"<< /Type /Page /Parent {pages_index} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << {font_res} >> >> /Contents {content_ids[i]} 0 R{annots_field} >>"
            )
            page_ids.append(pid)

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objs[pages_index - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"

        catalog_id = add(f"<< /Type /Catalog /Pages {pages_index} 0 R >>")
        info_id = add(f"<< /Title ({pdf_escape(title)}) /Producer (litreview md_to_pdf.py) >>")

        # serialize
        out = bytearray()
        out += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        offsets = [0]
        for i, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("latin-1")
            out += to_pdf_bytes(obj)
            out += b"\nendobj\n"
        xref_offset = len(out)
        n = len(objs) + 1
        out += f"xref\n0 {n}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode("latin-1")
        out += b"trailer\n"
        out += f"<< /Size {n} /Root {catalog_id} 0 R /Info {info_id} 0 R >>\n".encode("latin-1")
        out += f"startxref\n{xref_offset}\n%%EOF".encode("latin-1")
        return bytes(out)


def parse_inline_bold(text):
    """split 'foo **bar** baz' into [(foo ,False),(bar,True),(baz,False)]"""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    runs = []
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            runs.append((p[2:-2], True))
        else:
            runs.append((p, False))
    return runs


def convert(md_path, pdf_path, title):
    text = open(md_path, encoding="utf-8").read()
    # Source markdown is soft-wrapped (single \n within a paragraph, blank
    # line between paragraphs). Blocks must be joined on real paragraph
    # boundaries BEFORE inline-bold parsing / re-wrapping -- doing it
    # line-by-line breaks any **bold** span that crosses a source line
    # break, and inserts a spurious paragraph gap after every source line.
    blocks = re.split(r"\n\s*\n", text.strip())
    doc = PDFDoc()
    in_refs = False

    for block in blocks:
        first_line = block.split("\n", 1)[0].strip()
        joined = " ".join(l.strip() for l in block.split("\n") if l.strip())

        if first_line.startswith("# "):
            doc.paragraph([(joined[2:], True)], size=18, leading=1.25, gap_after=10)
            doc.hr_gap(4)
        elif first_line.startswith("### "):
            doc.paragraph([(joined[4:], True)], size=12.5, leading=1.3, gap_after=6)
        elif first_line.startswith("## "):
            heading = joined[3:]
            in_refs = heading.strip().lower() == "references"
            doc.hr_gap(4)
            doc.paragraph([(heading, True)], size=14.5, leading=1.3, gap_after=8)
        elif first_line.startswith("*") and joined.endswith("*") and not first_line.startswith("**"):
            doc.paragraph([(joined.strip("*"), False)], size=9, leading=1.35, gap_after=10, italic_all=True)
        elif in_refs and first_line.startswith("["):
            m = re.match(r"\[(\d+)\]\s*(.*?)\s*\[([^\]]+)\]\(([^)]+)\)\s*$", joined)
            if m:
                num, prefix, label, url = m.groups()
                doc.reference_line(num, prefix, label, url)
            else:
                doc.paragraph([(joined, False)], size=9.5, gap_after=6)
        else:
            runs = parse_inline_bold(joined)
            doc.paragraph(runs, size=10.5, leading=1.42, gap_after=8)

    pdf_bytes = doc.build(title=title)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Wrote {pdf_path}: {len(doc.pages)} pages, {sum(len(p.links) for p in doc.pages)} links")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "../output/generated_survey.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else "../output/generated_survey.pdf"
    convert(src, dst, title="Diffusion Language Models: A Survey")
