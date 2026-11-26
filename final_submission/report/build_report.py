"""Render report.md to report.pdf with explicit page grouping.

Requires reportlab. Run: python3 final_submission/report/build_report.py
The Markdown remains the content source; no measured values live in this renderer.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
TEXT = (HERE / "report.md").read_text(encoding="utf-8")
REFS = dict(re.findall(r"^\[(S\d+)\]:\s*(\S+)\s*$", TEXT, re.M))
BLUE = colors.HexColor("#18394f")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=18,
                          leading=22, spaceAfter=7, textColor=BLUE))
styles.add(ParagraphStyle("ReportHeading", fontName="Helvetica-Bold", fontSize=11.5,
                          leading=15, spaceBefore=11, spaceAfter=5, textColor=BLUE,
                          keepWithNext=True))
styles.add(ParagraphStyle("ReportBody", fontName="Helvetica", fontSize=11,
                          leading=15.5, spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle("ReportCell", parent=styles["ReportBody"], fontSize=8.6,
                          leading=11.2, spaceAfter=0))
styles.add(ParagraphStyle("ReportBullet", parent=styles["ReportBody"],
                          leftIndent=10, firstLineIndent=-7, spaceAfter=3))


def inline(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", out)
    # Grouped source references become clickable labels, not long URLs.
    def cite(match):
        ids = re.findall(r"S\d+", match.group())
        links = []
        for key in ids:
            if key not in REFS:
                raise ValueError(f"Unknown reference: {key}")
            url = REFS[key]
            if not url.startswith("https://"):
                # Relative links remain valid when the report folder travels together.
                url = "./" + url
            links.append(f'<a href="{html.escape(url, quote=True)}" color="#245579">{key}</a>')
        return "[" + ", ".join(links) + "]"
    return re.sub(r"\[S\d+(?:, S\d+)*\]", cite, out)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#53616d"))
    canvas.drawString(43, 27, "BITS F471 Group 3 | Task 5 report")
    canvas.drawRightString(A4[0] - 43, 27, str(doc.page))
    canvas.restoreState()


def main():
    story = []
    lines = TEXT.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or re.match(r"^\[S\d+\]:", line):
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["ReportTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["ReportHeading"]))
        elif line.startswith("|"):
            raw = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw.append(lines[i].strip()); i += 1
            cells = [[v.strip() for v in r.strip("|").split("|")]
                     for r in raw if not re.fullmatch(r"[|:\-\s]+", r)]
            data = [[Paragraph(inline(v), styles["ReportCell"]) for v in row]
                    for row in cells]
            widths = [111] + [(A4[0] - 86 - 111) / (len(cells[0]) - 1)] * (len(cells[0]) - 1)
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5")),
                ("LINEBELOW", (0, 0), (-1, 0), .6, BLUE),
                ("LINEBELOW", (0, -1), (-1, -1), .4, BLUE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 8)])
        elif line.startswith("- "):
            story.append(Paragraph("- " + inline(line[2:]), styles["ReportBullet"]))
        else:
            para = [line]
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "<!--", "- ", "|", "[S")):
                para.append(lines[i].strip()); i += 1
            story.append(Paragraph(inline(" ".join(para)), styles["ReportBody"]))
    doc = SimpleDocTemplate(str(HERE / "report.pdf"), pagesize=A4,
                            rightMargin=43, leftMargin=43, topMargin=38, bottomMargin=43,
                            title=lines[0].removeprefix("# ").strip(),
                            author="BITS F471 Group 3", invariant=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(HERE / "report.pdf")


if __name__ == "__main__":
    main()
