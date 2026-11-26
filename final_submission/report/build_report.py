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
INK = colors.HexColor("#172A3D")
BODY = colors.HexColor("#243444")
TEAL = colors.HexColor("#136E65")
MUTED = colors.HexColor("#546477")
RULE = colors.HexColor("#D6E2E6")
PALE = colors.HexColor("#F2F8F7")
MARGIN = 48
CONTENT_WIDTH = A4[0] - 2 * (MARGIN + 6)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportTitle", fontName="Times-Bold", fontSize=32,
                          leading=34, rightIndent=85, spaceAfter=12, textColor=INK, keepWithNext=True))
styles.add(ParagraphStyle("ReportSubtitle", fontName="Helvetica-Bold", fontSize=10,
                          leading=14, spaceAfter=17, textColor=TEAL, keepWithNext=True))
styles.add(ParagraphStyle("ReportHeading", fontName="Helvetica-Bold", fontSize=15,
                          leading=19, spaceBefore=16, spaceAfter=8, textColor=INK,
                          keepWithNext=True))
styles.add(ParagraphStyle("ReportBody", fontName="Helvetica", fontSize=11.4,
                          leading=17.5, spaceAfter=10, alignment=TA_LEFT, textColor=BODY))
styles.add(ParagraphStyle("ReportCell", parent=styles["ReportBody"], fontSize=10.5,
                          leading=13.8, spaceAfter=0))
styles.add(ParagraphStyle("ReportTableHead", parent=styles["ReportCell"],
                          fontName="Helvetica-Bold", textColor=TEAL))
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
            links.append(f'<a href="{html.escape(url, quote=True)}" color="#136E65">{key}</a>')
        return "[" + ", ".join(links) + "]"
    return re.sub(r"\[S\d+(?:, S\d+)*\]", cite, out)


def page_details(canvas, doc):
    canvas.saveState()
    width, height = A4
    left = MARGIN + 6
    right = width - left
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.65)
    canvas.line(left, height - 38, right, height - 38)
    canvas.line(left, 40, right, 40)
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.8)
    canvas.line(left, height - 38, left + 32, height - 38)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(left, height - 28, doc.title)
    canvas.drawString(left, 24, "BITS F471 Group 3  |  Task 5 discussion report")
    canvas.drawRightString(right, 24, str(doc.page))
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
            heading = inline(line[3:])
            heading = re.sub(r"^(Task [1-5](?: continued)?\.|Conclusion\.)",
                             r'<font color="#136E65">\1</font>', heading)
            story.append(Paragraph(heading, styles["ReportHeading"]))
        elif line.startswith("BITS F471 "):
            story.append(Paragraph(inline(line), styles["ReportSubtitle"]))
        elif line.startswith("|"):
            raw = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw.append(lines[i].strip()); i += 1
            cells = [[v.strip() for v in r.strip("|").split("|")]
                     for r in raw if not re.fullmatch(r"[|:\-\s]+", r)]
            data = [
                [Paragraph(inline(value), styles["ReportTableHead" if index == 0 else "ReportCell"])
                 for value in row]
                for index, row in enumerate(cells)
            ]
            widths = [95] + [(CONTENT_WIDTH - 95) / (len(cells[0]) - 1)] * (len(cells[0]) - 1)
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
                ("LINEBELOW", (0, 1), (-1, -1), 0.55, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (0, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 6)])
        elif line.startswith("- "):
            story.append(Paragraph("- " + inline(line[2:]), styles["ReportBullet"]))
        else:
            para = [line]
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "<!--", "- ", "|", "[S")):
                para.append(lines[i].strip()); i += 1
            story.append(Paragraph(inline(" ".join(para)), styles["ReportBody"]))
    doc = SimpleDocTemplate(str(HERE / "report.pdf"), pagesize=A4,
                            rightMargin=MARGIN, leftMargin=MARGIN, topMargin=58, bottomMargin=51,
                            title=lines[0].removeprefix("# ").strip(),
                            author="BITS F471 Group 3", invariant=1)
    doc.build(story, onFirstPage=page_details, onLaterPages=page_details)
    print(HERE / "report.pdf")


if __name__ == "__main__":
    main()
