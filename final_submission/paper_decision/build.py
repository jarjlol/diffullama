"""Assemble the decision-model edition of the paper (PR #76 results only) and build a PDF.

Same layout and checks as final_submission/paper/build.py; sections and bib live in this folder.

    python3 final_submission/paper/build.py                  # sections from this checkout
    python3 final_submission/paper/build.py --from-branches  # each section from its PR branch (preview before merge)
    python3 final_submission/paper/build.py --with-appendix  # also append the provenance appendix (not in the submission PDF)

Needs: pandoc (>= 2.x) and Node with Playwright's Chromium (`npm i playwright` gives `require('playwright')`).
Writes final_submission/paper/build/{paper.md, paper.html, paper.pdf} and prints a report:
pages, missing sections, unresolved citations and bibliography duplicates.

What it does, in order:
1. Reads the seven sections in paper order. HTML comments (owner notes, instructions) are dropped; a section
   that is still the "_TODO: draft here._" stub is reported as missing.
2. Merges every refs_*.bib (deduplicated by arXiv ID, then by key; conflicts are reported, first one kept).
3. Normalises citations to author-year: "[key]" or "[key1; key2]" becomes "(Surname et al., Year; ...)".
   Existing "(Surname et al., Year)" text is matched to a bib entry by first-author surname and year.
   Every cited entry goes into References; anything unmatched is listed so the writer can fix it.
4. pandoc -> HTML (MathML), two-column A4 CSS -> Chromium PDF. The reference list starts on a new page, so
   "body pages" (limit 4) and "reference pages" (not capped; Moodle: "4+ reference pages") can be counted separately.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "build"
ORDER = [("abstract", "Abstract", 13), ("introduction", "Introduction", 8), ("related_work", "Related Work", 9),
         ("problem_formulation", "Problem Formulation", 10), ("methodology", "Methodology", 11),
         ("experiments", "Experiments", 12), ("conclusion", "Conclusion", 13)]
BRANCH: dict[int, str] = {}  # all section PRs merged; add "issue: branch" here to preview an open section PR
# Sections are read from main unless listed above.


def git_show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{path}"], capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def read(path: str, issue: int, from_branches: bool) -> tuple[str | None, str]:
    if from_branches:
        for ref in ([f"origin/{BRANCH[issue]}"] if issue in BRANCH else []) + ["origin/main"]:
            t = git_show(ref, path)
            if t is not None:
                return t, ref
        return None, "missing"
    p = ROOT / path
    return (p.read_text(encoding="utf-8"), "checkout") if p.exists() else (None, "missing")


def parse_bib(text: str) -> dict[str, dict]:
    entries = {}
    for m in re.finditer(r"@\w+\{([^,\s]+),(.*?)\n\}", text, re.S):
        fields = {k.lower(): " ".join(v.split()) for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(2))}
        entries[m.group(1)] = fields
    return entries


def surname(e: dict) -> str:
    first = e.get("author", "Anon").split(" and ")[0].strip()
    return first.split(",")[0].strip() if "," in first else first.split()[-1]


def short(e: dict) -> str:
    n = len(e.get("author", "").split(" and "))
    tail = " et al." if n > 2 else (f" and {surname({'author': e['author'].split(' and ')[1]})}" if n == 2 else "")
    return f"{surname(e)}{tail}, {e.get('year', 'n.d.')}"


def pdf_pages(data: bytes) -> int:
    """Page count from the PDF page tree (/Type /Pages ... /Count n); outline /Count entries are ignored."""
    counts = [int(m.group(1)) for d in re.findall(rb"<<(?:[^<>]|<<[^<>]*>>)*>>", data) if re.search(rb"/Type\s*/Pages\b", d)
              for m in [re.search(rb"/Count\s+(\d+)", d)] if m]
    return max(counts) if counts else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-branches", action="store_true")
    ap.add_argument("--with-appendix", action="store_true",
                    help="append appendix_provenance.md after the references (off by default: the brief is 4 body + 1 reference page, #69)")
    a = ap.parse_args()
    report, body = [], []
    appendix_md = None

    bibs, sources = {}, []
    for p in sorted((HERE).glob("refs_*.bib")):
        sources.append((p.name, p.read_text(encoding="utf-8")))
    if a.from_branches:
        for issue, name in ((9, "refs_related_work.bib"), (11, "refs_methodology.bib")):
            t = git_show(f"origin/{BRANCH[issue]}" if issue in BRANCH else "origin/main", f"final_submission/paper/{name}")
            if t and name not in {n for n, _ in sources}:
                sources.append((name, t))
    seen_eprint = {}
    for name, text in sources:
        for k, e in parse_bib(text).items():
            ep = e.get("eprint")
            if k in bibs or (ep and ep in seen_eprint):
                other = bibs.get(k) or bibs[seen_eprint[ep]]
                if other.get("year") != e.get("year") or other.get("title") != e.get("title"):
                    report.append(f"bib conflict: {k} in {name} differs from the first copy (kept the first)")
                continue
            bibs[k] = dict(e, _file=name)
            if ep:
                seen_eprint[ep] = k
    by_name_year = {}
    for k, e in bibs.items():
        by_name_year.setdefault((surname(e).lower(), e.get("year")), k)
        y = re.search(r"(\d{4})", k)               # key year (e.g. wang2023...) as a second lookup
        if y:
            by_name_year.setdefault((surname(e).lower(), y.group(1)), k)

    cited, unresolved = [], []

    def cite_keys(m):
        keys = [k.strip() for k in re.split(r"[;,]", m.group(1))]
        if not all(k in bibs for k in keys):
            unresolved.extend(k for k in keys if k not in bibs)
            return m.group(0)
        for k in keys:                      # IEEE: number by first citation
            if k not in cited:
                cited.append(k)
        return "[" + ", ".join(str(cited.index(k) + 1) for k in keys) + "]"

    for slug, title, issue in ORDER:
        text, src = read(f"final_submission/paper_decision/sections/{slug}.md", issue, a.from_branches)
        if text is None:
            report.append(f"missing section: {slug}"); continue
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()
        text = re.sub(r"^#\s+.*\n", "", text, count=1).strip()          # the file's own H1; we add ours
        if not text or "_TODO: draft here._" in text:
            report.append(f"missing section: {slug} (still the stub, source {src})"); continue
        text = re.sub(r"\[([a-z][a-z0-9]*\d{4}[a-z0-9]*(?:\s*[;,]\s*[a-z][a-z0-9]*\d{4}[a-z0-9]*)*)\]", cite_keys, text)
        for sn, yr in re.findall(r"([A-Z][A-Za-z\-]+) et al\.,? (\d{4})", text):
            k = by_name_year.get((sn.lower(), yr))
            (cited.append(k) if k else unresolved.append(f"{sn} et al., {yr}"))
        text = re.sub(r"^## ", "### ", text, flags=re.M)
        if slug == "abstract":
            body.append(f"::: {{.abstract}}\n*Abstract*—{text}\n:::\n")
        else:
            body.append(f"## {title}\n\n{text}\n")
        report.append(f"section {slug}: from {src}")

    # Optional appendix (#36), only with --with-appendix: the 25 Nov brief is "4+1(reference) page paper" (#69), so the
    # submission PDF leaves it out; the file stays in the repo as evidence.
    app, asrc = (None, "") if not a.with_appendix else (git_show("origin/mesh/36-provenance-appendix", "final_submission/paper/appendix_provenance.md"), "origin/mesh/36-provenance-appendix") if a.from_branches else (None, "")
    if a.with_appendix and app is None and (HERE / "appendix_provenance.md").exists():
        app, asrc = (HERE / "appendix_provenance.md").read_text(encoding="utf-8"), "checkout"
    if a.with_appendix and app is None and a.from_branches:
        app, asrc = git_show("origin/main", "final_submission/paper/appendix_provenance.md"), "origin/main"
    if app:
        app = re.sub(r"<!--.*?-->", "", app, flags=re.S).strip()
        app = re.sub(r"^#\s+(.*)\n", r"## \1\n", app, count=1)
        appendix_md = app
        report.append(f"appendix provenance: from {asrc}")

    refs = list(dict.fromkeys(cited))
    lines = []

    def ieee_name(n: str) -> str:
        n = n.strip()
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
        else:
            parts = n.split(); last, first = parts[-1], " ".join(parts[:-1])
        initials = " ".join(f"{w[0]}." for w in re.split(r"[\s]+", first) if w)
        return f"{initials} {last}".strip()

    for i, k in enumerate(refs, 1):
        e = bibs[k]
        au = [ieee_name(x) for x in e.get("author", "").split(" and ")]
        au_s = (", ".join(au[:3]) + " et al.") if len(au) > 3 else (" and ".join(au) if len(au) <= 2 else ", ".join(au[:-1]) + ", and " + au[-1])
        ident = f" arXiv:{e['eprint']}" if e.get("eprint") else ""
        lines.append(f"[\\[{i}\\]]{{.rn}} {au_s}, “{e.get('title', '').strip('{}')},”{ident}, {e.get('year', 'n.d.')}.\n")
    md = ("---\ntitle: \"Knowing the Answer or Checking It? A Small Decision Model as an Answer Selector for Diffusion Language Models\"\n---\n\n"
          + "::: {.cols}\n\n" + "\n".join(body) + "\n:::\n\n::: {.refs}\n\n## References\n\n" + "\n".join(lines) + "\n:::\n" + (f"\n::: {{.appendix}}\n\n{appendix_md}\n\n:::\n" if appendix_md else ""))
    OUT.mkdir(exist_ok=True)
    (OUT / "paper.md").write_text(md, encoding="utf-8")
    css = OUT / "paper.css"
    css.write_text("""@page { size: A4; margin: 17mm 14mm 18mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.2; counter-reset: sec; }
.cols { columns: 2; column-gap: 6mm; }
header#title-block-header { text-align: center; } h1.title { font-size: 20pt; font-weight: normal; line-height: 1.15; margin: 0 0 5mm; }
.abstract p { font-weight: bold; font-size: 9pt; text-align: justify; margin: 0 0 2mm; }
.abstract em { font-style: italic; }
h2 { font-size: 10pt; font-weight: normal; font-variant: small-caps; text-align: center; margin: 3mm 0 1.5mm; counter-increment: sec; counter-reset: sub; }
.cols h2::before { content: counter(sec, upper-roman) ". "; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 2mm 0 1mm; counter-increment: sub; }
h3::before { content: counter(sub, upper-alpha) ". "; }
p { margin: 0 0 1.4mm; text-align: justify; text-indent: 0; }
.tcap p { text-align: center; font-variant: small-caps; font-size: 8pt; margin: 2mm 0 1mm; text-indent: 0; }
table { border-collapse: collapse; font-size: 7.5pt; width: 100%; margin: 1mm 0 2mm; border-top: 0.8pt solid #000; border-bottom: 0.8pt solid #000; }
th { border-bottom: 0.5pt solid #000; padding: 0.5mm 1mm; font-weight: normal; } td { padding: 0.4mm 1mm; }
ul, ol { padding-left: 4mm; margin: 0 0 1.4mm; } li { margin: 0 0 0.6mm; }
.refs { break-before: page; columns: 2; column-gap: 6mm; font-size: 8pt; }
.refs h2 { counter-increment: none; } .refs p { text-align: left; padding-left: 6mm; text-indent: -6mm; margin: 0 0 1mm; }
.rn { display: inline-block; width: 6mm; text-indent: 0; }
.appendix { break-before: page; font-size: 7.5pt; } math { font-size: 95%; }
""", encoding="utf-8")
    subprocess.run(["pandoc", str(OUT / "paper.md"), "-s", "--mathml", "-c", "paper.css", "-o", str(OUT / "paper.html")], check=True)
    js = """const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://'+process.argv[2]);await p.pdf({path:process.argv[3],format:'A4',preferCSSPageSize:true});
const n=await p.evaluate(()=>{const r=document.querySelector('.refs');return r?1:0});await b.close();})();"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(js)
    subprocess.run(["node", f.name, str(OUT / "paper.html"), str(OUT / "paper.pdf")], check=True, cwd=str(ROOT))
    pdf = (OUT / "paper.pdf").read_bytes()
    pages = pdf_pages(pdf)
    # body pages: render once more without the reference list
    html = (OUT / "paper.html").read_text(encoding="utf-8")
    body_html = OUT / "_body.html"
    body_html.write_text(re.sub(r'<(div|section)[^>]*class="(refs|appendix)"[^>]*>.*?</\1>', "", html, flags=re.S), encoding="utf-8")
    subprocess.run(["node", f.name, str(body_html), str(OUT / "_body.pdf")], check=True, cwd=str(ROOT))
    body_pages = pdf_pages((OUT / "_body.pdf").read_bytes())
    body_html.unlink(); (OUT / "_body.pdf").unlink(); Path(f.name).unlink()

    print("\n".join(report))
    print(f"references: {len(refs)} cited of {len(bibs)} in the merged bib")
    print("unresolved citations: " + (", ".join(sorted(set(unresolved))) or "none"))
    full_body = None  # pages before the References page in the real PDF (the re-render above can paginate differently)
    try:
        txt = subprocess.run(["pdftotext", "-layout", str(OUT / "paper.pdf"), "-"], capture_output=True, text=True).stdout
        for i, pg in enumerate(txt.split("\f")):
            if any(re.match(r"\s*References(\s{2,}|$)", line) for line in pg.splitlines()):  # -layout puts the next column on the same line
                full_body = i
                break
    except FileNotFoundError:
        pass
    if full_body is not None:
        body_pages = full_body
    print(f"pages: {pages} total, {body_pages} body (limit 4), {pages - body_pages} " + ("references + appendix" if appendix_md else "references (limit 1)")
          + ("" if full_body is not None else "  [body counted from a re-render without references: pdftotext missing or no References page found]"))
    try:
        font = subprocess.run(["fc-match", "-f", "%{family}", "Times New Roman"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        font = ""  # no fontconfig: cannot tell which font Chromium used
    if not re.search(r"Times|Liberation Serif|Tinos|Nimbus Roman", font):
        print(f"WARNING: 'Times New Roman' resolves to '{font or 'unknown'}' here, which is not metric-compatible; the page count "
              "is not the submission count (DejaVu Serif gives one extra body page). Build the final PDF where Times New Roman "
              "or Liberation Serif is installed (e.g. apt-get install fonts-liberation).")
    print(f"wrote {OUT / 'paper.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
