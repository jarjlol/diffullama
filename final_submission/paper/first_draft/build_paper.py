"""Build the first paper draft: four body pages and one reference page."""
from pathlib import Path
import html,re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Preformatted
HERE=Path(__file__).resolve().parent
OUT=HERE/'output/pdf/paper_first_draft.pdf'
W=A4[0]-90
body=ParagraphStyle('Body',fontName='Times-Roman',fontSize=10.3,leading=13.1,spaceAfter=7)
styles={'body':body,'title':ParagraphStyle('Title',fontName='Times-Bold',fontSize=18,leading=21,spaceAfter=9),
 'h2':ParagraphStyle('H2',fontName='Times-Bold',fontSize=12,leading=15,spaceBefore=8,spaceAfter=5,keepWithNext=True),
 'h3':ParagraphStyle('H3',fontName='Times-Bold',fontSize=10.5,leading=13,spaceBefore=6,spaceAfter=4,keepWithNext=True),
 'cell':ParagraphStyle('Cell',fontName='Times-Roman',fontSize=9,leading=11),
 'code':ParagraphStyle('Code',fontName='Courier',fontSize=8,leading=11,spaceAfter=7)}
def inline(s):
 s=html.escape(s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'https://[^\s]+',lambda m:'<link href="'+m.group()+'" color="#234f70">'+m.group()+'</link>',s)
 return s

def footer(canvas,doc):
 canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#64717b'))
 canvas.drawString(45,25,'BITS F471 Group 3 | First draft - not submission-ready | 21 Nov 2026')
 canvas.drawRightString(A4[0]-45,25,str(doc.page));canvas.restoreState()

def main():
 OUT.parent.mkdir(parents=True,exist_ok=True)
 lines=(HERE/'paper.md').read_text(encoding='utf-8').splitlines();story=[];i=0
 while i<len(lines):
  s=lines[i].strip();i+=1
  if not s:continue
  if s=='<!-- pagebreak -->':story.append(PageBreak());continue
  if s.startswith('```'):
   block=[]
   while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
   i+=1;story.append(Preformatted('\n'.join(block),styles['code']));continue
  if s.startswith('|'):
   rows=[s]
   while i<len(lines) and lines[i].lstrip().startswith('|'):rows.append(lines[i].strip());i+=1
   data=[[Paragraph(inline(c.strip()),styles['cell']) for c in r.strip('|').split('|')] for r in rows if not re.fullmatch(r'[\s|:\-]+',r)]
   cols=len(data[0]);widths=([W*.25]+[W*.15]*5) if cols==6 else ([W*.23,W*.18,W*.29,W*.18,W*.12] if cols==5 else [W*.6,W*.22,W*.18])
   t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
   t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef2f5')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#536574')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#536574')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
   story.extend([t,Spacer(1,8)]);continue
  kind='title' if s.startswith('# ') else 'h2' if s.startswith('## ') else 'h3' if s.startswith('### ') else 'body'
  if kind!='body':s=re.sub(r'^#+\s+','',s)
  else:
   while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','<!--')):s+=' '+lines[i].strip();i+=1
  story.append(Paragraph(inline(s),styles[kind]))
 doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=39,bottomMargin=43,title='Selecting Diffusion-Language-Model Answers - First Draft',author='BITS F471 Group 3',invariant=1)
 doc.build(story,onFirstPage=footer,onLaterPages=footer)
 print(OUT)
if __name__=='__main__':main()
