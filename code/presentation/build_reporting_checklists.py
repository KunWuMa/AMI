import os
from pathlib import Path
import json,csv,re,unicodedata,html
from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,LongTable,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
R=Path(__file__).resolve().parent
font_regular=Path(os.environ.get('REPORT_FONT_REGULAR', 'fonts/arial.ttf'));font_bold=Path(os.environ.get('REPORT_FONT_BOLD', 'fonts/arialbd.ttf'))
if not (font_regular.exists() and font_bold.exists()):
 import reportlab
 font_regular=Path(reportlab.__file__).parent/'fonts/Vera.ttf';font_bold=Path(reportlab.__file__).parent/'fonts/VeraBd.ttf'
pdfmetrics.registerFont(TTFont('Arial',str(font_regular)));pdfmetrics.registerFont(TTFont('ArialBold',str(font_bold)))
def norm(t):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',t)).strip()
main=[norm(re.sub(r'(?m)^\d{3}\s*$','',p.extract_text())) for p in PdfReader(R/'manuscript_submission.pdf').pages];supp=[norm(re.sub(r'(?m)^\d{3}\s*$','',p.extract_text())) for p in PdfReader(R/'supplement.pdf').pages]
main_titles={'1':'Background','2.1':'Study design, sources and admission selection','2.2':'Prediction landmark and endpoint hierarchy','2.3':'Laboratory harmonization and frozen predictors','2.4':'Care-pathway and observation-representation analyses','2.5':'Observation, joint-event and supporting analyses','2.6':'Statistical analysis and provenance','3.1':'Cohort and observation-dependent endpoints','3.2':'Frozen conditional performance across settings','3.3':'Care-pathway composition and internal gains','3.4':'Observation representation and probability behavior','3.5':'Observation, joint events and supporting analyses','4.1':'Principal findings','4.2':'Observation processes and care pathways','4.3':'Implications for prediction-model evaluation','4.4':'Limitations','5':'Conclusions','Declarations':'Declarations'}
sup_titles={'S1':'Source handling and endpoint implementation','S2':'Model freezing, supporting experiments and evidence limits'}
def find(texts,title):
 # TeX can hyphenate a word at a line ending. Search a dehyphenated equivalent too.
 if title in ['Declarations','Additional files']:
  for i,p in enumerate(PdfReader(R/'manuscript_submission.pdf').pages):
   if title in [line.strip() for line in p.extract_text().splitlines()]:return i+1
 target=norm(title).lower()
 for i,t in enumerate(texts):
  if target in t.lower() or target.replace('-','') in t.lower().replace('- ','').replace('-',''):return i+1
 raise AssertionError('Unlocated heading '+title)
def ranges(texts,titles,last_title=None):
 pairs=list(titles.items());out={}
 for j,(k,title) in enumerate(pairs):
  heading=(k+' '+title) if k!='Declarations' else title
  first=find(texts,heading)
  next_title=((pairs[j+1][0]+' '+pairs[j+1][1]) if pairs[j+1][0]!='Declarations' else pairs[j+1][1]) if j+1<len(pairs) else last_title
  end=first
  if next_title:
   nxt=find(texts,next_title);pos=texts[nxt-1].lower().replace('- ','').find(norm(next_title).lower())
   end=max(first,nxt-(1 if 0<=pos<35 else 0))
  out[k]=str(first) if first==end else str(first)+'-'+str(end)
 return out
abstract_end=find(main,'Keywords')
page={'Title':'1','Abstract':'1' if abstract_end==1 else '1-'+str(abstract_end)}
page.update(ranges(main,main_titles,'Additional files'));page.update(ranges(supp,sup_titles,'Supplementary tables'))
def loc(tokens):
 out=[]
 for k in tokens.split(';'):
  if k in ['Title','Abstract']:out.append(k+' (p. '+str(page[k])+')')
  elif k.startswith('S'):out.append('Suppl. methods '+k+' (p. '+str(page[k])+')')
  elif k=='Declarations':out.append('Declarations (p. '+str(page[k])+')')
  else:out.append('Main '+k+' (p. '+str(page[k])+')')
 return '; '.join(out)
items=json.loads((R/'reporting_items.json').read_text(encoding='utf8'));rowsout=[]
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TitleA',fontName='ArialBold',fontSize=16,leading=20,spaceAfter=12));styles.add(ParagraphStyle(name='HeadA',fontName='ArialBold',fontSize=11,leading=14,spaceAfter=8));styles.add(ParagraphStyle(name='TextA',fontName='Arial',fontSize=9,leading=12,spaceAfter=7));styles.add(ParagraphStyle(name='CellA',fontName='Arial',fontSize=7.5,leading=10));styles.add(ParagraphStyle(name='CellB',fontName='ArialBold',fontSize=7.5,leading=10));P=lambda t,sty='CellA':Paragraph(html.escape(t),styles[sty])
story=[P('Additional file 4: Reporting checklists','TitleA'),P('Cross-setting reliability of anemia prediction after acute myocardial infarction: a retrospective evaluation of observation processes and model transport','TextA'),P('Locations refer to the single-column manuscript and Additional file 1; ranges include section continuations. Item labels are short navigation summaries; the cited official instruments provide full wording. Status distinguishes reporting from methodological sufficiency. Pending items require factual author information and are not certified as completed.','TextA'),P('Sources: Collins GS et al. TRIPOD+AI statement. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378. Benchimol EI et al. RECORD statement. PLoS Med 2015;12:e1001885. doi:10.1371/journal.pmed.1001885.','TextA'),P('Official checklist sources: www.tripod-statement.org; www.record-statement.org. Original checklist sources retained from 15 October 2026; manuscript locations regenerated on 27 October 2026. This study-specific location map is not an endorsement or a methodological quality score.','TextA')]
labels={'TRIPOD_AI':'TRIPOD+AI - 52 reporting subitems','TRIPOD_AI_abstract':'TRIPOD+AI for Abstracts - 13 items','RECORD':'RECORD - 13 routine-data extensions','STROBE_context':'STROBE context for the RECORD extension'}
for j,(key,data) in enumerate(items.items()):
 if j:story.append(PageBreak())
 story.append(Spacer(1,8));story.append(P(labels[key],'HeadA'))
 tb=[[P('Item','CellB'),P('Topic','CellB'),P('Location','CellB'),P('Reporting status / scope','CellB')]]
 for ident,topic,tokens,status,note in data:
  location=loc(tokens);tb.append([P(ident),P(topic),P(location),P(status+'. '+note)]);rowsout.append({'instrument':key,'item':ident,'topic':topic,'location':location,'status':status,'note':note})
 t=LongTable(tb,colWidths=[32,69,193,209],repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.65,colors.black),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#BBBBBB')),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t)
def footer(canvas,doc):
 canvas.setFont('Arial',8);canvas.drawString(45,25,'Additional file 4 - reporting locations');canvas.drawRightString(A4[0]-45,25,str(doc.page))
SimpleDocTemplate(str(R/'Additional_file_4_reporting_checklists.pdf'),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=42,bottomMargin=42,title='Reporting checklists and locations',author='Study authors to complete').build(story,onFirstPage=footer,onLaterPages=footer)
with (R/'reporting_checklist_locations.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rowsout[0]));w.writeheader();w.writerows(rowsout)
(R/'reporting_location_pages.json').write_text(json.dumps(page,indent=2),encoding='utf8');print(json.dumps({'items':len(rowsout),'pdf_pages':len(PdfReader(R/'Additional_file_4_reporting_checklists.pdf').pages)}))
