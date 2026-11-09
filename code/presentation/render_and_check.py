import os,shutil
from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
import subprocess,json,re
R=Path(__file__).resolve().parent;P=R;Q=P/'qa';Q.mkdir(exist_ok=True);pop=os.environ.get('PDFTOPPM') or shutil.which('pdftoppm');report=[]
if not pop: raise RuntimeError('Set PDFTOPPM or install pdftoppm on PATH.')
for name in ['manuscript_reading','manuscript_submission','supplement','Additional_file_4_reporting_checklists']:
 for stale in Q.glob(name+'-*.png'):stale.unlink()
 reader=PdfReader(P/(name+'.pdf'));texts=[p.extract_text() for p in reader.pages];subprocess.run([pop,'-r','62','-png',str(P/(name+'.pdf')),str(Q/name)],check=True,capture_output=True)
 ims=sorted(Q.glob(name+'-*.png'),key=lambda x:int(re.search(r'-(\d+)\.png$',x.name)[1]));sheets=[]
 for k in range(0,len(ims),6):
  canvas=Image.new('RGB',(1110,1650),'#D8DDE2');d=ImageDraw.Draw(canvas)
  for j,path in enumerate(ims[k:k+6]):
   im=Image.open(path).convert('RGB');im.thumbnail((535,790));x=(j%2)*555+(555-im.width)//2;y=(j//2)*550+25;im.thumbnail((535,510));canvas.paste(im,(x,y));d.text(((j%2)*555+12,(j//2)*550+5),name+' p'+str(k+j+1),fill='black')
  dest=Q/(name+'_contact_'+str(k//6+1)+'.png');canvas.save(dest);sheets.append(dest.name)
 # Geometry flag outside page bounds, excluding whitespace-only text. Does not replace visual QA.
 import pdfplumber
 outside=[]
 with pdfplumber.open(P/(name+'.pdf')) as pdf:
  for i,page in enumerate(pdf.pages):
   bad=[c for c in page.chars if c.get('text','').strip() and (c['x0']<-1 or c['x1']>page.width+1 or c['top']<-1 or c['bottom']>page.height+1)]
   if bad:outside.append({'page':i+1,'outside_characters':len(bad)})
 report.append({'file':name+'.pdf','pages':len(texts),'text_lengths':[len(t) for t in texts],'contact_sheets':sheets,'outside_page_characters':outside,'unresolved_reference_marker':any('??' in t for t in texts)})
(P/'pdf_layout_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
