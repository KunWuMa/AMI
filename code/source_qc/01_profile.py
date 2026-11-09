import os
from pathlib import Path
import json,zipfile,io,xml.etree.ElementTree as ET,re,pickle
import pandas as pd,numpy as np
R=Path(os.environ.get('AMI_SOURCE_QC_ROOT', 'inputs/source_qc'))
def norm(x):
 if x is None or (isinstance(x,float) and not np.isfinite(x)):return ''
 return re.sub(r'^(\d+)\.0$',r'\1',str(x).strip())
ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def readxlsx(b):
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  strings=[]
  if 'xl/sharedStrings.xml' in z.namelist():
   for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns):strings.append(''.join(e.itertext()))
  wb=ET.fromstring(z.read('xl/workbook.xml'));rid=wb.find('m:sheets',ns)[0].attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'];rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(e.attrib['Target'] for e in rels if e.attrib['Id']==rid);path=target.lstrip('/') if target.startswith('/') else 'xl/'+target
  header=None;out=[]
  for _,row in ET.iterparse(io.BytesIO(z.read(path)),events=['end']):
   if row.tag!='{'+ns['m']+'}row':continue
   vals={}
   for c in row:
    ref=c.attrib.get('r','');col=re.sub(r'\d','',ref);v=c.find('m:v',ns);t=c.attrib.get('t');value=''
    if t=='inlineStr':value=''.join(c.find('m:is',ns).itertext())
    elif v is not None:value=strings[int(v.text)] if t=='s' else v.text or ''
    vals[col]=value
   if header is None:header={c:norm(v) for c,v in vals.items() if norm(v)}
   elif any(norm(v) for v in vals.values()):out.append({**{h:vals.get(c,'') for c,h in header.items()},'_file':'','_row':int(row.attrib['r'])})
   row.clear()
  return out

raw={};profile=[]
with zipfile.ZipFile(os.environ.get('AMI_HOSPITAL_ZIP', 'inputs/hospital_export.zip')) as z:
 assert z.testzip() is None
 for n in z.namelist():
  if not n.endswith('.xlsx'):continue
  name=Path(n).name;rows=readxlsx(z.read(n));raw[name]=rows
  profile.append({'file':name,'rows':len(rows),'columns':list(rows[0]) if rows else []})
pickle.dump(raw,open(R/'私有数据_禁止公开/raw_tables.pkl','wb'))
(R/'文件字段概览.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(profile,ensure_ascii=True))
