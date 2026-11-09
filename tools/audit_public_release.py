"""Static public-package validation; does not execute clinical analysis scripts."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','__pycache__','private_cache'}
files=[p for p in R.rglob('*') if p.is_file() and not set(p.relative_to(R).parts)&EXCLUDE]
errors=[];schemas=[];counts={'python':0,'csv':0,'pdf':0}
blocked={'.pkl','.pickle','.joblib','.parquet','.feather','.xlsx','.xls','.gz','.zip','.7z','.bin','.key','.pem'}
fields={'sid','pid','raw_sid','raw_pid','subject_id','hadm_id','patient_id','住院号','身份证号','姓名','admittime','dischtime'}
patterns={'workstation_path':r'(?i)(?:[A-Z]:[/\\](?:Users|心内科)|/Users/[^/]+/|/home/[^/]+/)', 'github_token':r'gh[pousr]_[A-Za-z0-9]{30,}', 'github_pat':r'github_pat_[A-Za-z0-9_]{30,}', 'private_key':r'-----BEGIN [A-Z ]*PRIVATE KEY-----', 'patient_pseudonym':r'(?<![A-Za-z0-9])[SP][a-f0-9]{16}(?![A-Za-z0-9])', 'mimic_id_literal':r'(?<![A-Za-z0-9])M[12][0-9]{7}(?![A-Za-z0-9])'}
for p in files:
 rel=p.relative_to(R).as_posix()
 if p.suffix.lower() in blocked:errors.append({'file':rel,'check':'prohibited binary/archive format'})
 if p.name.startswith('.env'):errors.append({'file':rel,'check':'environment secret file'})
 text=''
 if p.suffix=='.pdf':
  counts['pdf']+=1;reader=PdfReader(p);text='\n'.join(page.extract_text() or '' for page in reader.pages)+str(reader.metadata)
  if reader.attachments:errors.append({'file':rel,'check':'PDF embedded attachment'})
 elif p.suffix in {'.py','.csv','.json','.md','.tex','.bib','.txt','.cls','.bst'} or p.name in {'.gitignore','.gitattributes'}:text=p.read_text(encoding='utf-8-sig')
 for k,pattern in patterns.items():
  if rel=='tools/audit_public_release.py' and k=='workstation_path':continue  # literal detection patterns are not actual paths
  if re.search(pattern,text):errors.append({'file':rel,'check':k})
 if p.suffix=='.py':
  counts['python']+=1
  try:ast.parse(text,filename=rel)
  except SyntaxError as e:errors.append({'file':rel,'check':'syntax','line':e.lineno})
 if p.suffix=='.csv':
  counts['csv']+=1
  rows=list(csv.DictReader(text.splitlines()));cols=list(rows[0]) if rows else []
  schemas.append({'file':rel,'rows':len(rows),'columns':cols})
  if set(cols)&fields:errors.append({'file':rel,'check':'patient-level schema'})
 if p.name=='supporting_results.json':
  for k,v in json.loads(text).items():
   if set(v.get('columns',[]))&fields:errors.append({'file':rel,'check':'patient-level nested schema','table':k})
report={'passed':not errors,'file_counts':counts,'issues':errors,'csv_schemas':schemas,'scope':'Static schemas, syntax, identifier-like literals, credential patterns, workstation paths, PDF text/metadata/attachments. Curated aggregate-only inputs; no model execution.'}
(R/'docs/public_release_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
manifest=[]
for p in sorted(R.rglob('*')):
 if p.is_file() and not set(p.relative_to(R).parts)&EXCLUDE and p.name!='release_manifest.csv':manifest.append({'file':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
with (R/'release_manifest.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['file','bytes','sha256'],lineterminator='\n');w.writeheader();w.writerows(manifest)
print(json.dumps({'passed':not errors,'files':len(manifest),'counts':counts,'issues':errors},ensure_ascii=False))
raise SystemExit(1 if errors else 0)
