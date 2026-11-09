import os
from pathlib import Path
import pandas as pd,numpy as np,json,pickle,zipfile,io,openpyxl,hashlib
R=Path(os.environ.get('AMI_SOURCE_QC_ROOT', 'inputs/source_qc'));V=R/'私有数据_禁止公开';A=json.loads((R/'新数据核查统计.json').read_text(encoding='utf8'));B=json.loads((R/'新旧数据交叉核验.json').read_text(encoding='utf8'));h=pd.read_csv(V/'全部住院与候选队列筛选_含旧包既往史校正.csv',dtype={'raw_sid':str,'raw_pid':str});m=h[h.candidate_joint_history&h.O.eq(1)].copy();op=pd.read_csv(V/'手术及输血名称候选_不得直接视为已执行.csv',dtype={'raw_sid':str,'raw_pid':str});raw=pickle.load(open(V/'raw_tables.pkl','rb'));checks=[]
def ck(n,v):checks.append({'check':n,'passed':bool(v)})
# Independent XLSX reader verifies every source row count and header, preserving the original archive.
with zipfile.ZipFile(os.environ.get('AMI_HOSPITAL_ZIP', 'inputs/hospital_export.zip')) as z:
 for n in z.namelist():
  if not n.endswith('.xlsx'):continue
  name=Path(n).name;wb=openpyxl.load_workbook(io.BytesIO(z.read(n)),read_only=True,data_only=True);ws=wb.worksheets[0];ws.reset_dimensions();it=ws.iter_rows(values_only=True);header=list(next(it));count=sum(any(v is not None and str(v).strip() for v in row) for row in it);ck(name+'_independent_row_count',count==len(raw[name]));ck(name+'_independent_headers',set(str(x).strip() for x in header if x is not None)==set(k for k in raw[name][0] if not k.startswith('_')));wb.close()
ck('candidate_100_unique_people',len(m)==100 and m.raw_pid.nunique()==100);ck('event_counts_reconcile',int(m.y.eq(1).sum())==46 and int(m.y.eq(0).sum())==54);ck('frozen74_features_and_labels_identical',B['main_overlap_value_check']['label_different']==0 and all(v==0 for v in B['main_overlap_value_check']['feature_difference_counts'].values()));ck('no_unknown_as_negative',h.loc[h.early_normal&h.O.eq(0),'y'].isna().all())
# A readable local review listing keeps procedure/transfusion flags explicitly provisional.
review=m[['raw_sid','raw_pid','admit_time','discharge_time','age_raw','male','primary_diagnosis','last_hb','early_min_hb','late_min_hb','future_hb_days','y','old_stay_metadata_match']].copy()
for flag,col in [('CABG_name_supported','搭桥名称证据_待核验'),('PCI_candidate','PCI名称证据_待核验'),('transfusion_candidate','输血操作名称证据_待核验')]:
 names=op[op[flag]].groupby('raw_sid')['name'].apply(lambda x:';'.join(sorted(set(x))));review[col]=review.raw_sid.map(names).fillna('')
review['医生判定']='';review.to_csv(V/'候选100例与治疗路径证据_仅院内核验.csv',index=False,encoding='utf-8-sig')
# The candidate cohort remains a numerical feasibility set; no new model evaluation is performed.
extra={'candidate_years':m.groupby(pd.to_datetime(m.admit_time).dt.year).size().to_dict(),'unmatched_historical_records_events':int(m.loc[~m.old_stay_metadata_match,'y'].eq(1).sum()),'unmatched_historical_records_nonevents':int(m.loc[~m.old_stay_metadata_match,'y'].eq(0).sum()),'joint_early_normal_exclusion_counts':{},'checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks)}
f=h[h.first_observed_ACS&h.primary_AMI&h.over_D2_strict];extra['joint_early_normal_exclusion_counts']={'early_low_hb':int((f.early_min_hb<f.cut).sum()),'no_early_hb':int(f.early_hb_rows.eq(0).sum()),'early_hb_uninterpretable':int(f.early_hb_uncertain.sum()),'early_normal':int(f.early_normal.sum())};(R/'独立复核与最终检查.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(extra,ensure_ascii=True))
# Private files are kept outside manuscript/public supplements. Record file hashes for later audit.
manifest=[]
for p in V.iterdir():
 if p.is_file():manifest.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(R/'私有文件清单与校验值.csv',index=False,encoding='utf-8-sig')
