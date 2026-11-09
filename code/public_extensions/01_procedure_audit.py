from pathlib import Path
import pandas as pd,numpy as np,pickle,json,re,hmac,hashlib
R=Path(__file__).resolve().parents[2];S=R.parent.parent;ROOT=S.parent;V=S/'私有数据';O=R/'新增实验/结果';PV=R/'新增实验/私有数据'
O.mkdir(parents=True,exist_ok=True);PV.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(V/'harmonized_cohort.csv.gz');A=pd.read_pickle(V/'CAD_index_cohort.pkl');B=pd.read_pickle(V/'MIMIC_index_cohort.pkl')
# Export only necessary private metadata for the older modeling runtime.
A.to_csv(PV/'local_index_metadata.csv.gz',index=False);B.to_csv(PV/'public_index_metadata.csv.gz',index=False)
raw=pickle.load((ROOT/'双数据包_选题前探索分析/私有数据/raw_cache.pkl').open('rb'))['data']['cad'];H=pd.DataFrame(raw['病案首页表(0).xlsx']);key=(ROOT/'冠脉包_公开数据联合选题_20261012/基础实验_题目1与3/私有数据/pseudonym_key.bin').read_bytes()
def norm(v):return re.sub(r'^(\d+)\.0$',r'\1',str(v).strip()) if v is not None else ''
def uid(x,t):return t+hmac.new(key,(t+norm(x)).encode(),hashlib.sha256).hexdigest()[:16]
H['sid']=H['住院号'].map(lambda v:uid(v,'S'));H['rawsid']=H['住院号'].map(norm);H['rawpid']=H['患者唯一id序号由empi反向更新'].map(norm);H=H[H.sid.isin(A.sid)].drop_duplicates(['sid','rawpid']);assert H.sid.is_unique
lookup=H.set_index('rawsid')[['sid','rawpid']].to_dict('index');a=A.set_index('sid');local=[];mismatch=0
for filename,rows in raw.items():
 if not filename.startswith('医嘱'):continue
 for row in rows:
  rawsid=norm(row.get('住院号/挂号序号'));z=lookup.get(rawsid)
  if z is None:continue
  if norm(row.get('患者唯一id序号由empi反向更新'))!=z['rawpid']:mismatch+=1;continue
  text=norm(row.get('医嘱项目名称'));dt=pd.to_datetime(row.get('医嘱开始时间'),errors='coerce');sid=z['sid']
  if pd.isna(dt) or not a.loc[sid,'admit_time'].normalize()<=dt<=a.loc[sid,'discharge_time'].normalize()+pd.Timedelta(days=1):continue
  cabg=bool(re.search(r'冠状动脉旁路|冠脉旁路|冠状动脉搭桥|冠脉搭桥|CABG',text,re.I))
  pci=bool(re.search(r'冠状动脉.*(?:支架植入|支架置入|球囊扩张|球囊成形)|冠脉.*(?:支架植入|支架置入|球囊扩张|球囊成形)|经皮冠状动脉介入|PCI|PTCA',text,re.I))
  cag=bool(re.search(r'冠状动脉造影|冠脉造影',text))
  if cabg or pci or cag:local.append({'sid':sid,'date':dt,'CABG_order':cabg,'PCI_order':pci,'CAG_order':cag,'order_text':text})
L=pd.DataFrame(local,columns=['sid','date','CABG_order','PCI_order','CAG_order','order_text']);L.to_csv(PV/'local_procedure_order_evidence.csv.gz',index=False)
# Supplement documents have only an admission-like ID. Count exact-ID/time links only; do not infer matches from dates alone.
supp=pickle.load((ROOT/'心内科全方向_候选选题深入验证/私有缓存/all_supplements_source_bound.pkl').open('rb'));supp_rows=supp['rows'];suppids={norm(x.get('住院号')) for x in supp_rows if x.get('住院号') is not None};matches=set(lookup)&suppids
local_summary=[]
for group,c in [('Local_AMI_eligible',D[D.source.eq('CAD')&D.AMI&D.eligible]),('Local_AMI_observed',D[D.source.eq('CAD')&D.AMI&D.y.notna()]),('Local_ACS_eligible',D[D.source.eq('CAD')&D.eligible]),('Local_ACS_observed',D[D.source.eq('CAD')&D.y.notna()])]:
 ids=set(c.sid);q=L[L.sid.isin(ids)];local_summary.append({'cohort':group,'n':len(c),'has_any_coronary_order':q.sid.nunique(),'PCI_order_patients':q.loc[q.PCI_order.astype(bool)].sid.nunique(),'CABG_order_patients':q.loc[q.CABG_order.astype(bool)].sid.nunique(),'CAG_order_patients':q.loc[q.CAG_order.astype(bool)].sid.nunique(),'structured_performed_procedure_table':False,'absence_of_CABG_confirmed':False,'supplement_exact_id_match_patients':int(H[H.rawsid.isin(matches)&H.sid.isin(ids)].sid.nunique())})
pd.DataFrame(local_summary).to_csv(O/'local_procedure_evidence_summary.csv',index=False)
# Public coded procedures are complete only as recorded by this source, not a causal exposure.
b=B.assign(sid='M'+B.hadm_id.astype(str));pubids=set(b.hadm_id);path=ROOT/'公开数据集/MIMIC-IV/2.2/procedures_icd.csv.gz';pieces=[]
for c in pd.read_csv(path,dtype={'icd_code':str},chunksize=300000):
 c=c[c.hadm_id.isin(pubids)].copy();code=c.icd_code.str.replace('.','',regex=False).str.upper();c['CABG']=((c.icd_version==9)&code.str.startswith('361'))|((c.icd_version==10)&code.str.startswith(('0210','0211','0212','0213')));c['PCI']=((c.icd_version==9)&code.str.startswith(('0066','3606','3607')))|((c.icd_version==10)&code.str.startswith(('0270','0271','0272','0273')));c=c[c.CABG|c.PCI]
 if len(c):pieces.append(c)
P=pd.concat(pieces,ignore_index=True);P['sid']='M'+P.hadm_id.astype(str);P['chartdate']=pd.to_datetime(P.chartdate);P.to_csv(PV/'public_procedure_codes.csv.gz',index=False)
flags=P.groupby('sid')[['CABG','PCI']].max();pub=D[D.source.eq('MIMIC')&D.AMI&D.eligible].copy().set_index('sid');pub[['CABG_new','PCI_new']]=flags.reindex(pub.index).fillna(False).astype(bool).rename(columns={'CABG':'CABG_new','PCI':'PCI_new'});assert np.array_equal(pub.CABG_any.astype(bool),pub.CABG_new.astype(bool))
splits=pd.read_csv(V/'MIMIC_patient_split.csv');pub['split']=splits.set_index('sid').split.reindex(pub.index)
summary=[]
for target,c in [('all_eligible',pub),('all_observed',pub[pub.y.notna()]),('test_observed',pub[pub.split.eq('test')])]:
 for name,mask in [('CABG',c.CABG_new),('PCI_without_CABG',c.PCI_new&~c.CABG_new),('no_CABG_or_PCI_code',~c.CABG_new&~c.PCI_new),('all',pd.Series(True,index=c.index))]:
  q=c[mask];summary.append({'population':target,'pathway':name,'n':len(q),'O_observed':int(q.y.notna().sum()),'events':int(q.y.sum()),'conditional_event_fraction':q.y.mean(),'observation_fraction':q.y.notna().mean()})
assert all(sum(z['n'] for z in summary if z['population']==p and z['pathway']!='all')==next(z['n'] for z in summary if z['population']==p and z['pathway']=='all') for p in ['all_eligible','all_observed','test_observed'])
pd.DataFrame(summary).to_csv(O/'public_pathway_composition.csv',index=False)
# Timing of first coded bypass and first post-landmark recorded anemia, at calendar-day resolution.
U=pd.read_pickle(V/'MIMIC_common_labs.pkl');U['sid']='M'+U.hadm_id.astype(str);hb=U[U.variable.eq('hb')&U.day.ge(2)&U.day.le(7)&U.value_std.notna()].copy();hb=hb.merge(pub[['cut']].reset_index(),on='sid');an=hb[hb.value_std.lt(hb.cut)].groupby('sid').chart.min();first=P[P.CABG].groupby('sid').chartdate.min();meta=b.set_index('sid');rec=[]
for sid in pub[pub.CABG_new].index:
 day=int((first.loc[sid].normalize()-meta.loc[sid,'D0']).days);event=an.get(sid,pd.NaT);relation='no_recorded_event' if pd.isna(event) else 'before_CABG_date' if event.normalize()<first.loc[sid] else 'same_calendar_date' if event.normalize()==first.loc[sid] else 'after_CABG_date'
 rec.append({'sid':sid,'first_CABG_day':day,'CABG_before_D2':day<2,'first_recorded_anemia_relation':relation,'test':sid in set(splits[splits.split.eq('test')].sid)})
T=pd.DataFrame(rec);T.to_csv(PV/'public_CABG_timing.csv.gz',index=False);T.groupby(['CABG_before_D2','first_recorded_anemia_relation'],dropna=False).size().rename('n').reset_index().to_csv(O/'public_CABG_timing_summary.csv',index=False)
meta_audit={'local_order_patient_identity_mismatches':mismatch,'supplement_unique_ids':len(suppids),'supplement_exact_id_overlap_with_index_admissions':len(matches),'local_procedure_status':'orders are not evidence of execution; missing CABG evidence is not a verified exclusion','public_CABG_flags_reproduced':True,'public_CABG_date_resolution':'calendar day; same-day order is unknown','no_new_data':True}
(O/'procedure_audit.json').write_text(json.dumps(meta_audit,indent=2),encoding='utf8');print(json.dumps({'audit':meta_audit,'local_summary':local_summary,'public_summary':summary},default=str,ensure_ascii=True))
