import os
from pathlib import Path
import pandas as pd,numpy as np,json,hmac,hashlib,re
R=Path(__file__).resolve().parents[1];S=R.parent;A=Path(os.environ.get('AMI_SOURCE_QC_ROOT', 'inputs/source_qc'));V=R/'私有数据_禁止公开';O=R/'结果汇总';h=pd.read_csv(A/'私有数据_禁止公开/全部住院与候选队列筛选_含旧包既往史校正.csv',dtype={'raw_sid':str,'raw_pid':str});l=pd.read_csv(A/'私有数据_禁止公开/八项化验D0至D7清洗明细_单位待核实.csv',dtype={'raw_sid':str,'raw_pid':str});cfg=json.loads((A/'研究者确认的数据说明_20261022.json').read_text(encoding='utf8'));key=Path(os.environ.get('AMI_PSEUDONYM_KEY', 'inputs/private/pseudonym_key.bin')).read_bytes()
def uid(x,t):return t+hmac.new(key,(t+str(x)).encode(),hashlib.sha256).hexdigest()[:16]
h['sid']=h.raw_sid.map(lambda x:uid(x,'S'));h['pid']=h.raw_pid.map(lambda x:uid(x,'P'));h['AMI']=h.primary_AMI;h['year']=pd.to_datetime(h.admit_time).dt.year;h['source']='LOCAL_NEW';h['eligible']=h.candidate_joint_history
for f in ['admit_time','discharge_time','D0','landmark']:h[f]=pd.to_datetime(h[f])
h['ICU_any']=np.nan;h['CABG_any']=np.nan
for c in ['report_time','test_time']:l[c]=pd.to_datetime(l[c]);
l['sid']=l.raw_sid.map(lambda x:uid(x,'S'));l['unit_confirmed']=l.variable.map(cfg['units']);l['unit_source']='researcher_confirmation_20261022'
for r in h.itertuples():
 g=l[l.sid.eq(r.sid)&l.variable.eq('hb')];f=g[g.day.between(2,7)].dropna(subset=['value_std']);d=f.groupby('day').value_std.median().sort_index()
 for name,val in {'y_daily_median':float(d.min()<r.cut) if len(d) else np.nan,'y_unique_day':float(d.min()<r.cut) if len(d) else np.nan,'y_last_day':float(d.iloc[-1]<r.cut) if len(d) else np.nan,'y_lt110':float(f.value_std.min()<110) if len(f) else np.nan,'y_drop20':float(r.first_hb-f.value_std.min()>=20) if len(f) else np.nan,'y_D2D3':float(f.loc[f.day.le(3),'value_std'].min()<r.cut) if f.day.le(3).any() else np.nan,'y_D2D5':float(f.loc[f.day.le(5),'value_std'].min()<r.cut) if f.day.le(5).any() else np.nan}.items():h.loc[h.sid.eq(r.sid),name]=val
op=pd.read_csv(A/'私有数据_禁止公开/手术及输血名称候选_不得直接视为已执行.csv',dtype={'raw_sid':str});op['time']=pd.to_datetime(op.time);op['sid']=op.raw_sid.map(lambda x:uid(x,'S'))
for name,col in [('CABG_name_supported','CABG_recorded_name'),('PCI_candidate','PCI_recorded_name'),('transfusion_candidate','transfusion_recorded_name')]:h[col]=h.sid.isin(op.loc[op[name],'sid'])
# Original records are used for private linkage only, never exported in the public package.
import pickle
raw=pickle.load(open(A/'私有数据_禁止公开/raw_tables.pkl','rb'));doc=pd.DataFrame(raw['文书病历主文档(0).xlsx']);doc['sid']=doc['门诊就诊号/住院号'].map(lambda x:uid(x,'S'))
for typ,col in [('手术记录','has_operation_note'),('输血记录','has_transfusion_note')]:h[col]=h.sid.isin(doc.loc[doc['文档类型名称'].eq(typ),'sid'])
h['any_transfusion_evidence']=h.transfusion_recorded_name|h.has_transfusion_note
for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']:
 assert h['n_days_'+v].between(0,2).all()
for f in ['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose']:h['missing_'+f]=h[f].isna().astype(int)
for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']:h['any_n_days_'+v]=h['n_days_'+v].ge(1).astype(int);h['repeat_n_days_'+v]=h['n_days_'+v].ge(2).astype(int)
e=h[h.eligible].copy();m=e[e.O.eq(1)];assert len(e)==116 and len(m)==100 and m.pid.is_unique;assert m.y.sum()==46;assert e.loc[e.O.eq(0),'y'].isna().all()
# Restore exact source precedence for already matched records is unnecessary for a calendar-day analysis.
features=['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose']
h.to_csv(V/'all_stays.csv.gz',index=False);e.to_csv(V/'eligible116.csv.gz',index=False);l.to_csv(V/'labs_confirmed_units.csv.gz',index=False);h[['sid','pid','raw_sid','raw_pid']].to_csv(V/'private_crosswalk.csv',index=False)
# D3 reuses first-ACS selection BEFORE a new Hb eligibility check, so it is a distinct target population.
d3=[]
for r in h[h.valid&h.first_ACS_in_old_plus_new&h.primary_AMI&~h.suspect&h.primary_count.eq(1)].itertuples():
 if r.discharge_time<=r.D0+pd.Timedelta(days=3):continue
 g=l[l.sid.eq(r.sid)];early=g[g.day.le(2)];eh=early[early.variable.eq('hb')];ev=eh.value_std.dropna()
 if len(ev)==0 or eh.value_std.isna().any() or ev.min()<r.cut:continue
 future=g[g.variable.eq('hb')&g.day.between(3,7)].value_std.dropna()
 if not len(future):continue
 t={'sid':r.sid,'pid':r.pid,'age':r.age,'male':r.male,'cut':r.cut,'y':float(future.min()<r.cut),'source':'LOCAL_NEW','AMI':True}
 for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']:
  d=early[early.variable.eq(v)].groupby('day').value_std.median().dropna().sort_index();t['last_'+v]=d.iloc[-1] if len(d) else np.nan;t['n_days_'+v]=len(d)
 t['last_margin']=t['last_hb']-r.cut
 for f in features:t['missing_'+f]=int(pd.isna(t[f]))
 d3.append(t)
pd.DataFrame(d3).to_csv(V/'D3_local_cohort.csv.gz',index=False)
flow=[{'stage':'Exported stays','n':436},{'stage':'Exported people','n':399},{'stage':'Primary AMI stays','n':320},{'stage':'First visible ACS is AMI (diagnosis-triggered export)','n':306},{'stage':'Strictly beyond D2 discharge-date boundary','n':287},{'stage':'Early normal (diagnosis-triggered export)','n':118},{'stage':'Early normal after linked prior-ACS history correction','n':116},{'stage':'Future Hb observed','n':100},{'stage':'Recorded anemia','n':46},{'stage':'No recorded anemia among observed','n':54},{'stage':'Future Hb unknown','n':16}];pd.DataFrame(flow).to_csv(O/'local_cohort_flow.csv',index=False)
checks={'units_researcher_confirmed':True,'one_eligible_stay_per_person':e.pid.is_unique,'main_count_100':len(m)==100,'events46':m.y.sum()==46,'unobserved_label_unknown':e.loc[e.O.eq(0),'y'].isna().all(),'first_ACS_rule_before_Hb_eligibility':True,'no_local_model_fit':True};(O/'cohort_checks.json').write_text(json.dumps(checks,indent=2,default=bool));assert all(checks.values());print(json.dumps({'eligible':len(e),'observed':len(m),'events':int(m.y.sum()),'D3':len(d3)},ensure_ascii=True))
