from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pandas as pd,numpy as np,json,time
from collections import Counter
P=Path(__file__).resolve().parents[1];R=P.parent;O=P/'结果';V=P/'私有数据';M=R/'公开数据集/MIMIC-IV/2.2'
C=pd.read_pickle(R/'公开数据集_接入核查_20261014/私有缓存/MIMIC_first_ACS_cohort.pkl');C=C[C.still_at_D2].copy()
items={'hb':[51222],'cr':[50912,52546],'alb':[50862,53085],'wbc':[51300,51301],'platelet':[51265],'na':[50983,52623],'k':[50971,52610],'glucose':[50931,52569]};mp={i:v for v,ids in items.items() for i in ids}
parts=[];total=0;st=time.time();path=M/'labevents.csv.gz';before=path.stat();cols=['labevent_id','specimen_id','subject_id','hadm_id','itemid','charttime','storetime','value','valuenum','valueuom']
for n,x in enumerate(pd.read_csv(path,usecols=cols,chunksize=500000,dtype={'labevent_id':'int64','specimen_id':'int64','subject_id':'int64','hadm_id':'float64','itemid':'int32','charttime':'string','storetime':'string','value':'string','valuenum':'float64','valueuom':'string'}),1):
 total+=len(x);x=x[x.itemid.isin(mp)&x.hadm_id.isin(C.hadm_id)].copy()
 if len(x):parts.append(x)
 if n%10==0:(O/'extract_progress.json').write_text(json.dumps({'rows':total,'matched':sum(len(a) for a in parts),'seconds':round(time.time()-st,1)}),encoding='utf8');print(total,flush=True)
L=pd.concat(parts,ignore_index=True);L['hadm_id']=L.hadm_id.astype('int64');L=L.merge(C[['subject_id','hadm_id','D0','landmark','D7end','discharge','deathtime']],on='hadm_id',suffixes=('_lab','_index'),validate='many_to_one');bad=int(L.subject_id_lab.ne(L.subject_id_index).sum());assert bad==0
L['chart']=pd.to_datetime(L.charttime,errors='coerce');L['store']=pd.to_datetime(L.storetime,errors='coerce');L['available']=L[['chart','store']].max(axis=1);L['day']=(L.chart.dt.normalize()-L.D0).dt.days
L=L[L.chart.ge(L.D0)&L.chart.lt(L.D7end)&L.chart.le(L.discharge)].copy();death=pd.to_datetime(L.deathtime,errors='coerce');L=L[death.isna()|L.chart.le(death)].copy();L['variable']=L.itemid.map(mp)
units=L.groupby(['variable','valueuom'],dropna=False).size().rename('n').reset_index();units.to_csv(O/'MIMIC_units.csv',index=False)
unit=L.valueuom.fillna('').str.lower().str.replace(' ','',regex=False);censored=L.value.fillna('').str.contains(r'[<>≤≥]',regex=True);L['number']=L.valuenum.where(~censored)
allowed={'hb':{'g/dl':10,'g/l':1},'cr':{'mg/dl':88.4,'umol/l':1,'µmol/l':1,'μmol/l':1},'alb':{'g/dl':10,'g/l':1},'wbc':{'k/ul':1,'10^9/l':1},'platelet':{'k/ul':1,'10^9/l':1},'na':{'meq/l':1,'mmol/l':1},'k':{'meq/l':1,'mmol/l':1},'glucose':{'mg/dl':1/18,'mmol/l':1}}
L['factor']=[allowed[v].get(u,np.nan) for v,u in zip(L.variable,unit)];L['value_std']=L.number*L.factor
bounds={'hb':(20,250),'cr':(5,5000),'alb':(5,70),'wbc':(.1,200),'platelet':(1,2000),'na':(80,200),'k':(1,12),'glucose':(.5,100)}
invalid=np.array([not lo<=x<=hi if pd.notna(x) else False for x,(lo,hi) in zip(L.value_std,L.variable.map(bounds))]);L.loc[invalid,'value_std']=np.nan
qc={'rows_scanned':total,'rows_in_window':len(L),'identity_mismatch':bad,'unknown_units':int(L.factor.isna().sum()),'numeric_value_with_placeholder_text':int((L.value.fillna('').eq('___')&L.valuenum.notna()).sum()),'explicitly_censored_values':int(censored.sum()),'range_invalid':int(invalid.sum()),'missing_storetime':int(L.store.isna().sum()),'delayed_early_results':int((L.day.le(1)&L.available.ge(L.landmark)).sum())}
assert L.labevent_id.is_unique;L=L.drop_duplicates('labevent_id');L.to_pickle(V/'MIMIC_common_labs.pkl')
# Verify primary diagnosis metadata and CABG procedure codes using official dictionary.
D=pd.read_csv(M/'diagnoses_icd.csv.gz',dtype={'icd_code':'string'});D=D[D.seq_num.eq(1)&D.hadm_id.isin(C.hadm_id)];assert not D.hadm_id.duplicated().any();C=C.merge(D[['hadm_id','icd_code','icd_version']],on='hadm_id',validate='one_to_one')
Q=pd.read_csv(M/'procedures_icd.csv.gz',dtype={'icd_code':'string'});DI=pd.read_csv(M/'d_icd_procedures.csv.gz',dtype={'icd_code':'string'});DI['code']=DI.icd_code.str.replace('.','',regex=False)
sel=(DI.icd_version.eq(9)&DI.code.str.startswith('361'))|(DI.icd_version.eq(10)&DI.code.str.match(r'^021[0-3]'))
DI[sel].to_csv(O/'CABG_code_dictionary.csv',index=False);keys=set(zip(DI.loc[sel,'icd_version'],DI.loc[sel,'icd_code']));Q=Q[Q.hadm_id.isin(C.hadm_id)].copy();Q['CABG']=np.array([(int(v),str(c)) in keys for v,c in zip(Q.icd_version,Q.icd_code)])
C['CABG_any']=C.hadm_id.isin(Q.loc[Q.CABG,'hadm_id']);C['type2_or_other_MI']=C.icd_code.isin(['I21A1','I21A9']);C['age']=C.age.clip(upper=90);C.to_pickle(V/'MIMIC_index_cohort.pkl')
after=path.stat();qc.update({'gzip_EOF':True,'file_unchanged':before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'seconds':round(time.time()-st,1),'cohort_n':len(C),'CABG_any':int(C.CABG_any.sum()),'type2_or_other_MI':int(C.type2_or_other_MI.sum())});assert qc['file_unchanged'];(O/'extraction_qc.json').write_text(json.dumps(qc,indent=2),encoding='utf8');print('COMPLETE',flush=True)
