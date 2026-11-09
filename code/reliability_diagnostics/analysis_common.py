from pathlib import Path
import os,pickle,json
os.environ['LOKY_MAX_CPU_COUNT']='2'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd
from scipy.special import logit,expit
P=Path(os.environ['AMI_STUDY_ROOT']);D=Path(__file__).resolve().parents[2];R=D/'aggregate_results/reliability_diagnostics';V=P/'私有数据';X=P/'AMI新数据扩展实验_20261022';PV=X/'私有数据_禁止公开'
PHYS=['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose'];LABS=['hb','cr','alb','wbc','platelet','na','k','glucose'];COUNTS=['n_days_'+v for v in LABS];MISS=['missing_'+f for f in PHYS];MAIN=['M1_clinical','M2_routine_LR','M3_observation_LR','M4_routine_GB']
def save(d,n):
 pd.DataFrame(d).to_csv(R/n,index=False,encoding='utf-8-sig')
def obj(n):
 path=V/(n+'.pkl') if n!='O3' else P/'论文_BMC_MIDM_LaTeX/审稿意见修订_R1_20261015/新增实验/私有数据/observation_M3_observation_LR.pkl'
 with open(path,'rb') as f:return pickle.load(f)
def predict(o,d):
 raw=o['model'].predict_proba(d[o['features']])[:,1];return o['calibrator'].predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1]
def enrich(d):
 d=d.copy()
 for f in PHYS:d['missing_'+f]=d[f].isna().astype(int)
 for v in LABS:
  d['available_'+v]=d['last_'+v].notna().astype(int);d['two_days_'+v]=d['n_days_'+v].eq(2).astype(int)
 return d.set_index('sid',drop=False)
def load():
 h=enrich(pd.read_csv(PV/'all_stays.csv.gz'));l=enrich(pd.read_csv(PV/'eligible116.csv.gz'))
 p=pd.read_csv(V/'harmonized_cohort.csv.gz');p=enrich(p[p.source.eq('MIMIC')&p.AMI&p.eligible]);s=pd.read_csv(V/'MIMIC_patient_split.csv');t=p.loc[s.loc[s.split.eq('test'),'sid']].copy()
 dates=pd.read_csv(D/'private_cache/public_dates.csv').set_index('sid')
 for d in [p,t]:
  dd=dates.loc[d.index];d['calendar_LOS']=(pd.to_datetime(dd.discharge).dt.normalize()-pd.to_datetime(dd.D0)).dt.days.to_numpy();d['remaining_calendar_days_after_D2']=(d.calendar_LOS-2).clip(lower=0)
 for d in [h,l]:
  d['calendar_LOS']=(pd.to_datetime(d.discharge_time).dt.normalize()-pd.to_datetime(d.D0)).dt.days;d['remaining_calendar_days_after_D2']=(d.calendar_LOS-2).clip(lower=0)
 for d in [p,t,l]:
  d['observed']=d.y.notna().astype(int)
  for n in MAIN[:3]+['O3']:d['risk_'+n]=predict(obj(n),d)
 b=h[h.valid & h.first_ACS_in_old_plus_new & h.primary_AMI & ~h.suspect & h.primary_count.eq(1) & h.over_D2_strict].copy()
 assert len(b)==280 and b.pid.is_unique and len(l)==116 and l.pid.is_unique
 assert len(p)==1934 and p.pid.is_unique and p.y.notna().sum()==1846 and len(t)==370
 assert l.y.notna().sum()==100 and l.y.sum()==46
 b['selected']=b.sid.isin(l.sid).astype(int)
 assert b.selected.sum()==116
 return b,l,p,t

def stats(x,binary=False):
 x=pd.to_numeric(x,errors='coerce');z=x.dropna();n=len(x)
 r={'N':n,'nonmissing':len(z),'missing':int(x.isna().sum()),'mean':z.mean(),'sd':z.std(),'median':z.median(),'q25':z.quantile(.25),'q75':z.quantile(.75)}
 if binary:r.update(positive=int(z.sum()),denominator_observed=len(z),percent_observed=100*z.mean())
 return r

def compare(a,b,features,setting,A,B,binary=()):
 rows=[]
 for f in features:
  x=pd.to_numeric(a[f],errors='coerce').dropna();y=pd.to_numeric(b[f],errors='coerce').dropna();bi=f in binary
  den=np.sqrt((x.mean()*(1-x.mean())+y.mean()*(1-y.mean()))/2) if bi else np.sqrt((x.var()+y.var())/2)
  diff=x.mean()-y.mean();smd=diff/den if den>0 else (0 if diff==0 else np.nan)
  for label,d in [(A,a),(B,b)]:rows.append({'analysis':setting,'group':label,'variable':f,'type':'binary' if bi else 'continuous','SMD_A_minus_B':smd,**stats(d[f],bi)})
 return rows
AV=['available_'+v for v in LABS];TWO=['two_days_'+v for v in LABS];BINARY=['male']+AV+TWO+MISS
FEATURES=PHYS+['last_hb','early_min_hb','calendar_LOS']+AV+COUNTS+TWO+MISS
