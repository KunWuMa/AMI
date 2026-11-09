from pathlib import Path
import pandas as pd,numpy as np,pickle,json,re,hmac,hashlib,time
from collections import Counter
from functools import lru_cache
P=Path(__file__).resolve().parents[1];R=P.parent;V=P/'私有数据';O=P/'结果';OLD=R/'冠脉包_公开数据联合选题_20261012/基础实验_题目1与3'
manifest=json.loads((R/'双数据包_选题前探索分析/结果/source_manifest.json').read_text(encoding='utf8'));loaded=pickle.load((R/'双数据包_选题前探索分析/私有数据/raw_cache.pkl').open('rb'));assert loaded['fingerprint']==manifest['fingerprint'];raw=loaded['data']['cad'];del loaded
key=(OLD/'私有数据/pseudonym_key.bin').read_bytes()
def norm(x):
 if x is None or (isinstance(x,float) and not np.isfinite(x)):return ''
 return re.sub(r'^(\d+)\.0$',r'\1',str(x).strip())
def uid(x,t):return t+hmac.new(key,(t+norm(x)).encode(),hashlib.sha256).hexdigest()[:16]
@lru_cache(maxsize=300000)
def date(x):return pd.to_datetime(x,errors='coerce')
A=pd.read_pickle(OLD/'私有数据/cad_only_analysis.pkl');A=A[['sid','pid','age','male','AMI','admit_time','discharge_time','year','disposition']].copy();assert A.pid.is_unique;A['age']=A.age.clip(upper=90);A['D0']=A.admit_time.dt.normalize();A['landmark']=A.D0+pd.Timedelta(days=2);A['source']='CAD';A['ICU_any']=np.nan;A['CABG_any']=np.nan;A['type2_or_other_MI']=np.nan
H=pd.DataFrame(raw['病案首页表(0).xlsx']);H['rawsid']=H['住院号'].map(norm);H['rawpid']=H['患者唯一id序号由empi反向更新'].map(norm);H['sid']=H.rawsid.map(lambda x:uid(x,'S'));H=H[H.sid.isin(A.sid)].drop_duplicates(['rawsid','rawpid']);assert H.sid.is_unique;lookup=H.set_index('rawsid')[['rawpid','sid']].to_dict('index');aidx=A.set_index('sid')
items={'hb':['血红蛋白'],'cr':['肌酐','肌酐（氧化酶法）'],'alb':['白蛋白','白蛋白(溴甲酚绿法)'],'wbc':['白细胞'],'platelet':['血小板计数'],'na':['钠'],'k':['钾'],'glucose':['葡萄糖']};mp={n:k for k,ns in items.items() for n in ns};bounds={'hb':(20,250),'cr':(5,5000),'alb':(5,70),'wbc':(.1,200),'platelet':(1,2000),'na':(80,200),'k':(1,12),'glucose':(.5,100)};allowed={'hb':['g/l'],'alb':['g/l'],'cr':['umol/l'],'wbc':['10^9/l'],'platelet':['10^9/l'],'na':['mmol/l'],'k':['mmol/l'],'glucose':['mmol/l']};qc=Counter();rows=[];units=Counter()
for name,rr in raw.items():
 if not name.startswith('检验报告'):continue
 for x in rr:
  rs=norm(x.get('挂号序号/住院号'));var=mp.get(norm(x.get('检验子项目中文名称')).lstrip('*'))
  if rs not in lookup or not var:continue
  z=lookup[rs];sid=z['sid'];c=aidx.loc[sid]
  if norm(x.get('患者唯一id序号由empi反向更新'))!=z['rawpid']:qc['identity_mismatch']+=1;continue
  if re.search('血气|尿|脑脊液|胸水|腹水|透析液',norm(x.get('检验组项目名称'))):qc['nonblood_excluded']+=1;continue
  dt=date(norm(x.get('报告时间')))
  if pd.isna(dt):qc['unknown_date']+=1;continue
  d=dt.normalize();day=(d-c.D0).days
  if day<0 or day>7 or d>c.discharge_time.normalize():qc['out_of_window']+=1;continue
  unit=norm(x.get('单位')).lower().replace('μ','u').replace('µ','u').replace('*','').replace('×','').replace(' ','');units[(var,unit)]+=1
  if unit not in allowed[var]:qc['excluded_unit_'+var]+=1;continue
  text=norm(x.get('检验结果文字描述'));v=float(text) if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',text) else np.nan
  if pd.notna(v) and not bounds[var][0]<=v<=bounds[var][1]:qc['range_invalid_'+var]+=1;v=np.nan
  if pd.isna(v):qc['uncertain_value_'+var]+=1
  rows.append({'sid':sid,'day':day,'feature_day':day,'variable':var,'value_std':v,'raw':text})
L=pd.DataFrame(rows).drop_duplicates(['sid','day','variable','raw']);L.to_pickle(V/'CAD_common_labs.pkl');pd.DataFrame([{'variable':v,'unit':u,'n':n} for (v,u),n in units.items()]).to_csv(O/'CAD_units.csv',index=False)
qc['valid_or_uncertain_rows']=len(L);qc['same_day_multiple_valid_values']=int(L.groupby(['sid','day','variable']).value_std.nunique().gt(1).sum());qc['source_fingerprint_matches']=True;qc['raw_people_selected']=H.rawpid.nunique();(O/'CAD_cleaning_qc.json').write_text(json.dumps(dict(qc),indent=2),encoding='utf8')
A.to_pickle(V/'CAD_index_cohort.pkl');del raw,H,lookup
B=pd.read_pickle(V/'MIMIC_index_cohort.pkl');B=B[pd.to_datetime(B.deathtime,errors='coerce').isna()|pd.to_datetime(B.deathtime,errors='coerce').gt(B.landmark)].copy();B['sid']='M'+B.hadm_id.astype(str);B['pid']='M'+B.subject_id.astype(str);B['source']='MIMIC';B['year']=np.nan
U=pd.read_pickle(V/'MIMIC_common_labs.pkl');U['sid']='M'+U.hadm_id.astype(str);U['feature_day']=(U.available.dt.normalize()-U.D0).dt.days
Uearly=U[U.day.le(1)&U.available.lt(U.landmark)];Ufuture=U[U.day.ge(2)];U=pd.concat([Uearly,Ufuture],ignore_index=True)
vars=list(items)
def build(C,L):
 by={k:g for k,g in L.groupby('sid')};empty=L.iloc[:0];result=[];future_arrays=[]
 for c in C.itertuples():
  g=by.get(c.sid,empty);e=g[g.day.le(1)];f=g[g.day.between(2,7)];eh=e[e.variable.eq('hb')];fh=f[f.variable.eq('hb')];cut=130 if c.male==1 else 120 if c.male==0 else np.nan;valid=eh.value_std.dropna();eligible=bool(len(valid)>0 and eh.value_std.notna().all() and pd.notna(cut) and valid.min()>=cut)
  r={'sid':c.sid,'pid':c.pid,'source':c.source,'age':c.age,'male':c.male,'AMI':bool(c.AMI),'year':c.year,'ICU_any':c.ICU_any,'CABG_any':c.CABG_any,'type2_or_other_MI':c.type2_or_other_MI,'cut':cut,'eligible':eligible,'early_hb_uncertain':bool(eh.value_std.isna().any()),'early_hb_rows':len(eh),'future_hb_days':fh.loc[fh.value_std.notna(),'day'].nunique(),'future_hb_rows':int(fh.value_std.notna().sum())}
  for v in vars:
   q=e[e.variable.eq(v)];d=q.groupby('feature_day').value_std.median().dropna().sort_index();r['n_days_'+v]=len(d);r['last_'+v]=d.iloc[-1] if len(d) else np.nan
   if v=='hb':r['first_hb']=d.iloc[0] if len(d) else np.nan;r['delta_hb']=d.iloc[-1]-d.iloc[0] if len(d) else np.nan
  r['last_margin']=r['last_hb']-cut
  fv=fh.value_std.dropna();dm=fh.groupby('day').value_std.median().dropna();unique=fh.groupby('day').value_std.agg(['nunique','min','count','size']);uv=unique.loc[unique['nunique'].eq(1)&unique['count'].eq(unique['size']),'min']
  r['y']=float(fv.min()<cut) if eligible and len(fv) else np.nan;r['y_daily_median']=float(dm.min()<cut) if eligible and len(dm) else np.nan;r['y_unique_day']=float(uv.min()<cut) if eligible and len(uv) else np.nan;r['y_last_day']=float(dm.iloc[-1]<cut) if eligible and len(dm) else np.nan;r['y_lt110']=float(fv.min()<110) if eligible and len(fv) else np.nan;r['y_drop20']=float(fv.min()<=r['first_hb']-20) if eligible and len(fv) else np.nan;r['late_min_hb']=fv.min() if len(fv) else np.nan;r['early_min_hb']=valid.min() if len(valid) else np.nan
  result.append(r)
  if eligible and len(fv):future_arrays.append({'sid':c.sid,'cut':cut,'day_values':{int(d):q.value_std.dropna().tolist() for d,q in fh.groupby('day') if q.value_std.notna().any()}})
 D=pd.DataFrame(result);assert D.pid.is_unique;return D,future_arrays
DA,FA=build(A,L);DB,FB=build(B,U);D=pd.concat([DB,DA],ignore_index=True);assert D.pid.is_unique;D.to_pickle(V/'harmonized_cohort.pkl');D.to_csv(V/'harmonized_cohort.csv.gz',index=False);pickle.dump(FA+FB,(V/'future_Hb_arrays.pkl').open('wb'))
flow=[]
for name,c in [('MIMIC_all',DB),('MIMIC_AMI',DB[DB.AMI]),('CAD_all_ACS',DA),('CAD_AMI',DA[DA.AMI]),('CAD_non_AMI',DA[~DA.AMI])]:
 flow.append({'cohort':name,'at_D2':len(c),'early_uncertain':int(c.early_hb_uncertain.sum()),'normal_early_eligible':int(c.eligible.sum()),'observed':int(c.y.notna().sum()),'events':int(c.y.sum()),'unknown':int((c.eligible&c.y.isna()).sum()),'early_two_days_in_observed':int((c.y.notna()&c.n_days_hb.ge(2)).sum()),'moderate_lt110':int(c.y_lt110.sum()),'drop20':int(c.y_drop20.sum())})
pd.DataFrame(flow).to_csv(O/'cohort_flow.csv',index=False);(O/'cohort_build_complete.json').write_text(json.dumps({'flows':flow,'features':'Only available D0-D1 results; future variables label/evaluation only.','same_day_policy':'median features, minimum for normality guard and outcome','one_patient_one_index':True},indent=2),encoding='utf8');print('COHORT_COMPLETE',flush=True)
