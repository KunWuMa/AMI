from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pandas as pd,numpy as np,json
from scipy.special import logit
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
P=Path(__file__).resolve().parents[1];O=P/'结果';V=P/'私有数据';PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');rng=np.random.default_rng(20261014);rows=[];cal=[]
for group,q in PR.groupby('cohort'):
 w=q.pivot(index='sid',columns='model',values='p');y=q.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy();idx=rng.integers(0,len(y),(1000,len(y)));b=(w.to_numpy()-y[:,None])**2;ref=list(w.columns).index('M1_clinical')
 for j,name in enumerate(w.columns):
  if j==ref:continue
  d=b[:,j]-b[:,ref];boot=d[idx].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);rows.append({'cohort':group,'model':name,'baseline':'M1_clinical','delta_Brier':d.mean(),'low':lo,'high':hi,'conditional_on_fixed_predictions':True})
 for name,c in q.groupby('model'):
  pp=np.clip(c.p.to_numpy(),1e-6,1-1e-6);yy=c.y.to_numpy();ci=sm.GLM(yy,np.ones((len(yy),1)),offset=logit(pp),family=sm.families.Binomial()).fit();cs=sm.GLM(yy,sm.add_constant(logit(pp)),family=sm.families.Binomial()).fit();a=ci.conf_int()[0];s=cs.conf_int()[1];cal.append({'cohort':group,'model':name,'intercept':ci.params[0],'intercept_low':a[0],'intercept_high':a[1],'slope':cs.params[1],'slope_low':s[0],'slope_high':s[1],'interval_type':'Model-based Wald 95%, fixed predictions'})
pd.DataFrame(rows).to_csv(O/'paired_Brier_differences.csv',index=False);pd.DataFrame(cal).to_csv(O/'calibration_intervals.csv',index=False)
D=pd.read_csv(V/'harmonized_cohort.csv.gz');pub=D[D.source.eq('MIMIC')&D.AMI&D.y.notna()];test_ids=set(PR.loc[PR.cohort.eq('MIMIC_test'),'sid']);g=[]
for subset,c in [('MIMIC_AMI_all_observed',pub),('MIMIC_AMI_test',pub[pub.sid.isin(test_ids)])]:
 for flag,q in c.groupby('CABG_any'):g.append({'cohort':subset,'CABG_during_stay':bool(flag),'n':len(q),'events':int(q.y.sum()),'event_fraction':q.y.mean()})
pd.DataFrame(g).to_csv(O/'CABG_endpoint_composition.csv',index=False);# Direct increments beyond physiological laboratory model; future CABG is evaluation-only.
comparisons=[];groups={g:q.copy() for g,q in PR.groupby('cohort')};public=groups['MIMIC_test'];groups['MIMIC_test_no_CABG']=public[~public.CABG_any.astype(str).str.lower().eq('true')]
for group,q in groups.items():
 w=q.pivot(index='sid',columns='model',values='p');y=q.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy();base=w['M2_routine_LR'].to_numpy()
 for name in ['M3_observation_LR','M4_routine_GB']:
  prob=w[name].to_numpy();boot=[]
  for b in range(1000):
   ix=rng.integers(0,len(y),len(y));yy=y[ix]
   if len(np.unique(yy))<2:continue
   boot.append([roc_auc_score(yy,prob[ix])-roc_auc_score(yy,base[ix]),np.mean((prob[ix]-yy)**2-(base[ix]-yy)**2)])
  a=np.asarray(boot)
  for k,metric in enumerate(['delta_auc','delta_Brier']):
   lo,hi=np.quantile(a[:,k],[.025,.975]);point=(roc_auc_score(y,prob)-roc_auc_score(y,base)) if k==0 else np.mean((prob-y)**2-(base-y)**2);comparisons.append({'cohort':group,'model':name,'baseline':'M2_routine_LR','metric':metric,'difference':point,'low':lo,'high':hi,'conditional_on_fixed_predictions':True})
pd.DataFrame(comparisons).to_csv(O/'increment_beyond_routine_labs.csv',index=False);print('ADDITIONAL_STATISTICS_COMPLETE')
