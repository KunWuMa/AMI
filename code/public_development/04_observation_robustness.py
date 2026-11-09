from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pandas as pd,numpy as np,pickle,json
from sklearn.metrics import roc_auc_score,brier_score_loss
P=Path(__file__).resolve().parents[1];V=P/'私有数据';O=P/'结果';G=P/'图表';SEED=20261014
D=pd.read_csv(V/'harmonized_cohort.csv.gz');PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');FA=pickle.load((V/'future_Hb_arrays.pkl').open('rb'));lookup={a['sid']:a for a in FA};rng=np.random.default_rng(SEED);trials=[]
# Always fixed eligible population, fixed already-fitted predictions, sampled labels only.
for group in ['MIMIC_test','CAD_AMI','CAD_all_ACS','CAD_non_AMI']:
 q=PR[PR.cohort.eq(group)&PR.model.eq('M1_clinical')].copy();assert q.sid.is_unique;arrays=[lookup[s] for s in q.sid];dense=q.y.astype(int).to_numpy();p=q.p.to_numpy();assert all(int(min(min(a) for a in z['day_values'].values())<z['cut'])==y for z,y in zip(arrays,dense))
 for trial in range(200):
  labels={k:[] for k in ['one_per_day','max1_day','max2_days','max3_days']}
  for z in arrays:
   days=sorted(z['day_values']);sampled={d:float(rng.choice(z['day_values'][d])) for d in days};perm=rng.permutation(days);labels['one_per_day'].append(int(min(sampled.values())<z['cut']))
   for k,n in [('max1_day',1),('max2_days',2),('max3_days',3)]:labels[k].append(int(min(sampled[d] for d in perm[:min(n,len(days))])<z['cut']))
  for setting,y in labels.items():
   y=np.asarray(y);assert not ((y==1)&(dense==0)).any();trials.append({'cohort':group,'trial':trial,'setting':setting,'n':len(y),'dense_recorded_events':int(dense.sum()),'sampled_events':int(y.sum()),'event_retention_vs_dense_records':y.sum()/dense.sum(),'auc_fixed_M1':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,'brier_fixed_M1':brier_score_loss(y,p)})
T=pd.DataFrame(trials);T.to_csv(O/'observation_sparsification_trials.csv',index=False);rows=[]
for (g,s),q in T.groupby(['cohort','setting']):
 r={'cohort':g,'setting':s,'n':int(q.n.iloc[0]),'dense_recorded_events':int(q.dense_recorded_events.iloc[0]),'trials':len(q)}
 for f in ['sampled_events','event_retention_vs_dense_records','auc_fixed_M1','brier_fixed_M1']:r[f+'_mean']=q[f].mean();r[f+'_MC_q025']=q[f].quantile(.025);r[f+'_MC_q975']=q[f].quantile(.975)
 rows.append(r)
pd.DataFrame(rows).to_csv(O/'observation_sparsification_summary.csv',index=False)
# Direct recorded endpoint changes and early/future observed-day distributions.
defs=[];freq=[]
for group,c in [('MIMIC_AMI',D[D.source.eq('MIMIC')&D.AMI&D.y.notna()]),('CAD_AMI',D[D.source.eq('CAD')&D.AMI&D.y.notna()]),('CAD_all_ACS',D[D.source.eq('CAD')&D.y.notna()])]:
 for target in ['y','y_daily_median','y_unique_day','y_last_day','y_lt110','y_drop20']:
  z=c[c[target].notna()];defs.append({'cohort':group,'target':target,'observed':len(z),'events':int(z[target].sum()),'event_fraction':z[target].mean(),'unknown_on_original_observed_population':int(c[target].isna().sum())})
 for d,q in c.groupby('future_hb_days'):freq.append({'cohort':group,'future_observed_days':int(d),'n':len(q),'recorded_events':int(q.y.sum()),'recorded_event_fraction':q.y.mean(),'note':'Descriptive, future-day count is not a predictor or causal effect.'})
pd.DataFrame(defs).to_csv(O/'endpoint_definition_counts.csv',index=False);pd.DataFrame(freq).to_csv(O/'endpoint_by_future_observation_days.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(8,4));settings=['one_per_day','max3_days','max2_days','max1_day'];x=np.arange(4)
for offset,g in [(-.15,'MIMIC_test'),(0,'CAD_AMI'),(.15,'CAD_all_ACS')]:
 a=S[S.cohort.eq(g)].set_index('setting').loc[settings];yy=a.event_retention_vs_dense_records_mean.to_numpy();lo=a.event_retention_vs_dense_records_MC_q025.to_numpy();hi=a.event_retention_vs_dense_records_MC_q975.to_numpy();ax.errorbar(x+offset,yy,yerr=[yy-lo,hi-yy],fmt='o-',capsize=3,label=g)
ax.set(xticks=x,xticklabels=settings,ylim=(0,1.05),ylabel='Event retention relative to all recorded Hb');ax.legend();fig.tight_layout();fig.savefig(G/'Observation_event_retention.png',dpi=200);plt.close(fig)
(O/'observation_experiments_complete.json').write_text(json.dumps({'complete':True,'seed':SEED,'trials_per_setting':200,'reference':'All recorded future Hb, NOT clinical gold standard','intervals':'Monte Carlo variation, NOT patient confidence intervals','no_model_retraining':True},indent=2),encoding='utf8');print('OBSERVATION_COMPLETE')
