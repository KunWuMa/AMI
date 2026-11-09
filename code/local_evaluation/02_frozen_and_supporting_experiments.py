from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,numpy as np,pandas as pd,warnings,hashlib,time
from scipy.special import logit,expit
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
warnings.filterwarnings('ignore');R=Path(__file__).resolve().parents[1];S=R.parent;V=R/'私有数据_禁止公开';O=R/'结果汇总';OLDV=S/'私有数据';EXT=S/'论文_BMC_MIDM_LaTeX/审稿意见修订_R1_20261015/新增实验/私有数据';SEED=20261022;sys.path.insert(0,str(S/'代码'));import supplement_common as sc
E=pd.read_csv(V/'eligible116.csv.gz');D=sc.load_data();SP=sc.splits();PUB=D.loc[SP['test']].copy();L=E[E.O.eq(1)].copy();sets={'Public_test':PUB,'Local100':L};rows=[];pred=[];objects={}
model_names=['M0_Hb','M1_clinical','M1D_early_change','M2_routine_LR','M3_observation_LR','M4_routine_GB','A1_missing_only','A2_days_only','A3_M2_without_alb','A4_M3_without_alb_process','A5_GB_without_alb','A6_M3_no_albumin_any','B1_M3_binary_presence','B2_M3_repeat_indicators']
def save(df,n):df.to_csv(O/(n+'.csv'),index=False)
def aucs(y,a):
 y=np.asarray(y,int);a=np.asarray(a,float);a=a[:,None] if a.ndim==1 else a;n1=y.sum();n0=len(y)-n1
 return ((rankdata(a,axis=0)*y[:,None]).sum(axis=0)-n1*(n1+1)/2)/(n1*n0) if n1*n0 else np.full(a.shape[1],np.nan)
def paired(pr,B=1000,seed=SEED):
 rng=np.random.default_rng(seed);ints=[];diff=[]
 pairs=[('M1_clinical','M0_Hb'),('M2_routine_LR','M1_clinical'),('M3_observation_LR','M1_clinical'),('M3_observation_LR','M2_routine_LR'),('A4_M3_without_alb_process','M3_observation_LR'),('A6_M3_no_albumin_any','M3_observation_LR'),('B1_M3_binary_presence','M3_observation_LR'),('B2_M3_repeat_indicators','M3_observation_LR')]
 for cohort,g in pr.groupby('cohort'):
  w=g.pivot(index='sid',columns='model',values='p');names=list(w.columns);y=g.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy().astype(int);a=w.to_numpy();point=np.array([aucs(y,a),np.mean((a-y[:,None])**2,axis=0)]);boots=np.full((B,2,len(names)),np.nan)
  for b in range(B):
   ix=rng.integers(0,len(y),len(y));boots[b,0]=aucs(y[ix],a[ix]);boots[b,1]=np.mean((a[ix]-y[ix,None])**2,axis=0)
  for j,name in enumerate(names):
   for k,metric in enumerate(['auc','Brier']):
    v=boots[:,k,j];lo,hi=np.nanquantile(v,[.025,.975]);ints.append({'cohort':cohort,'model':name,'metric':metric,'point':point[k,j],'low':lo,'high':hi,'valid_resamples':int(np.isfinite(v).sum()),'B':B})
  for name,base in pairs:
   if name not in names or base not in names:continue
   j=names.index(name);z=names.index(base)
   for k,metric in enumerate(['auc','Brier']):
    lo,hi=np.nanquantile(boots[:,k,j]-boots[:,k,z],[.025,.975]);diff.append({'cohort':cohort,'model':name,'baseline':base,'metric':'delta_'+metric,'difference':point[k,j]-point[k,z],'low':lo,'high':hi,'B':B})
 return pd.DataFrame(ints),pd.DataFrame(diff)
for name in model_names:
 obj=pickle.load(open((EXT if name.startswith(('A6','B1','B2')) else OLDV)/(name+'.pkl'),'rb'));objects[name]=obj
 for cohort,c0 in sets.items():
  c=c0.copy()
  for f in sc.COUNTS:c['any_'+f]=c[f].ge(1).astype(int);c['repeat_'+f]=c[f].ge(2).astype(int)
  raw,p=sc.predict(obj,c);rows.append({'cohort':cohort,'model':name,**sc.scores(c.y,p)});pred.append(c[['sid','pid','y']].assign(cohort=cohort,model=name,p=p,p_raw=raw))
PR=pd.concat(pred,ignore_index=True);PR.to_csv(V/'conditional_predictions.csv.gz',index=False);save(pd.DataFrame(rows),'conditional_metrics');ii,dd=paired(PR);save(ii,'conditional_intervals');save(dd,'conditional_paired');print('conditional complete',flush=True)

# Public probability thresholds retain original public-only selection.
th=pd.read_csv(S/'结果/calibration_selected_thresholds.csv');thr=[]
for name,g in PR.groupby('model'):
 if name not in th.model.tolist():continue
 t=float(th.loc[th.model.eq(name),'threshold'].iloc[0])
 for cohort,q in g.groupby('cohort'):
  y=q.y.to_numpy().astype(int);p=q.p.to_numpy();pos=p>=t;thr.append({'cohort':cohort,'model':name,'threshold':t,'n':len(q),'flagged_fraction':pos.mean(),'sensitivity':float((pos&(y==1)).sum()/sum(y==1)),'specificity':float((~pos&(y==0)).sum()/sum(y==0)),'PPV':float(y[pos].mean()) if pos.any() else np.nan,'NPV':float((1-y[~pos]).mean()) if (~pos).any() else np.nan})
save(pd.DataFrame(thr),'fixed_public_thresholds')
# Calibration quantile summaries (Wilson intervals) and bootstrap diagnostic slopes/intercepts.
from statsmodels.stats.proportion import proportion_confint
bins=[];cals=[];rng=np.random.default_rng(SEED+1)
for (cohort,name),q in PR[PR.model.isin(['M1_clinical','M2_routine_LR','M3_observation_LR'])].groupby(['cohort','model']):
 q=q.copy();q['bin']=pd.qcut(q.p,5,duplicates='drop')
 for bi,g in q.groupby('bin',observed=True):
  lo,hi=proportion_confint(int(g.y.sum()),len(g),method='wilson');bins.append({'cohort':cohort,'model':name,'mean_prediction':g.p.mean(),'observed':g.y.mean(),'n':len(g),'events':int(g.y.sum()),'low':lo,'high':hi})
 if cohort!='Local100':continue
 y=q.y.to_numpy();z=logit(np.clip(q.p.to_numpy(),1e-6,1-1e-6));vals=[]
 for b in range(1000):
  ix=rng.integers(0,len(y),len(y));yy=y[ix]
  if len(np.unique(yy))<2:continue
  try:
   ci=sm.GLM(yy,np.ones((len(ix),1)),offset=z[ix],family=sm.families.Binomial()).fit();cs=sm.GLM(yy,sm.add_constant(z[ix]),family=sm.families.Binomial()).fit();vals.append([ci.params[0],cs.params[1]])
  except Exception:pass
 for k,metric in enumerate(['calibration_intercept','calibration_slope']):
  v=np.asarray(vals)[:,k];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]);cals.append({'cohort':cohort,'model':name,'metric':metric,'low':lo,'high':hi,'valid_fits':len(v),'B':1000})
save(pd.DataFrame(bins),'calibration_bins');save(pd.DataFrame(cals),'calibration_diagnostic_intervals')
# O and J use separately frozen public observation models, never impute an unobserved anemia label.
sp2=pd.read_csv(EXT/'two_stage_split_ids.csv.gz');ep=D.loc[sp2[sp2.split.eq('test')].sid].copy();stages=[];stpred=[]
for name in ['M1_clinical','M2_routine_LR','M3_observation_LR']:
 ob=pickle.load(open(EXT/('observation_'+name+'.pkl'),'rb'))
 for cohort,c in [('Public_eligible',ep),('Local_eligible116',E)]:
  _,po=sc.predict(ob,c);_,pa=sc.predict(objects[name],c)
  for task,y,p in [('O',c.y.notna().astype(int),po),('J',c.y.fillna(0).astype(int),po*pa)]:
   stages.append({'cohort':cohort,'task':task,'model':name,**sc.scores(y,p)});stpred.append(c[['sid']].assign(cohort=cohort,task=task,model=name,y=y,p=p))
ST=pd.concat(stpred,ignore_index=True);ST.to_csv(V/'two_stage_predictions.csv.gz',index=False);save(pd.DataFrame(stages),'two_stage_metrics')
for task,g in ST.groupby('task'):
 i,d=paired(g);save(i,'two_stage_'+task+'_intervals');save(d,'two_stage_'+task+'_paired')
print('calibration and observation complete',flush=True)
# Baseline shift and logical bounds, with source-specific denominator explicit.
bas=[];cov=[];bounds=[]
for cohort,c in [('Public_all_observed',D[D.source.eq('MIMIC')&D.AMI&D.y.notna()]),*sets.items()]:
 for f in ['age','male','last_hb','last_margin',*sc.PHYS[3:],*sc.COUNTS]:
  x=c[f];bas.append({'cohort':cohort,'feature':f,'n':len(c),'available':int(x.notna().sum()),'missing_fraction':x.isna().mean(),'mean':x.mean(),'median':x.median(),'q25':x.quantile(.25),'q75':x.quantile(.75),'two_days_n':int(x.ge(2).sum()) if f.startswith('n_days_') else np.nan})
for cohort,c in [('Public_eligible',D[D.source.eq('MIMIC')&D.AMI&D.eligible]),('Local_eligible116',E)]:
 n=len(c);unknown=int(c.y.isna().sum());events=int(c.y.sum());bounds.append({'cohort':cohort,'eligible':n,'observed':n-unknown,'unknown':unknown,'events':events,'conditional_fraction':events/(n-unknown),'joint_fraction':events/n,'logical_lower':events/n,'logical_upper':(events+unknown)/n})
 for status,g in c.groupby(c.y.notna()):
  for f in ['age','male','last_hb','last_cr','n_days_hb']:cov.append({'cohort':cohort,'observed':status,'feature':f,'n':len(g),'median_or_fraction':g[f].mean() if f=='male' else g[f].median()})
save(pd.DataFrame(bas),'baseline_characteristics');save(pd.DataFrame(bounds),'observation_bounds');save(pd.DataFrame(cov),'observation_selection')
# Frozen model sensitivities: definitions, windows, source additions, care-name flags.
sens=[];senspred=[]
for name in ['M1_clinical','M2_routine_LR','M3_observation_LR','M4_routine_GB']:
 q=PR[PR.cohort.eq('Local100')&PR.model.eq(name)].set_index('sid').loc[L.sid];p=q.p.to_numpy()
 for target in ['y_daily_median','y_last_day','y_lt110','y_drop20','y_D2D3','y_D2D5']:
  v=L[target].notna();y=L.loc[v,target];pp=p[v];sens.append({'analysis':target,'model':name,**sc.scores(y,pp)});senspred.append(L.loc[v,['sid']].assign(cohort=target,model=name,y=y,p=pp))
 for label,v in [('normal_margin_ge10',L.early_min_hb.ge(L.cut+10)),('exclude_recorded_CABG_name',~L.CABG_recorded_name),('exclude_any_transfusion_evidence',~L.any_transfusion_evidence),('PCI_name_evidence',L.PCI_recorded_name),('recent_2024_2025',L.year.ge(2024)),('male',L.male.eq(1)),('female',L.male.eq(0)),('early_Hb_two_days',L.n_days_hb.eq(2))]:
  y=L.loc[v,'y'];pp=p[v];sens.append({'analysis':label,'model':name,**sc.scores(y,pp)});senspred.append(L.loc[v,['sid']].assign(cohort=label,model=name,y=y,p=pp))
save(pd.DataFrame(sens),'fixed_prediction_sensitivities');senspr=pd.concat(senspred);i,d=paired(senspr);save(i,'sensitivity_intervals');save(d,'sensitivity_paired')
# Use alternative source-fitted objects; they never refit local outcomes.
alt=[];altpred=[]
for prefix,names,data in [('iterative',['M2_routine_LR','M3_observation_LR','M4_routine_GB'],L),('MIMIC_period',sc.MAIN,L),('nonCABG',sc.MAIN,L),('D3_calendar',sc.MAIN,pd.read_csv(V/'D3_local_cohort.csv.gz'))]:
 for name in names:
  obj=pickle.load(open(OLDV/'补充实验'/(prefix+'_'+name+'.pkl'),'rb'));_,p=sc.predict(obj,data);alt.append({'analysis':prefix,'model':name,**sc.scores(data.y,p)});altpred.append(data[['sid','y']].assign(cohort=prefix,model=name,p=p))
save(pd.DataFrame(alt),'alternative_public_models');i,d=paired(pd.concat(altpred));save(i,'alternative_model_intervals');save(d,'alternative_model_paired');print('sensitivity complete',flush=True)
# Recorded care groups: missing evidence is not a clinical negative, and status is retrospective.
path=[]
for name,v in [('Recorded_CABG_name',L.CABG_recorded_name),('Recorded_PCI_name_without_CABG',L.PCI_recorded_name&~L.CABG_recorded_name),('Neither_name_in_available_procedure_table',~L.CABG_recorded_name&~L.PCI_recorded_name),('Transfusion_positive_evidence',L.any_transfusion_evidence),('Operation_note_available',L.has_operation_note)]:
 g=L[v];path.append({'group':name,'n':len(g),'events':int(g.y.sum()),'event_fraction':g.y.mean(),'classification':'available-record name evidence, not independently matched surgical absence'})
save(pd.DataFrame(path),'local_care_pathways')
# Continuous outcomes with original frozen public Ridge models and public training constants.
cont=[];cp=[];ct=[]
for target,yy in [('target_nadir_Hb',L.late_min_hb),('target_decline',L.first_hb-L.late_min_hb),('target_nadir_margin',L.late_min_hb-L.cut)]:
 for name in ['R_constant','M0_Hb','M1_clinical','M2_routine_LR','M3_observation_LR']:
  if name=='R_constant':
   tr=D.loc[SP['train']];v=tr.late_min_hb if target=='target_nadir_Hb' else tr.first_hb-tr.late_min_hb if target=='target_decline' else tr.late_min_hb-tr.cut;p=np.full(len(L),v.median())
  else:obj=pickle.load(open(EXT/(target+'_'+name+'_ridge.pkl'),'rb'));p=obj['model'].predict(L[obj['features']])+obj['public_mean_shift']
  cont.append({'target':target,'model':name,'n':len(L),'MAE':mean_absolute_error(yy,p),'RMSE':np.sqrt(mean_squared_error(yy,p)),'R2':r2_score(yy,p),'mean_prediction':p.mean(),'mean_outcome':yy.mean()});cp.append(L[['sid']].assign(target=target,model=name,y=yy,p=p))
 for name in ['M0_Hb','M1_clinical','M2_routine_LR','M3_observation_LR']:
  p=PR[PR.cohort.eq('Local100')&PR.model.eq(name)].set_index('sid').loc[L.sid,'p'].to_numpy();ct.append({'target':target,'model':name,'Spearman_rho':spearmanr(yy,p).statistic,'n':len(L)})
save(pd.DataFrame(cont),'continuous_metrics');save(pd.DataFrame(ct),'risk_continuous_associations');cp=pd.concat(cp);cp.to_csv(V/'continuous_predictions.csv.gz',index=False);ci=[];cd=[];rng=np.random.default_rng(SEED+4)
for target,g in cp.groupby('target'):
 w=g.pivot(index='sid',columns='model',values='p');y=g.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy();err=w.to_numpy()-y[:,None];a=np.stack([np.abs(err).mean(0),np.sqrt((err**2).mean(0))]);boots=np.empty((1000,2,len(w.columns)))
 for b in range(1000):ix=rng.integers(0,len(y),len(y));e=err[ix];boots[b]=[np.abs(e).mean(0),np.sqrt((e**2).mean(0))]
 names=list(w.columns)
 for j,name in enumerate(names):
  for k,metric in enumerate(['MAE','RMSE']):lo,hi=np.quantile(boots[:,k,j],[.025,.975]);ci.append({'target':target,'model':name,'metric':metric,'point':a[k,j],'low':lo,'high':hi})
 for name,base in [('M3_observation_LR','M2_routine_LR'),('M3_observation_LR','M1_clinical')]:
  j=names.index(name);z=names.index(base)
  for k,metric in enumerate(['MAE','RMSE']):lo,hi=np.quantile(boots[:,k,j]-boots[:,k,z],[.025,.975]);cd.append({'target':target,'model':name,'baseline':base,'metric':metric,'difference':a[k,j]-a[k,z],'low':lo,'high':hi})
save(pd.DataFrame(ci),'continuous_intervals');save(pd.DataFrame(cd),'continuous_paired');print('continuous complete',flush=True)
# O-model weighting is an assumption-dependent sensitivity, not recovery of missing biological outcomes.
weights=[]
for name in ['M1_clinical','M2_routine_LR','M3_observation_LR']:
 ob=pickle.load(open(EXT/'observation_M3_observation_LR.pkl','rb'));_,po=sc.predict(ob,L);p=PR[PR.cohort.eq('Local100')&PR.model.eq(name)].set_index('sid').loc[L.sid,'p'].to_numpy()
 for lower in [.05,.1]:
  w=1/np.clip(po,lower,1);weights.append({'model':name,'propensity_lower_clip':lower,'effective_n':float(w.sum()**2/(w*w).sum()),**sc.scores(L.y,p,weight=w)})
save(pd.DataFrame(weights),'observation_weighted_sensitivity')
# One retained future Hb per calendar day; changes only the recorded endpoint, not early predictors.
labs=pd.read_csv(V/'labs_confirmed_units.csv.gz');f=labs[labs.sid.isin(L.sid)&labs.variable.eq('hb')&labs.day.between(2,7)&labs.value_std.notna()];groups=[(sid,g.value_std.to_numpy()) for (sid,day),g in f.groupby(['sid','day'])];sidix={s:i for i,s in enumerate(L.sid)};draws=[];rng=np.random.default_rng(SEED+5)
for b in range(200):
 mins=np.full(len(L),np.inf)
 for sid,a in groups:mins[sidix[sid]]=min(mins[sidix[sid]],rng.choice(a))
 y=(mins<L.cut.to_numpy()).astype(int)
 for name in ['M1_clinical','M2_routine_LR','M3_observation_LR']:
  p=PR[PR.cohort.eq('Local100')&PR.model.eq(name)].set_index('sid').loc[L.sid,'p'].to_numpy();draws.append({'replicate':b,'model':name,'events':int(y.sum()),'auc':roc_auc_score(y,p),'Brier':np.mean((y-p)**2)})
save(pd.DataFrame(draws).groupby('model').agg(events_mean=('events','mean'),events_min=('events','min'),events_max=('events','max'),auc_median=('auc','median'),auc_q025=('auc',lambda x:x.quantile(.025)),auc_q975=('auc',lambda x:x.quantile(.975)),Brier_median=('Brier','median')).reset_index(),'record_sampling_summary')
# Exploratory repeated cross-fitting for local probability updating: fully separate from frozen evaluation.
upd=[];upr=[]
for rep in range(10):
 cv=StratifiedKFold(5,shuffle=True,random_state=SEED+100+rep)
 for name in ['M1_clinical','M2_routine_LR','M3_observation_LR']:
  base=PR[PR.cohort.eq('Local100')&PR.model.eq(name)].set_index('sid').loc[L.sid,'p'].to_numpy();z=logit(np.clip(base,1e-6,1-1e-6));y=L.y.to_numpy().astype(int);pi=np.empty(len(y));ps=np.empty(len(y))
  for tr,te in cv.split(z,y):
   inter=sm.GLM(y[tr],np.ones((len(tr),1)),offset=z[tr],family=sm.families.Binomial()).fit().params[0];pi[te]=expit(z[te]+inter);mod=LogisticRegression(C=1000,max_iter=3000).fit(z[tr,None],y[tr]);ps[te]=mod.predict_proba(z[te,None])[:,1]
  for label,p in [('intercept',pi),('intercept_slope',ps)]:upd.append({'replicate':rep,'model':name,'update':label,**sc.scores(y,p)});upr.append(L[['sid','pid','y']].assign(replicate=rep,model=name,update=label,p=p))
save(pd.DataFrame(upd),'local_crossfit_updating');pd.concat(upr).to_csv(V/'local_crossfit_predictions.csv.gz',index=False)
# Dataset-origin diagnosis uses held-out folds and is descriptive only.
origin=[];cdat=pd.concat([PUB.assign(is_local=0),L.assign(is_local=1)],ignore_index=True);y=cdat.is_local.to_numpy();cv=StratifiedKFold(5,shuffle=True,random_state=SEED+201)
for label,feats in [('clinical',sc.FEATURES['M1_clinical']),('physiological',sc.PHYS),('observation_only',sc.COUNTS+sc.MISS),('physiology_plus_observation',sc.FEATURES['M3_observation_LR'])]:
 p=np.empty(len(y))
 for tr,te in cv.split(cdat,y):
  model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=3000));model.fit(cdat.iloc[tr][feats],y[tr]);p[te]=model.predict_proba(cdat.iloc[te][feats])[:,1]
 origin.append({'features':label,'endpoint':'dataset_origin_not_anemia',**sc.scores(y,p)})
save(pd.DataFrame(origin),'dataset_origin_diagnostic');print('weighting sampling updating origin complete',flush=True)
# Record executable settings, object hashes, input hashes and successful completion.
import sklearn
hashes=[]
for name,obj in objects.items():
 p=(EXT if name.startswith(('A6','B1','B2')) else OLDV)/(name+'.pkl');hashes.append({'model':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'features':';'.join(obj['features'])})
save(pd.DataFrame(hashes),'frozen_model_manifest')
(R/'experiment_complete.json').write_text(json.dumps({'complete':True,'seed':SEED,'python':sys.version,'sklearn':sklearn.__version__,'pandas':pd.__version__,'local_eligible':len(E),'local_observed':len(L),'local_events':int(L.y.sum()),'bootstrap_fixed':1000,'record_sampling':200,'updating_repeats':10,'no_local_training_of_frozen_models':True,'local_adaptation_separately_labeled':True,'all_principal_and_applicable_local_supporting_analyses_completed':True,'precise48h_local':'not identifiable from date-only admission and report fields'},indent=2));print('ALL ANALYSES COMPLETE',flush=True)
