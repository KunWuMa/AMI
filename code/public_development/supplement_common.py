from pathlib import Path
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_k]='1'
import pandas as pd,numpy as np,pickle,json,warnings,time
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from scipy.special import logit,expit
from scipy.optimize import brentq
import statsmodels.api as sm
P=Path(__file__).resolve().parents[1];V=P/'私有数据';O=P/'结果/补充实验';PV=V/'补充实验';G=P/'图表/补充实验';SEED=20261014
PHYS=['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose'];COUNTS=['n_days_'+v for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']];MISS=['missing_'+f for f in PHYS]
FEATURES={'M0_Hb':['last_margin'],'M1_clinical':['age','male','last_margin'],'M1D_early_change':['age','male','last_margin','delta_hb'],'M2_routine_LR':PHYS,'M3_observation_LR':PHYS+COUNTS+MISS,'M4_routine_GB':PHYS};MAIN=['M1_clinical','M2_routine_LR','M3_observation_LR','M4_routine_GB'];FORBIDDEN={'y','AMI','CABG_any','ICU_any','future_hb_days','future_hb_rows','late_min_hb','year','source','observed','discharge','LOS_days'}
def load_data(path=None):
 d=pd.read_csv(path or V/'harmonized_cohort.csv.gz');assert d.pid.is_unique;d=d.set_index('sid',drop=False)
 for f,m in zip(PHYS,MISS):d[m]=d[f].isna().astype(int)
 return d
def flag(x):return x.astype(str).str.lower().eq('true')
def splits():
 s=pd.read_csv(V/'MIMIC_patient_split.csv');return {k:s.loc[s.split.eq(k),'sid'].tolist() for k in ['train','calibration','test']}
def standard_sets(d,te):return {'MIMIC_test':d.loc[te],'CAD_AMI':d[d.source.eq('CAD')&d.AMI&d.y.notna()],'CAD_all_ACS':d[d.source.eq('CAD')&d.y.notna()]}
def fit_model(name,d,tr,ca,seed=SEED,tune=True,imputation='median',gbparams=None):
 feats=FEATURES[name];assert not set(feats)&FORBIDDEN;assert set(tr).isdisjoint(ca);assert d.loc[tr,'source'].eq('MIMIC').all() and d.loc[ca,'source'].eq('MIMIC').all()
 if imputation=='iterative':
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  # Count/missing flags are fully observed. Hb/age/sex are known in the eligible cohort.
  bounds={'age':(18,90),'male':(0,1),'last_margin':(0,250),'last_cr':(5,5000),'last_alb':(5,70),'last_wbc':(.1,200),'last_platelet':(1,2000),'last_na':(80,200),'last_k':(1,12),'last_glucose':(.5,100)}
  limits=[bounds.get(f,(0,2) if f.startswith('n_days_') else (0,1)) for f in feats]
  imp=IterativeImputer(initial_strategy='median',max_iter=10,tol=.001,random_state=seed,skip_complete=True,keep_empty_features=False,min_value=np.array([b[0] for b in limits]),max_value=np.array([b[1] for b in limits]))
 else:imp=SimpleImputer(strategy='median',keep_empty_features=True)
 x=d.loc[tr,feats];y=d.loc[tr,'y'].astype(int)
 if imputation=='iterative':assert not x.isna().all().any(), 'Iterative sensitivity requires nonempty training columns'
 if name=='M4_routine_GB':
  params=gbparams or {'max_iter':150,'learning_rate':.05,'early_stopping':False,'random_state':seed};params=dict(params);params['random_state']=seed;model=make_pipeline(imp,HistGradientBoostingClassifier(**params))
  if tune:
   grid=GridSearchCV(model,{'histgradientboostingclassifier__max_leaf_nodes':[7,15],'histgradientboostingclassifier__l2_regularization':[1,10]},cv=StratifiedKFold(3,shuffle=True,random_state=seed),scoring='roc_auc',n_jobs=1);grid.fit(x,y);model=grid.best_estimator_;tuning={'parameters':grid.best_params_,'CV_auc':float(grid.best_score_)}
  else:model.fit(x,y);tuning={'parameters':params,'CV_auc':None}
 else:model=make_pipeline(imp,StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=seed));model.fit(x,y);tuning={'parameters':{'C':1},'CV_auc':None}
 pc=model.predict_proba(d.loc[ca,feats])[:,1];cal=LogisticRegression(penalty=None,max_iter=3000).fit(logit(np.clip(pc,1e-6,1-1e-6)).reshape(-1,1),d.loc[ca,'y'].astype(int))
 return {'model':model,'calibrator':cal,'features':feats,'train_ids':list(tr),'calibration_ids':list(ca),'seed':seed,'tuning':tuning,'imputation':imputation}
def predict(obj,c):
 raw=obj['model'].predict_proba(c[obj['features']])[:,1];p=obj['calibrator'].predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1];return raw,p

def scores(y,p,weight=None,calibration=True):
 y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6);w=np.ones(len(y)) if weight is None else np.asarray(weight,dtype=float);m={'n':len(y),'events':int(y.sum()),'prevalence':float(np.average(y,weights=w)),'mean_prediction':float(np.average(p,weights=w)),'auc':float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else np.nan,'AP':float(average_precision_score(y,p,sample_weight=w)) if y.sum()>0 else np.nan,'Brier':float(np.average((p-y)**2,weights=w))}
 if calibration and len(np.unique(y))==2:
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');ci=sm.GLM(y,np.ones((len(y),1)),offset=logit(p),freq_weights=w,family=sm.families.Binomial()).fit();cs=sm.GLM(y,sm.add_constant(logit(p)),freq_weights=w,family=sm.families.Binomial()).fit()
   m.update(calibration_intercept=float(ci.params[0]),calibration_slope=float(cs.params[1]))
  except Exception as e:m.update(calibration_intercept=np.nan,calibration_slope=np.nan,calibration_error=type(e).__name__)
 return m

def evaluate_models(objects,datasets,tag):
 metrics=[];pred=[]
 for name,obj in objects.items():
  for group,c in datasets.items():
   if not len(c):continue
   raw,p=predict(obj,c)
   for version,pp in [('raw',raw),('public_calibrated',p)]:metrics.append({'experiment':tag,'model':name,'cohort':group,'version':version,**scores(c.y,pp)})
   pred.append(c[['sid','pid','source','AMI','y']].assign(model=name,cohort=group,p=p,p_raw=raw,experiment=tag))
 return pd.DataFrame(metrics),pd.concat(pred,ignore_index=True)

def bootstrap_fixed(pred,baseline='M1_clinical',B=1000,seed=SEED,model_column='model'):
 rng=np.random.default_rng(seed);interval=[];paired=[]
 for group,q in pred.groupby('cohort'):
  wide=q.pivot(index='sid',columns=model_column,values='p');y=q.drop_duplicates('sid').set_index('sid').loc[wide.index,'y'].to_numpy();a=wide.to_numpy();boot=np.full((B,len(wide.columns),3),np.nan)
  for b in range(B):
   ix=rng.integers(0,len(y),len(y));yy=y[ix]
   if len(np.unique(yy))<2:continue
   for j in range(a.shape[1]):boot[b,j]=[roc_auc_score(yy,a[ix,j]),average_precision_score(yy,a[ix,j]),np.mean((a[ix,j]-yy)**2)]
  for j,name in enumerate(wide.columns):
   point=scores(y,a[:,j],calibration=False)
   for k,m in enumerate(['auc','AP','Brier']):lo,hi=np.nanquantile(boot[:,j,k],[.025,.975]);interval.append({'cohort':group,'model':name,'metric':m,'point':point[m],'low':lo,'high':hi,'valid_bootstraps':int(np.isfinite(boot[:,j,k]).sum())})
  if baseline in wide.columns:
   r=list(wide.columns).index(baseline)
   for j,name in enumerate(wide.columns):
    if name==baseline:continue
    for k,m in [(0,'auc'),(2,'Brier')]:v=boot[:,j,k]-boot[:,r,k];lo,hi=np.nanquantile(v,[.025,.975]);paired.append({'cohort':group,'model':name,'baseline':baseline,'metric':'delta_'+m,'difference':scores(y,a[:,j],calibration=False)[m]-scores(y,a[:,r],calibration=False)[m],'low':lo,'high':hi})
 return pd.DataFrame(interval),pd.DataFrame(paired)

def complete(stage,details):
 details={'complete':True,'stage':stage,**details};(O/(stage+'_complete.json')).write_text(json.dumps(details,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8');print(stage+'_COMPLETE',flush=True)
