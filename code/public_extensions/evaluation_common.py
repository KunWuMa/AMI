from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,warnings,hashlib
import pandas as pd,numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[2];S=R.parent.parent;V=S/'私有数据';O=R/'新增实验/结果';PV=R/'新增实验/私有数据';SEED=20261015
O.mkdir(parents=True,exist_ok=True);PV.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(S/'代码'));import supplement_common as sc

def aucs(y,a):
 y=np.asarray(y,int);a=np.asarray(a,float);a=a[:,None] if a.ndim==1 else a;n1=y.sum();n0=len(y)-n1
 if n1==0 or n0==0:return np.full(a.shape[1],np.nan)
 return ((rankdata(a,axis=0)*y[:,None]).sum(axis=0)-n1*(n1+1)/2)/(n1*n0)

def paired_bootstrap(pred,pairs,B=1000,seed=SEED):
 intervals=[];diffs=[];rng=np.random.default_rng(seed)
 for cohort,q in pred.groupby('cohort'):
  w=q.pivot(index='sid',columns='model',values='p');y=q.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].astype(int).to_numpy();a=w.to_numpy();names=list(w.columns);point=np.array([aucs(y,a),np.mean((a-y[:,None])**2,axis=0)])
  assert all(abs(point[0,j]-roc_auc_score(y,a[:,j]))<1e-12 for j in range(len(names)))
  boots=np.full((B,2,len(names)),np.nan)
  for b in range(B):
   ix=rng.integers(0,len(y),len(y));yy=y[ix];boots[b,0]=aucs(yy,a[ix]);boots[b,1]=np.mean((a[ix]-yy[:,None])**2,axis=0)
  for j,m in enumerate(names):
   for k,metric in enumerate(['auc','Brier']):
    low,high=np.nanquantile(boots[:,k,j],[.025,.975]);intervals.append({'cohort':cohort,'model':m,'metric':metric,'point':point[k,j],'low':low,'high':high,'valid_resamples':int(np.isfinite(boots[:,k,j]).sum()),'B':B,'conditional_on_fitted_models':True})
  for model,base in pairs:
   if model not in names or base not in names:continue
   j=names.index(model);z=names.index(base)
   for k,metric in enumerate(['auc','Brier']):
    delta=boots[:,k,j]-boots[:,k,z];lo,hi=np.nanquantile(delta,[.025,.975]);diffs.append({'cohort':cohort,'model':model,'baseline':base,'metric':'delta_'+metric,'difference':point[k,j]-point[k,z],'low':lo,'high':hi,'valid_resamples':int(np.isfinite(delta).sum()),'B':B,'posthoc':True})
 return pd.DataFrame(intervals),pd.DataFrame(diffs)

def done(name,details):
 (O/(name+'_complete.json')).write_text(json.dumps({'complete':True,'posthoc':True,**details},ensure_ascii=False,indent=2,default=str),encoding='utf8');print(name+'_COMPLETE '+json.dumps(details,default=str),flush=True)
