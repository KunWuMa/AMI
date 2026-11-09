from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pandas as pd,numpy as np,pickle,json
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score
from scipy.special import logit
import statsmodels.api as sm
P=Path(__file__).resolve().parents[1];V=P/'私有数据';O=P/'结果';S=20261014
D=pd.read_csv(V/'harmonized_cohort.csv.gz');SP=pd.read_csv(V/'MIMIC_patient_split.csv');D=D.set_index('sid',drop=False);tr=SP.loc[SP.split.eq('train'),'sid'];ca=SP.loc[SP.split.eq('calibration'),'sid'];te=SP.loc[SP.split.eq('test'),'sid']
phys=['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose'];counts=['n_days_'+v for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']];miss=['missing_'+f for f in phys]
for f,m in zip(phys,miss):D[m]=D[f].isna().astype(int)
features={'A1_missing_only':phys+miss,'A2_days_only':phys+counts,'A3_M2_without_alb':[f for f in phys if f!='last_alb'],'A4_M3_without_alb_process':phys+[f for f in counts+miss if f not in ['n_days_alb','missing_last_alb']],'A5_GB_without_alb':[f for f in phys if f!='last_alb']}
sets={'MIMIC_test':D.loc[te],'CAD_AMI':D[D.source.eq('CAD')&D.AMI&D.y.notna()],'CAD_all_ACS':D[D.source.eq('CAD')&D.y.notna()]};metrics=[];pred=[];coeff=[]
for name,feats in features.items():
 if name.startswith('A5'):
  chosen=pickle.load((V/'M4_routine_GB.pkl').open('rb'))['model'][-1].get_params();model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingClassifier(**chosen))
 else:model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=3000,random_state=S))
 model.fit(D.loc[tr,feats],D.loc[tr,'y'].astype(int));pcal=model.predict_proba(D.loc[ca,feats])[:,1];cal=LogisticRegression(penalty=None,max_iter=3000).fit(logit(np.clip(pcal,1e-6,1-1e-6)).reshape(-1,1),D.loc[ca,'y'].astype(int))
 for group,c in sets.items():
  raw=model.predict_proba(c[feats])[:,1];pp=cal.predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1];pred.append(c[['sid','y']].assign(cohort=group,model=name,p=pp))
  for ver,p in [('raw',raw),('public_calibrated',pp)]:
   p=np.clip(p,1e-6,1-1e-6);ci=sm.GLM(c.y,np.ones((len(c),1)),offset=logit(p),family=sm.families.Binomial()).fit();cs=sm.GLM(c.y,sm.add_constant(logit(p)),family=sm.families.Binomial()).fit();metrics.append({'model':name,'cohort':group,'version':ver,'n':len(c),'events':int(c.y.sum()),'auc':roc_auc_score(c.y,p),'AP':average_precision_score(c.y,p),'Brier':brier_score_loss(c.y,p),'mean_prediction':p.mean(),'calibration_intercept':float(ci.params.iloc[0]),'calibration_slope':float(cs.params.iloc[1]),'status':'Post-hoc exploratory ablation, not primary model selection'})
 if not name.startswith('A5'):coeff.extend([{'model':name,'feature':f,'beta_per_SD':b} for f,b in zip(feats,model[-1].coef_[0])])
 pickle.dump({'model':model,'calibrator':cal,'features':feats},(V/(name+'.pkl')).open('wb'))
M=pd.DataFrame(metrics);M.to_csv(O/'exploratory_ablation_metrics.csv',index=False);pd.DataFrame(coeff).to_csv(O/'exploratory_ablation_coefficients.csv',index=False);A=pd.concat(pred,ignore_index=True);A.to_csv(V/'exploratory_ablation_predictions.csv.gz',index=False)
orig=pd.read_csv(V/'fixed_external_predictions.csv.gz');orig=orig[orig.model.eq('M2_routine_LR')&orig.cohort.isin(sets)][['sid','cohort','model','y','p']];A=pd.concat([A,orig],ignore_index=True);rng=np.random.default_rng(S);deltas=[]
for group,q in A.groupby('cohort'):
 w=q.pivot(index='sid',columns='model',values='p');y=q.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].astype(int).to_numpy();arr=w.to_numpy();ref=list(w.columns).index('M2_routine_LR');boot=np.full((1000,arr.shape[1],2),np.nan)
 for b in range(1000):
  ix=rng.integers(0,len(y),len(y));yy=y[ix]
  if len(np.unique(yy))<2:continue
  ra=roc_auc_score(yy,arr[ix,ref]);rb=brier_score_loss(yy,arr[ix,ref])
  for j in range(arr.shape[1]):boot[b,j]=[roc_auc_score(yy,arr[ix,j])-ra,brier_score_loss(yy,arr[ix,j])-rb]
 for j,name in enumerate(w.columns):
  if j==ref:continue
  for k,metric in enumerate(['delta_auc','delta_Brier']):lo,hi=np.nanquantile(boot[:,j,k],[.025,.975]);point=(roc_auc_score(y,arr[:,j])-roc_auc_score(y,arr[:,ref])) if k==0 else (brier_score_loss(y,arr[:,j])-brier_score_loss(y,arr[:,ref]));deltas.append({'cohort':group,'model':name,'baseline':'M2_routine_LR','metric':metric,'difference':point,'low':lo,'high':hi,'posthoc':True})
pd.DataFrame(deltas).to_csv(O/'exploratory_ablation_paired_intervals.csv',index=False);(O/'ablations_complete.json').write_text(json.dumps({'complete':True,'posthoc':True,'Chinese_labels_not_used_for_training':True,'seed':S},indent=2),encoding='utf8');print('ABLATIONS_COMPLETE')
