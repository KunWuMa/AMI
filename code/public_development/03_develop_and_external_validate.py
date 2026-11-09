from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pandas as pd,numpy as np,json,pickle,warnings,time,sys,hashlib
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve
from sklearn.inspection import permutation_importance
from scipy.special import logit
import statsmodels.api as sm
warnings.filterwarnings('ignore',category=RuntimeWarning)
P=Path(__file__).resolve().parents[1];O=P/'结果';V=P/'私有数据';G=P/'图表';SEED=20261014;start=time.time()
D=pd.read_csv(V/'harmonized_cohort.csv.gz');assert D.pid.is_unique
phys=['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose'];counts=['n_days_'+v for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']];miss=['missing_'+f for f in phys]
for f,m in zip(phys,miss):D[m]=D[f].isna().astype(int)
feature_sets={'M0_Hb':['last_margin'],'M1_clinical':['age','male','last_margin'],'M1D_early_change':['age','male','last_margin','delta_hb'],'M2_routine_LR':phys,'M3_observation_LR':phys+counts+miss,'M4_routine_GB':phys}
assert not set(sum(feature_sets.values(),[]))&{'y','AMI','ICU_any','CABG_any','future_hb_days','future_hb_rows','late_min_hb','year'}
PUB=D[D.source.eq('MIMIC')&D.AMI&D.y.notna()].copy();tr,rest=train_test_split(PUB.index,test_size=.4,random_state=SEED,stratify=PUB.y);ca,te=train_test_split(rest,test_size=.5,random_state=SEED+1,stratify=PUB.loc[rest,'y']);splits={'train':tr,'calibration':ca,'test':te};splitframe=pd.concat([D.loc[ix,['sid','pid']].assign(split=n) for n,ix in splits.items()]);assert not splitframe.pid.duplicated().any();splitframe.to_csv(V/'MIMIC_patient_split.csv',index=False)
sets={'MIMIC_test':D.loc[te],'CAD_AMI':D[D.source.eq('CAD')&D.AMI&D.y.notna()],'CAD_all_ACS':D[D.source.eq('CAD')&D.y.notna()],'CAD_non_AMI':D[D.source.eq('CAD')&~D.AMI&D.y.notna()]}
extra={'MIMIC_test_no_CABG':sets['MIMIC_test'][~sets['MIMIC_test'].CABG_any.fillna(False).astype(bool)],'MIMIC_test_ICU':sets['MIMIC_test'][sets['MIMIC_test'].ICU_any.fillna(False).astype(bool)],'MIMIC_test_no_ICU':sets['MIMIC_test'][~sets['MIMIC_test'].ICU_any.fillna(False).astype(bool)],'MIMIC_test_no_type2_other':sets['MIMIC_test'][~sets['MIMIC_test'].type2_or_other_MI.fillna(False).astype(bool)],'CAD_recent_all_ACS':sets['CAD_all_ACS'][sets['CAD_all_ACS'].year.ge(2024)]}
metrics=[];pred=[];models={};calibrators={};tuning=[];coefs=[]
def metrics_one(y,p):
 y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p),1e-6,1-1e-6);m={'n':len(y),'events':int(y.sum()),'prevalence':y.mean(),'mean_prediction':p.mean(),'auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,'AP':average_precision_score(y,p) if y.sum()>0 else np.nan,'Brier':brier_score_loss(y,p),'null_Brier':float(y.mean()*(1-y.mean()))}
 if len(np.unique(y))==2:
  try:
   ci=sm.GLM(y,np.ones((len(y),1)),offset=logit(p),family=sm.families.Binomial()).fit();cs=sm.GLM(y,sm.add_constant(logit(p)),family=sm.families.Binomial()).fit();m.update({'calibration_intercept':float(ci.params[0]),'calibration_slope':float(cs.params[1])})
  except Exception:m.update({'calibration_intercept':np.nan,'calibration_slope':np.nan})
 return m
for name,feats in feature_sets.items():
 if name.endswith('_GB'):
  base=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,random_state=SEED,early_stopping=False));grid=GridSearchCV(base,{'histgradientboostingclassifier__max_leaf_nodes':[7,15],'histgradientboostingclassifier__l2_regularization':[1,10]},scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=SEED),n_jobs=1,refit=True);grid.fit(D.loc[tr,feats],D.loc[tr,'y'].astype(int));model=grid.best_estimator_
  for i,params in enumerate(grid.cv_results_['params']):tuning.append({'model':name,'parameters':json.dumps(params),'train_CV_mean_auc':grid.cv_results_['mean_test_score'][i],'selected':i==grid.best_index_})
 else:model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,solver='lbfgs',max_iter=3000,random_state=SEED));model.fit(D.loc[tr,feats],D.loc[tr,'y'].astype(int))
 cpred=np.clip(model.predict_proba(D.loc[ca,feats])[:,1],1e-6,1-1e-6);cal=LogisticRegression(penalty=None,max_iter=3000).fit(logit(cpred).reshape(-1,1),D.loc[ca,'y'].astype(int));models[name]=model;calibrators[name]=cal
 for group,c in {**sets,**extra}.items():
  if not len(c):continue
  raw=model.predict_proba(c[feats])[:,1];pc=cal.predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1]
  for version,p in [('raw',raw),('public_calibrated',pc)]:metrics.append({'model':name,'cohort':group,'version':version,**metrics_one(c.y,p)})
 for group,c in sets.items():
  raw=model.predict_proba(c[feats])[:,1];pc=cal.predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1];pred.append(c[['sid','pid','source','AMI','year','ICU_any','CABG_any','early_min_hb','cut','future_hb_days','y','y_daily_median','y_unique_day','y_last_day','y_lt110','y_drop20']].assign(cohort=group,model=name,p_raw=raw,p=pc))
 if not name.endswith('_GB'):
  beta=model[-1].coef_[0];scale=model[1].scale_;coefs.extend([{'model':name,'feature':f,'beta_per_SD':float(b),'beta_per_native_unit':float(b/s),'OR_per_SD':float(np.exp(b))} for f,b,s in zip(feats,beta,scale)])
 pickle.dump({'model':model,'calibrator':cal,'features':feats},(V/(name+'.pkl')).open('wb'))
 print('FIT',name,flush=True);pd.DataFrame(metrics).to_csv(O/'model_metrics.csv',index=False)
PR=pd.concat(pred,ignore_index=True);PR.to_csv(V/'fixed_external_predictions.csv.gz',index=False);pd.DataFrame(tuning).to_csv(O/'training_tuning.csv',index=False);pd.DataFrame(coefs).to_csv(O/'LR_coefficients.csv',index=False)
# Patient bootstrap conditional on the trained model; paired models use the identical resamples.
intervals=[];diffs=[];rng=np.random.default_rng(SEED);B=1000
for group in sets:
 frame=PR[PR.cohort.eq(group)];wide=frame.pivot(index='sid',columns='model',values='p');yy=frame.drop_duplicates('sid').set_index('sid').loc[wide.index,'y'].astype(int).to_numpy();arr=wide.to_numpy();boot=np.full((B,arr.shape[1],3),np.nan)
 for b in range(B):
  ix=rng.integers(0,len(yy),len(yy));yb=yy[ix]
  if len(np.unique(yb))<2:continue
  for j in range(arr.shape[1]):boot[b,j]=[roc_auc_score(yb,arr[ix,j]),average_precision_score(yb,arr[ix,j]),brier_score_loss(yb,arr[ix,j])]
 for j,name in enumerate(wide.columns):
  for k,m in enumerate(['auc','AP','Brier']):lo,hi=np.nanquantile(boot[:,j,k],[.025,.975]);intervals.append({'cohort':group,'model':name,'metric':m,'low':lo,'high':hi,'valid_bootstraps':int(np.isfinite(boot[:,j,k]).sum())})
 baseline=list(wide.columns).index('M1_clinical')
 for j,name in enumerate(wide.columns):
  if j==baseline:continue
  delta=boot[:,j,0]-boot[:,baseline,0];lo,hi=np.nanquantile(delta,[.025,.975]);diffs.append({'cohort':group,'model':name,'baseline':'M1_clinical','delta_auc':roc_auc_score(yy,arr[:,j])-roc_auc_score(yy,arr[:,baseline]),'low':lo,'high':hi})
 print('BOOTSTRAP',group,flush=True)
pd.DataFrame(intervals).to_csv(O/'performance_intervals.csv',index=False);pd.DataFrame(diffs).to_csv(O/'paired_auc_differences.csv',index=False)
# Definition sensitivity: fixed predictions; no model selection or fitting to alternative labels.
sens=[]
for (group,name),q in PR.groupby(['cohort','model']):
 for target in ['y_daily_median','y_unique_day','y_last_day','y_lt110','y_drop20']:
  z=q[q[target].notna()];sens.append({'cohort':group,'model':name,'target':target,**metrics_one(z[target],z.p)})
 z=q[q.early_min_hb.ge(q.cut+10)];
 if len(z):sens.append({'cohort':group,'model':name,'target':'y_normal_margin_ge10',**metrics_one(z.y,z.p)})
pd.DataFrame(sens).to_csv(O/'fixed_prediction_sensitivities.csv',index=False)
# Public-calibration threshold for 80% sensitivity only. Clinical use not established.
thresholds=[];riskrows=[];dc=[]
for name,feats in feature_sets.items():
 raw=models[name].predict_proba(D.loc[ca,feats])[:,1];pp=calibrators[name].predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1];fpr,tpr,th=roc_curve(D.loc[ca,'y'],pp);t=float(th[np.flatnonzero(tpr>=.8)[0]]);thresholds.append({'model':name,'threshold':t,'selection':'MIMIC calibration subset, first ROC threshold reaching sensitivity >=0.8'})
 for group,q in PR[PR.model.eq(name)].groupby('cohort'):
  pos=q.p.ge(t);y=q.y.eq(1);riskrows.append({'model':name,'cohort':group,'threshold':t,'flagged_fraction':pos.mean(),'sensitivity':(pos&y).sum()/y.sum(),'specificity':(~pos&~y).sum()/(~y).sum(),'PPV':(pos&y).sum()/max(pos.sum(),1),'NPV':(~pos&~y).sum()/max((~pos).sum(),1)})
  for pt in [.1,.2,.3,.4,.5,.6]:
   pos=q.p.ge(pt);nb=((pos&y).sum()-(pos&~y).sum()*pt/(1-pt))/len(q);allnb=y.mean()-(1-y.mean())*pt/(1-pt);dc.append({'model':name,'cohort':group,'threshold':pt,'net_benefit':nb,'treat_all':allnb,'treat_none':0})
pd.DataFrame(thresholds).to_csv(O/'calibration_selected_thresholds.csv',index=False);pd.DataFrame(riskrows).to_csv(O/'fixed_threshold_performance.csv',index=False);pd.DataFrame(dc).to_csv(O/'illustrative_decision_curves.csv',index=False)
# Baseline source shift (standardized mean differences are descriptive).
base=[];cov=[]
for group,c in sets.items():
 for f in phys+counts:
  a=c[f];ref=sets['MIMIC_test'][f];sd=np.sqrt((a.var()+ref.var())/2);base.append({'cohort':group,'feature':f,'n_available':int(a.notna().sum()),'missing_fraction':a.isna().mean(),'median':a.median(),'q25':a.quantile(.25),'q75':a.quantile(.75),'SMD_vs_MIMIC_test':(a.mean()-ref.mean())/sd if sd>0 else np.nan})
pd.DataFrame(base).to_csv(O/'cohort_characteristics_and_shift.csv',index=False)
# Future observation: true outcome is unknown. Report crude identification bounds, no imputed endpoint.
obs=[]
for name,c in [('MIMIC_AMI',D[D.source.eq('MIMIC')&D.AMI&D.eligible]),('CAD_all_ACS',D[D.source.eq('CAD')&D.eligible]),('CAD_AMI',D[D.source.eq('CAD')&D.AMI&D.eligible])]:
 n=len(c);unknown=int(c.y.isna().sum());ev=int(c.y.sum());obs.append({'cohort':name,'eligible':n,'observed':int(c.y.notna().sum()),'unknown':unknown,'recorded_events':ev,'logical_event_lower_bound':ev/n,'logical_event_upper_bound':(ev+unknown)/n})
 for status,q in c.groupby(c.y.notna()):
  for f in ['age','male','last_hb','last_cr','n_days_hb']:cov.append({'cohort':name,'future_observed':bool(status),'feature':f,'n':len(q),'median_or_fraction':q[f].mean() if f=='male' else q[f].median()})
pd.DataFrame(obs).to_csv(O/'unknown_outcome_bounds.csv',index=False);pd.DataFrame(cov).to_csv(O/'future_observation_selection.csv',index=False)
# Quantitative explanations and scientific figures.
pi=permutation_importance(models['M4_routine_GB'],D.loc[te,phys],D.loc[te,'y'],scoring='roc_auc',n_repeats=20,random_state=SEED,n_jobs=1);pd.DataFrame({'feature':phys,'test_auc_importance':pi.importances_mean,'SD':pi.importances_std}).to_csv(O/'GB_test_permutation_importance.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,group in zip(axs,['MIMIC_test','CAD_AMI','CAD_all_ACS']):
 for name in ['M1_clinical','M2_routine_LR','M3_observation_LR','M4_routine_GB']:
  q=PR[PR.cohort.eq(group)&PR.model.eq(name)];fpr,tpr,_=roc_curve(q.y,q.p);ax.plot(fpr,tpr,label=name+' '+format(roc_auc_score(q.y,q.p),'.3f'))
 ax.plot([0,1],[0,1],'k--',lw=.7);ax.set(title=group,xlabel='False positive rate',ylabel='True positive rate');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(G/'ROC_external_validation.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4));calpoints=[]
for ax,group in zip(axs,['MIMIC_test','CAD_AMI','CAD_all_ACS']):
 for name in ['M1_clinical','M2_routine_LR','M3_observation_LR','M4_routine_GB']:
  q=PR[PR.cohort.eq(group)&PR.model.eq(name)].copy();q['bin']=pd.qcut(q.p,5,duplicates='drop');a=q.groupby('bin',observed=True).agg(predicted=('p','mean'),observed=('y','mean'),n=('y','size'));ax.plot(a.predicted,a.observed,'o-',label=name);calpoints.extend(a.reset_index(drop=True).assign(model=name,cohort=group).to_dict('records'))
 ax.plot([0,1],[0,1],'k--',lw=.7);ax.set(title=group,xlabel='Predicted probability',ylabel='Observed fraction',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(G/'Calibration_external_validation.png',dpi=200);plt.close(fig);pd.DataFrame(calpoints).to_csv(O/'calibration_plot_points.csv',index=False)
summary={'completed':True,'seed':SEED,'bootstraps':B,'train_n':len(tr),'calibration_n':len(ca),'test_n':len(te),'external_AMI_n':len(sets['CAD_AMI']),'leakage_checks':{'disjoint_patient_splits':True,'only_early_features':True,'Chinese_patients_not_in_training_or_calibration':True,'no_target_imputation':True},'Python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'seconds':round(time.time()-start,1)};(O/'model_experiments_complete.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print('MODEL_COMPLETE',flush=True)
