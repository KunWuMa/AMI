from evaluation_common import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import spearmanr
D=sc.load_data();SP=sc.splits();D['target_nadir_Hb']=D.late_min_hb;D['target_decline']=D.first_hb-D.late_min_hb;D['target_nadir_margin']=D.late_min_hb-D.cut;sets=sc.standard_sets(D,SP['test']);targets=['target_nadir_Hb','target_decline','target_nadir_margin'];models=['M0_Hb','M1_clinical','M2_routine_LR','M3_observation_LR'];rows=[];pred=[];calparams=[]
for target in targets:
 tr=D.loc[SP['train']];ca=D.loc[SP['calibration']];assert tr[target].notna().all() and ca[target].notna().all()
 for name in ['R_constant',*models]:
  if name=='R_constant':model=None;shift=0;constant=tr[target].median()
  else:
   f=sc.FEATURES[name];model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=1));model.fit(tr[f],tr[target]);shift=float(np.mean(ca[target]-model.predict(ca[f])));pickle.dump({'model':model,'features':f,'public_mean_shift':shift,'target':target},(PV/(target+'_'+name+'_ridge.pkl')).open('wb'))
  calparams.append({'target':target,'model':name,'calibration_intercept_shift':shift,'training_constant':constant if model is None else np.nan,'alpha':1 if model is not None else np.nan})
  for cohort,c in sets.items():
   p=np.full(len(c),constant) if model is None else model.predict(c[f])+shift;y=c[target].to_numpy();rows.append({'target':target,'cohort':cohort,'model':name,'n':len(c),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p),'mean_prediction':p.mean(),'mean_outcome':y.mean(),'prediction_spearman':spearmanr(y,p).statistic if np.std(p)>0 else np.nan});pred.append(c[['sid']].assign(target=target,cohort=cohort,model=name,y=y,p=p))
pd.DataFrame(rows).to_csv(O/'continuous_regression_metrics.csv',index=False);pd.DataFrame(calparams).to_csv(O/'continuous_public_calibration.csv',index=False);PR=pd.concat(pred,ignore_index=True);PR.to_csv(PV/'continuous_regression_predictions.csv.gz',index=False)
interval=[];pair=[];rng=np.random.default_rng(SEED+41)
for (target,cohort),q in PR.groupby(['target','cohort']):
 w=q.pivot(index='sid',columns='model',values='p');y=q.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy();a=w.to_numpy();err=a-y[:,None];boot=np.empty((1000,2,len(w.columns)))
 for b in range(1000):
  ix=rng.integers(0,len(y),len(y));e=err[ix];boot[b]=[np.abs(e).mean(axis=0),np.sqrt((e**2).mean(axis=0))]
 point=[np.abs(err).mean(axis=0),np.sqrt((err**2).mean(axis=0))];names=list(w.columns)
 for j,m in enumerate(names):
  for k,metric in enumerate(['MAE','RMSE']):lo,hi=np.quantile(boot[:,k,j],[.025,.975]);interval.append({'target':target,'cohort':cohort,'model':m,'metric':metric,'point':point[k][j],'low':lo,'high':hi})
 for m,base in [('M1_clinical','M0_Hb'),('M2_routine_LR','M1_clinical'),('M3_observation_LR','M1_clinical'),('M3_observation_LR','M2_routine_LR')]:
  j=names.index(m);k0=names.index(base)
  for k,metric in enumerate(['MAE','RMSE']):lo,hi=np.quantile(boot[:,k,j]-boot[:,k,k0],[.025,.975]);pair.append({'target':target,'cohort':cohort,'model':m,'baseline':base,'metric':'delta_'+metric,'difference':point[k][j]-point[k][k0],'low':lo,'high':hi})
pd.DataFrame(interval).to_csv(O/'continuous_regression_intervals.csv',index=False);pd.DataFrame(pair).to_csv(O/'continuous_regression_paired.csv',index=False)
# Fixed binary risk versus continuous severity, with no binary-model refitting.
fixed=pd.read_csv(V/'fixed_external_predictions.csv.gz');corr=[]
for (cohort,name),q in fixed[fixed.cohort.isin(sets)&fixed.model.isin(models)].groupby(['cohort','model']):
 for target in targets:
  y=D.loc[q.sid,target].to_numpy();p=q.p.to_numpy();rho=spearmanr(y,p).statistic;v=[]
  for b in range(1000):
   ix=rng.integers(0,len(y),len(y));v.append(spearmanr(y[ix],p[ix]).statistic)
  lo,hi=np.nanquantile(v,[.025,.975]);corr.append({'cohort':cohort,'model':name,'target':target,'n':len(q),'Spearman_rho':rho,'low':lo,'high':hi})
pd.DataFrame(corr).to_csv(O/'fixed_risk_continuous_associations.csv',index=False)
summary=[]
for cohort,c in sets.items():
 for target in targets:
  v=c[target];summary.append({'cohort':cohort,'target':target,'n':len(v),'median':v.median(),'q25':v.quantile(.25),'q75':v.quantile(.75),'min':v.min(),'max':v.max()})
pd.DataFrame(summary).to_csv(O/'continuous_target_distributions.csv',index=False)
done('04_continuous',{'ridge_alpha':1,'public_mean_correction_on_original_calibration_subset':True,'local_outcomes_not_used_to_fit_regression':True,'targets':targets,'negative_declines_preserved':True,'no_unobserved_Hb_imputed':True})
