from supplement_common import *
d=load_data();s=splits();sets=standard_sets(d,s['test']);objects={};diagnostics=[]
for name in ['M2_routine_LR','M3_observation_LR','M4_routine_GB']:
 objects[name]=fit_model(name,d,s['train'],s['calibration'],tune=True,imputation='iterative');pickle.dump(objects[name],(PV/('iterative_'+name+'.pkl')).open('wb'));imp=objects[name]['model'][0]
 assert imp.n_iter_>0 and len(imp.imputation_sequence_)>0, 'Iterative imputation did not execute'
 train_raw=d.loc[s['train'],objects[name]['features']].to_numpy(dtype=float);train_filled=imp.transform(d.loc[s['train'],objects[name]['features']]);observed_mask=np.isfinite(train_raw)
 assert np.allclose(train_raw[observed_mask],train_filled[observed_mask]), 'Observed values changed by imputation'
 for group,c in sets.items():
  transformed=imp.transform(c[objects[name]['features']]);diagnostics.append({'model':name,'cohort':group,'finite':bool(np.isfinite(transformed).all()),'iterations':int(imp.n_iter_),'feature_count_preserved':transformed.shape[1]==len(objects[name]['features'])})
 print('IMPUTATION_FIT',name,flush=True)
metrics,pred=evaluate_models(objects,sets,'iterative_imputation');metrics.to_csv(O/'iterative_imputation_metrics.csv',index=False);pred.to_csv(PV/'iterative_imputation_predictions.csv.gz',index=False);ii,_=bootstrap_fixed(pred,baseline='M2_routine_LR',B=1000);ii.to_csv(O/'iterative_imputation_intervals.csv',index=False)
comp=[];rng=np.random.default_rng(SEED+1000)
for name,obj in objects.items():
 original=pickle.load((V/(name+'.pkl')).open('rb'))
 for group,c in sets.items():
  _,p=predict(obj,c);_,ref=predict(original,c);y=c.y.to_numpy();boot=[]
  for b in range(1000):
   ix=rng.integers(0,len(y),len(y));yy=y[ix]
   if len(np.unique(yy))<2:continue
   boot.append([roc_auc_score(yy,p[ix])-roc_auc_score(yy,ref[ix]),np.mean((p[ix]-yy)**2-(ref[ix]-yy)**2)])
  boot=np.asarray(boot)
  for k,f in enumerate(['auc','Brier']):lo,hi=np.quantile(boot[:,k],[.025,.975]);comp.append({'cohort':group,'model':name,'metric':'delta_'+f,'comparison':'Iterative versus median imputation','difference':scores(y,p,calibration=False)[f]-scores(y,ref,calibration=False)[f],'low':lo,'high':hi})
pd.DataFrame(comp).to_csv(O/'iterative_vs_median_imputation.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'iterative_imputation_diagnostics.csv',index=False);assert all(z['finite'] and z['feature_count_preserved'] for z in diagnostics)
complete('15_imputation',{'models':list(objects),'training_only_imputation':True,'iterative_initialization':'median','wide_biological_bounds_on_imputed_values':True,'does_not_establish_MAR_or_correct_MNAR':True})
