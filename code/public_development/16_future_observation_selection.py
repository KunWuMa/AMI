from supplement_common import *
d=load_data();all_observation=[];folds=[];diag=[];rng=np.random.default_rng(SEED+1100)
for source,c in [('MIMIC',d[d.source.eq('MIMIC')&d.AMI&d.eligible]),('CAD',d[d.source.eq('CAD')&d.eligible])]:
 feats=PHYS+COUNTS;r=c.y.notna().astype(int).to_numpy();prop=np.zeros(len(c));cv=StratifiedKFold(5,shuffle=True,random_state=SEED+1100)
 for fold,(itr,ite) in enumerate(cv.split(c,r)):
  model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=3000));model.fit(c.iloc[itr][feats],r[itr]);prop[ite]=model.predict_proba(c.iloc[ite][feats])[:,1];assert set(c.iloc[itr].sid).isdisjoint(c.iloc[ite].sid);folds.extend({'source':source,'fold':fold,'role':role,'sid':sid} for role,ids in [('train',c.iloc[itr].sid),('test',c.iloc[ite].sid)] for sid in ids)
 z=c[['sid','source','AMI','y']].copy();z['observed']=r;z['propensity_observed']=prop;z['weight_cap10']=np.minimum(10,1/np.clip(prop,.05,1));z['weight_cap5']=np.minimum(5,1/np.clip(prop,.05,1));all_observation.append(z);diag.append({'source':source,'eligible_n':len(c),'observed_n':int(r.sum()),'R_OOF_AUC':roc_auc_score(r,prop),'R_Brier':brier_score_loss(r,prop),'propensity_min':prop.min(),'propensity_q05':np.quantile(prop,.05),'propensity_median':np.median(prop),'propensity_max':prop.max(),'note':'Propensity relates to future observation, not the clinical anemia outcome.'})
OB=pd.concat(all_observation,ignore_index=True).set_index('sid');OB.to_csv(PV/'future_observation_propensity.csv.gz');pd.DataFrame(folds).to_csv(PV/'observation_propensity_folds.csv.gz',index=False);pd.DataFrame(diag).to_csv(O/'observation_propensity_diagnostics.csv',index=False);PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');PR=PR[PR.model.isin(MAIN)&PR.cohort.isin(['MIMIC_test','CAD_AMI','CAD_all_ACS'])];rows=[];interval=[];pair=[]
for group,qq in PR.groupby('cohort'):
 w=qq.pivot(index='sid',columns='model',values='p');y=qq.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy();arr=w.to_numpy();base=list(w.columns).index('M2_routine_LR');ww=OB.loc[w.index,'weight_cap10'].to_numpy();ess=ww.sum()**2/np.sum(ww**2)
 for name,q in qq.groupby('model'):
  for cap in [5,10]:weights=OB.loc[q.sid,'weight_cap'+str(cap)].to_numpy();rows.append({'cohort':group,'model':name,'weight_cap':cap,'effective_sample_size':weights.sum()**2/np.sum(weights**2),'maximum_weight':weights.max(),'assumption':'Conditional MAR + positivity, not testable with these records',**scores(q.y,q.p,weight=weights)})
 boot=np.full((500,len(w.columns),3),np.nan)
 for b in range(500):
  ix=rng.integers(0,len(y),len(y));yy=y[ix]
  if len(np.unique(yy))<2:continue
  for j in range(arr.shape[1]):boot[b,j]=[roc_auc_score(yy,arr[ix,j],sample_weight=ww[ix]),average_precision_score(yy,arr[ix,j],sample_weight=ww[ix]),np.average((arr[ix,j]-yy)**2,weights=ww[ix])]
 for j,name in enumerate(w.columns):
  point=scores(y,arr[:,j],weight=ww,calibration=False)
  for k,f in enumerate(['auc','AP','Brier']):lo,hi=np.nanquantile(boot[:,j,k],[.025,.975]);interval.append({'cohort':group,'model':name,'metric':f,'point':point[f],'low':lo,'high':hi,'replicates':500,'fixed_propensity_and_prediction':True})
  if j!=base:
   for k,f in [(0,'auc'),(2,'Brier')]:delta=boot[:,j,k]-boot[:,base,k];lo,hi=np.nanquantile(delta,[.025,.975]);pair.append({'cohort':group,'model':name,'baseline':'M2_routine_LR','metric':'delta_'+f,'low':lo,'high':hi})
pd.DataFrame(rows).to_csv(O/'IPW_model_metrics.csv',index=False);pd.DataFrame(interval).to_csv(O/'IPW_conditional_intervals.csv',index=False);pd.DataFrame(pair).to_csv(O/'IPW_paired_intervals.csv',index=False);assert not OB.loc[OB.observed.eq(0),'y'].notna().any();assert np.isfinite(OB.propensity_observed).all()
complete('16_observation_selection',{'propensity_folds':5,'weight_caps':[5,10],'bootstrap_evaluation_replicates':500,'clinical_unknown_labels_not_imputed':True,'weights_diagnostic_not_proof_of_missingness_correction':True,'AMI_local_propensity_borrowed_from_crossfit_all_ACS':True})
