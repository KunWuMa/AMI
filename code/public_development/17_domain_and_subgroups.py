from supplement_common import *
# Dataset origin is a diagnostic outcome, not a patient health outcome.
d=load_data();c=d[d.AMI&d.eligible].copy();origin=c.source.eq('CAD').astype(int).to_numpy();spec={'clinical':FEATURES['M1_clinical'],'physiological':PHYS,'observation_only':COUNTS+MISS,'physiology_plus_observation':FEATURES['M3_observation_LR']};pred=[];rows=[];membership=[];cv=StratifiedKFold(5,shuffle=True,random_state=SEED+1200)
for name,feats in spec.items():
 p=np.zeros(len(c))
 for fold,(tr,te) in enumerate(cv.split(c,origin)):
  m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=3000));m.fit(c.iloc[tr][feats],origin[tr]);p[te]=m.predict_proba(c.iloc[te][feats])[:,1]
  if name=='clinical':membership.extend({'fold':fold,'role':role,'sid':sid} for role,arr in [('train',c.iloc[tr].sid),('test',c.iloc[te].sid)] for sid in arr)
 rows.append({'features':name,'n':len(c),'local_n':int(origin.sum()),'source_AUC':roc_auc_score(origin,p),'source_AP':average_precision_score(origin,p),'interpretation':'Crossfit identification of dataset origin, NOT clinical prediction or causal mechanism'});pred.append(c[['sid','source']].assign(features=name,p_local_source=p))
pd.DataFrame(rows).to_csv(O/'source_classifier_metrics.csv',index=False);pd.concat(pred,ignore_index=True).to_csv(PV/'source_classifier_predictions.csv.gz',index=False);pd.DataFrame(membership).to_csv(PV/'source_classifier_folds.csv.gz',index=False)
PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');PR=PR[PR.model.isin(MAIN)&PR.cohort.isin(['MIMIC_test','CAD_AMI','CAD_all_ACS'])];rows=[];intervals=[];rng=np.random.default_rng(SEED+1201)
for group,qq in PR.groupby('cohort'):
 ids=qq.drop_duplicates('sid').sid;meta=d.loc[ids];strata={'male':meta.male.eq(1),'female':meta.male.eq(0),'age_lt65':meta.age.lt(65),'age_ge65':meta.age.ge(65),'early_margin_lt10':(meta.early_min_hb-meta.cut).lt(10),'early_margin_ge10':(meta.early_min_hb-meta.cut).ge(10)}
 if group.startswith('CAD'):strata.update(before2024=meta.year.lt(2024),year2024plus=meta.year.ge(2024))
 for stratum,mask in strata.items():
  subset=set(meta.loc[mask,'sid']);q=qq[qq.sid.isin(subset)]
  if not len(q):continue
  w=q.pivot(index='sid',columns='model',values='p');y=q.drop_duplicates('sid').set_index('sid').loc[w.index,'y'].to_numpy();a=w.to_numpy();stable=y.sum()>=5 and len(y)-y.sum()>=5
  for j,name in enumerate(w.columns):rows.append({'cohort':group,'stratum':stratum,'model':name,'minimum_support_met':bool(stable),**scores(y,a[:,j])})
  if not stable:continue
  boot=np.full((500,a.shape[1],2),np.nan)
  for b in range(500):
   ix=rng.integers(0,len(y),len(y));yy=y[ix]
   if len(np.unique(yy))<2:continue
   for j in range(a.shape[1]):boot[b,j]=[roc_auc_score(yy,a[ix,j]),np.mean((a[ix,j]-yy)**2)]
  for j,name in enumerate(w.columns):
   for k,f in enumerate(['auc','Brier']):lo,hi=np.nanquantile(boot[:,j,k],[.025,.975]);intervals.append({'cohort':group,'stratum':stratum,'model':name,'metric':f,'low':lo,'high':hi,'patient_bootstraps':500,'no_interaction_test_or_multiple_testing_correction':True})
pd.DataFrame(rows).to_csv(O/'subgroup_metrics.csv',index=False);pd.DataFrame(intervals).to_csv(O/'subgroup_conditional_intervals.csv',index=False)
complete('17_domain_subgroups',{'source_classifier_is_not_clinical_model':True,'clinical_models_frozen':True,'source_classifier_cohort_uses_early_normal_regardless_of_future_Hb':True,'subgroup_small_support_flagged':True,'subgroup_multiplicity_not_corrected':True})
