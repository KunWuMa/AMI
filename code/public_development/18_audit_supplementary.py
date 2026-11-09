from supplement_common import *
import hashlib,py_compile
checks={};d=load_data();s=splits();original=json.loads((P/'结果/model_audit.json').read_text(encoding='utf8'));checks['primary_predictions_unchanged']=hashlib.sha256((V/'fixed_external_predictions.csv.gz').read_bytes()).hexdigest()==original['predictions_sha256'];source=json.loads((P/'结果/data_audit.json').read_text(encoding='utf8'))
for path,sha in source['derived_input_sha256'].items():checks['primary_input_unchanged_'+path]=hashlib.sha256((P/path).read_bytes()).hexdigest()==sha
stages=['09_preparation','10_nonCABG','11_stability','12_development_bootstrap','13_local_update','14_time_period','15_imputation','16_observation_selection','17_domain_subgroups']
for stage in stages:checks['complete_'+stage]=json.loads((O/(stage+'_complete.json')).read_text(encoding='utf8'))['complete']

def audit_objects(prefix,predfile,frame):
 pred=pd.read_csv(PV/predfile)
 for name,q in pred.groupby('model'):
  obj=pickle.load((PV/(prefix+name+'.pkl')).open('rb'));tr=obj['train_ids'];ca=obj['calibration_ids'];checks[prefix+name+'_train_cal_disjoint']=set(tr).isdisjoint(ca);checks[prefix+name+'_public_fitting_only']=bool(frame.loc[tr+ca,'source'].eq('MIMIC').all());checks[prefix+name+'_only_early_inputs']=not bool(set(obj['features'])&FORBIDDEN)
  for group,sub in q.groupby('cohort'):
   checks[prefix+name+'_'+group+'_test_disjoint']=set(sub.sid).isdisjoint(tr+ca);_,pp=predict(obj,frame.loc[sub.sid]);checks[prefix+name+'_'+group+'_saved_prediction_reproduces']=bool(np.allclose(pp,sub.p,atol=1e-12));checks[prefix+name+'_'+group+'_label_matches']=bool(np.array_equal(frame.loc[sub.sid,'y'].to_numpy(),sub.y.to_numpy()))
  if prefix=='nonCABG_':checks[prefix+name+'_no_CABG_in_fitting']=not flag(frame.loc[tr+ca,'CABG_any']).any()

audit_objects('nonCABG_','nonCABG_predictions.csv.gz',d);audit_objects('iterative_','iterative_imputation_predictions.csv.gz',d)
for tag in ['MIMIC_period','D3_calendar','strict48h']:
 z=d if tag=='MIMIC_period' else load_data(PV/(tag+'_cohort.csv.gz'));audit_objects(tag+'_',tag+'_predictions.csv.gz',z)
 if tag!='MIMIC_period':
  meta=pd.read_csv(PV/(tag+'_cohort.csv.gz'));lm=pd.to_datetime(meta.landmark_new,format='ISO8601');end=pd.to_datetime(meta.end_new,format='ISO8601');dis=pd.to_datetime(meta.discharge_time,format='ISO8601');checks[tag+'_still_in_hospital']=bool(dis.gt(lm).all());checks[tag+'_known_labels_early_normal']=bool(meta.loc[meta.y.notna(),'early_min_hb'].ge(meta.loc[meta.y.notna(),'cut']).all());checks[tag+'_unknown_not_zero']=bool(meta.loc[meta.eligible&meta.late_min_hb.isna(),'y'].isna().all())
  if tag=='strict48h':checks['strict48h_exact_landmark']=bool((lm-pd.to_datetime(meta.admit_time,format='ISO8601')).eq(pd.Timedelta(hours=48)).all());checks['strict48h_120h_horizon']=bool((end-lm).eq(pd.Timedelta(hours=120)).all())
 else:
  meta=pd.read_csv(PV/'MIMIC_timing_metadata.csv.gz').set_index('sid');obj=pickle.load((PV/'MIMIC_period_M1_clinical.pkl').open('rb'));pred=pd.read_csv(PV/'MIMIC_period_predictions.csv.gz');te=pred[pred.cohort.eq('MIMIC_late_period')].sid.unique();checks['period_training_range_upper_le2016']=bool(meta.loc[obj['train_ids']+obj['calibration_ids'],'admit_year_upper'].le(2016).all());checks['period_test_range_lower_ge2017']=bool(meta.loc[te,'admit_year_lower'].ge(2017).all())

# Repeated splits: every public patient has exactly one role per repeat; tuning used training only by construction.
rep=pd.read_csv(PV/'repeated_public_patient_splits.csv.gz');checks['twenty_patient_splits']=rep['repeat'].nunique()==20;checks['repeated_split_patient_roles_unique']=not rep.duplicated(['repeat','sid']).any();checks['repeated_split_public_only']=rep.sid.str.startswith('M').all()
for r in [0,19]:
 pred=pd.read_csv(PV/'repeated_split_local_predictions.csv.gz');pred=pred[pred['repeat'].eq(r)]
 for name,q in pred.groupby('model'):
  obj=pickle.load((PV/('stability_'+str(r)+'_'+name+'.pkl')).open('rb'));_,p=predict(obj,d.loc[q.sid]);checks['repeated_prediction_'+str(r)+'_'+name]=bool(np.allclose(p,q.p,atol=1e-12));checks['repeated_fit_not_local_'+str(r)+'_'+name]=all(x.startswith('M') for x in obj['train_ids']+obj['calibration_ids'])

# Development resampling audit: actual saved draws, saved models and sampled evaluation metrics.
draw=np.load(PV/'development_bootstrap_draws.npz');checks['bootstrap_300_draws']=draw['train'].shape==(300,len(s['train'])) and draw['calibration'].shape==(300,len(s['calibration']));bm=pd.read_csv(O/'development_bootstrap_metrics.csv');bp=pd.read_csv(PV/'development_bootstrap_audit_predictions.csv.gz')
for r in [0,149,299]:
 for name in MAIN:
  obj=pickle.load((PV/('development_boot_'+str(r)+'_'+name+'.pkl')).open('rb'));checks['bootstrap_train_draw_'+str(r)+'_'+name]=obj['train_ids']==np.asarray(s['train'])[draw['train'][r]].tolist();checks['bootstrap_cal_draw_'+str(r)+'_'+name]=obj['calibration_ids']==np.asarray(s['calibration'])[draw['calibration'][r]].tolist()
  for group,q in bp[bp.replicate.eq(r)&bp.model.eq(name)].groupby('cohort'):
   _,p=predict(obj,d.loc[q.sid]);checks['bootstrap_saved_p_'+str(r)+'_'+name+'_'+group]=bool(np.allclose(p,q.p,atol=1e-12));ix=draw[group][r];m=scores(q.y.to_numpy()[ix],p[ix],calibration=False);stored=bm[bm.replicate.eq(r)&bm.model.eq(name)&bm.cohort.eq(group)].iloc[0];checks['bootstrap_scores_'+str(r)+'_'+name+'_'+group]=all(abs(m[f]-stored[f])<1e-10 for f in ['auc','AP','Brier'])

# Crossfit updates: own outcome never in the fit; vectorized parameter/prediction reconstruction.
CF=pd.read_csv(PV/'local_update_crossfit_predictions.csv.gz');FP=pd.read_csv(O/'local_update_fold_parameters.csv');FM=pd.read_csv(PV/'local_update_fold_membership.csv.gz');PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');base=PR[PR.model.isin(MAIN)&PR.cohort.isin(['CAD_AMI','CAD_all_ACS'])].set_index(['cohort','model','sid'])
for (g,r,f),q in FM.groupby(['cohort','repeat','fold']):checks['update_roles_'+g+'_'+str(r)+'_'+str(f)]=set(q.loc[q.role.eq('update_train'),'sid']).isdisjoint(q.loc[q.role.eq('update_test'),'sid'])
merged=CF.merge(FP[['cohort','repeat','fold','base_model','method','a','b']],on=['cohort','repeat','fold','base_model','method'],validate='many_to_one');key=pd.MultiIndex.from_arrays([merged.cohort,merged.base_model,merged.sid],names=['cohort','model','sid']);original_p=base.p.reindex(key).to_numpy();expected=expit(merged.a.to_numpy()+merged.b.to_numpy()*logit(np.clip(original_p,1e-6,1-1e-6)));checks['local_update_predictions_match_fold_parameters']=bool(np.allclose(expected,merged.p,atol=1e-12));checks['local_update_labels_match']=bool(np.array_equal(base.y.reindex(key).to_numpy(),merged.y.to_numpy()));checks['all_local_crossfit_repeat_counts']=CF.groupby(['cohort','base_model','method','sid'])['repeat'].nunique().eq(10).all();testkeys=set(map(tuple,FM[FM.role.eq('update_test')][['cohort','repeat','fold','sid']].to_numpy()));checks['all_update_predictions_from_heldout_roles']=all(tuple(z) in testkeys for z in CF[['cohort','repeat','fold','sid']].to_numpy())
TP=pd.read_csv(PV/'local_temporal_update_predictions.csv.gz');checks['local_temporal_test_is_recent']=d.loc[TP.sid,'year'].ge(2024).all();checks['local_temporal_predictions_finite']=np.isfinite(TP.p).all()
OB=pd.read_csv(PV/'future_observation_propensity.csv.gz');checks['propensity_unknown_labels_remain_unknown']=OB.loc[OB.observed.eq(0),'y'].isna().all();checks['propensity_weights_bounded']=OB.weight_cap10.between(1,10).all() and OB.weight_cap5.between(1,5).all();checks['propensity_finite']=np.isfinite(OB.propensity_observed).all()
for filename in ['nonCABG_predictions.csv.gz','local_update_mean_predictions.csv.gz','iterative_imputation_predictions.csv.gz','development_bootstrap_audit_predictions.csv.gz']:
 q=pd.read_csv(PV/filename);checks['finite_'+filename]=np.isfinite(q.p).all() and q.p.between(0,1).all()
# Verify actual iterative fitting and observed-value preservation after the manuscript-stage correction.
for name in ['M2_routine_LR','M3_observation_LR','M4_routine_GB']:
 obj=pickle.load((PV/('iterative_'+name+'.pkl')).open('rb'));imp=obj['model'][0]
 checks['actual_iterative_rounds_'+name]=imp.n_iter_>0 and len(imp.imputation_sequence_)>0
 raw=d.loc[obj['train_ids'],obj['features']].to_numpy(dtype=float);filled=imp.transform(d.loc[obj['train_ids'],obj['features']]);mask=np.isfinite(raw)
 checks['iterative_observed_values_unchanged_'+name]=np.allclose(raw[mask],filled[mask])

for file in (P/'代码').glob('*.py'):py_compile.compile(str(file),doraise=True)
checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),[k for k,v in checks.items() if not v]
manifest={'passed':True,'check_count':len(checks),'checks':checks,'stages':stages,'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'代码').glob('*.py')},'output_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.glob('*.csv')},'clinical_review_and_ethics':'Author will supply; not falsely marked complete','supplementary_analyses':'Post-hoc exploration, frozen original external validation kept unchanged'};(O/'18_supplementary_audit.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('SUPPLEMENTARY_AUDIT_PASS',len(checks))
