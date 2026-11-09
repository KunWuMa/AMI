from evaluation_common import *
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.special import expit,logit
checks={}
def check(name,condition):
 condition=bool(condition);checks[name]=condition
 if not condition:raise AssertionError(name)
D=sc.load_data();sp=sc.splits();stage=pd.read_csv(PV/'two_stage_split_ids.csv.gz');check('one_patient_one_original_index',D.pid.is_unique);check('observation_stage_sids_unique',stage.sid.is_unique)
for name,ids in sp.items():check('original_'+name+'_allocation_preserved',set(ids)<=set(stage[stage.split.eq(name)].sid))
check('all_1934_public_eligible_in_observation_stage',len(stage)==1934 and set(stage.sid)==set(D[D.source.eq('MIMIC')&D.AMI&D.eligible].sid));check('no_local_observation_training_ids',D.loc[stage.sid,'source'].eq('MIMIC').all())
train=stage[stage.split.eq('train')].sid.tolist();cal=stage[stage.split.eq('calibration')].sid.tolist();te=stage[stage.split.eq('test')].sid.tolist();check('stage_disjoint_training_calibration_test',not(set(train)&set(cal) or set(train)&set(te) or set(cal)&set(te)))
for f,sha in json.loads((O/'original_frozen_hashes.json').read_text(encoding='utf8')).items():check('original_unchanged_'+f,hashlib.sha256((V/f).read_bytes()).hexdigest()==sha)
PR=pd.read_csv(PV/'two_stage_predictions.csv.gz')
for model in ['M1_clinical','M2_routine_LR','M3_observation_LR']:
 obj=pickle.load((PV/('observation_'+model+'.pkl')).open('rb'));check('O_'+model+'_fit_ids_public_only',set(obj['train_ids'])==set(train) and set(obj['calibration_ids'])==set(cal));check('O_'+model+'_no_future_features',not set(obj['features'])&sc.FORBIDDEN);am=pickle.load((V/(model+'.pkl')).open('rb'))
 for cohort in PR.cohort.unique():
  q=PR[(PR.model==model)&(PR.cohort==cohort)];o=q[q.task=='O_future_Hb_observed'].set_index('sid');j=q[q.task=='J_joint_recorded_event'].set_index('sid');c=D.loc[o.index];_,po=sc.predict(obj,c);_,pa=sc.predict(am,c)
  check(model+'_'+cohort+'_O_equals_available_future_Hb',np.array_equal(o.y.astype(int),c.future_hb_days.ge(1).astype(int)));check(model+'_'+cohort+'_J_values_are_record_endpoint_only',np.array_equal(j.loc[o.index].y.astype(int),c.y.fillna(0).astype(int)));check(model+'_'+cohort+'_O_probabilities_reproduced',np.max(np.abs(o.p-po))<1e-12);check(model+'_'+cohort+'_J_factorization_reproduced',np.max(np.abs(j.loc[o.index].p-po*pa))<1e-12);check(model+'_'+cohort+'_unknown_A_left_unknown',c.loc[o.y.eq(0),'y'].isna().all())
original_features=set(sc.FEATURES['M3_observation_LR']);a4=pickle.load((V/'A4_M3_without_alb_process.pkl').open('rb'));check('A4_keeps_albumin_value','last_alb' in a4['features']);check('A4_removes_only_two_albumin_process_inputs',set(a4['features'])==original_features-{'n_days_alb','missing_last_alb'})
for name in ['A6_M3_no_albumin_any','B1_M3_binary_presence','B2_M3_repeat_indicators']:
 obj=pickle.load((PV/(name+'.pkl')).open('rb'));check(name+'_same_original_public_fit_ids',set(obj['train_ids'])==set(sp['train']) and set(obj['calibration_ids'])==set(sp['calibration']));check(name+'_no_future_inputs',not set(obj['features'])&sc.FORBIDDEN)
 if name.startswith('A6'):check('A6_removes_all_three_albumin_fields',set(obj['features'])==original_features-{'last_alb','n_days_alb','missing_last_alb'})
 else:check(name+'_no_explicit_missingness_flags',not any(x.startswith('missing_') for x in obj['features']))
meta=json.loads((O/'procedure_audit.json').read_text(encoding='utf8'));check('local_CABG_not_imputed_absent',D[D.source.eq('CAD')].CABG_any.isna().all());check('no_unverified_supplement_record_links',meta['supplement_exact_id_overlap_with_index_admissions']==0);check('public_CABG_flags_reproduced',meta['public_CABG_flags_reproduced'])
comp=pd.read_csv(O/'public_pathway_composition.csv')
for pop,c in comp.groupby('population'):check(pop+'_pathway_partition_sums',c[c.pathway.ne('all')].n.sum()==c[c.pathway.eq('all')].n.iloc[0])
check('recorded_public_PCI_counts_match_manuscript',int(comp[(comp.population=='all_observed')&comp.pathway.eq('PCI_without_CABG')].n.iloc[0])==1024)
RP=pd.read_csv(PV/'continuous_regression_predictions.csv.gz');RM=pd.read_csv(O/'continuous_regression_metrics.csv');d=D.copy();d['target_nadir_Hb']=d.late_min_hb;d['target_decline']=d.first_hb-d.late_min_hb;d['target_nadir_margin']=d.late_min_hb-d.cut
for (target,cohort,model),c in RP.groupby(['target','cohort','model']):
 check(target+cohort+model+'_target_exact',np.max(np.abs(c.y.to_numpy()-d.loc[c.sid,target].to_numpy()))<1e-10)
 met=RM[(RM.target==target)&(RM.cohort==cohort)&(RM.model==model)].iloc[0];check(target+cohort+model+'_MAE_recomputed',abs(mean_absolute_error(c.y,c.p)-met.MAE)<1e-10);check(target+cohort+model+'_RMSE_recomputed',abs(np.sqrt(mean_squared_error(c.y,c.p))-met.RMSE)<1e-10)
 if model!='R_constant':
  obj=pickle.load((PV/(target+'_'+model+'_ridge.pkl')).open('rb'));tr=d.loc[sp['train']];ca=d.loc[sp['calibration']];check(target+model+'_training_only_imputation_statistics',np.max(np.abs(obj['model'][0].statistics_-np.nanmedian(tr[obj['features']].to_numpy(),axis=0)))<1e-10);shift=np.mean(ca[target]-obj['model'].predict(ca[obj['features']]));check(target+model+'_public_calibration_shift',abs(shift-obj['public_mean_shift'])<1e-10)
check('negative_recorded_declines_retained',int(RP[(RP.target=='target_decline')&(RP.cohort=='CAD_AMI')&(RP.model=='M1_clinical')].y.lt(0).sum())>0)
check('unfavorable_regression_R2_reported',RM[(RM.cohort=='CAD_AMI')&(RM.model=='M3_observation_LR')].R2.lt(0).all())
(O/'R1_computational_audit.json').write_text(json.dumps({'complete':True,'checks_passed':sum(checks.values()),'checks':checks,'no_new_clinical_adjudication':True,'originals_unchanged':True},indent=2),encoding='utf8');print('AUDIT_PASS',sum(checks.values()))
