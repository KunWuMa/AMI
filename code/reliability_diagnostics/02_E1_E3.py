from analysis_common import *
b,l,p,t=load()
rows=compare(b[b.selected.eq(1)],b[b.selected.eq(0)],FEATURES,'Local selection','Eligible','Ineligible',BINARY)
save(rows,'E1_cohort_selection_summary.csv')
# Mutually exclusive frozen eligibility reasons; uninterpretable takes precedence.
b['exclusion_reason']=np.select([b.selected.eq(1),b.early_hb_uncertain.fillna(False).astype(bool),b.early_min_hb.isna(),b.early_min_hb.lt(b.cut)],['Eligible','Uninterpretable early Hb','No interpretable early Hb','Early Hb below sex-specific threshold'],default='Other frozen exclusion')
save(b.groupby('exclusion_reason').size().reset_index(name='n'),'E1_exclusion_reasons.csv')
rows=[]
for name,d in [('Local',l),('Public',p)]:rows+=compare(d[d.observed.eq(1)],d[d.observed.eq(0)],FEATURES+['risk_'+n for n in MAIN[:3]+['O3']],name,'Future Hb observed','Future Hb unobserved',BINARY)
save(rows,'E1_future_Hb_observation_groups.csv')
# Verify all reused primary predictions, including O3, against saved frozen predictions.
pr=pd.read_csv(PV/'conditional_predictions.csv.gz');checks=[]
for name,d in [('Public_test',t),('Local100',l[l.observed.eq(1)])]:
 for n in MAIN:
  q=pr[(pr.cohort==name)&(pr.model==n)].set_index('sid').loc[d.index];err=np.max(np.abs(predict(obj(n),d)-q.p));assert err<1e-12
  checks.append({'check':name+' '+n+' frozen probability','maximum_absolute_error':err,'passed':True})
q=pd.read_csv(PV/'two_stage_predictions.csv.gz');q=q[(q.cohort=='Local_eligible116')&(q.task=='O')&(q.model=='M3_observation_LR')].set_index('sid').loc[l.index];err=np.max(np.abs(l.risk_O3-q.p));assert err<1e-12
checks.append({'check':'Local116 O3 frozen probability','maximum_absolute_error':err,'passed':True});save(checks,'frozen_prediction_checks.csv')
m=pd.read_csv(X/'结果汇总/conditional_metrics.csv');m=m[m.cohort.isin(['Public_test','Local100'])&m.model.isin(MAIN)].copy();ci=pd.read_csv(X/'结果汇总/calibration_diagnostic_intervals.csv')
m['p_ref']=m.events/m.n;m['Brier_ref']=m.p_ref*(1-m.p_ref);m['BSS']=1-m.Brier/m.Brier_ref;m['expected_events']=m.mean_prediction*m.n;m['observed_events']=m.events;m['expected_observed_ratio']=m.expected_events/m.events
for metric in ['calibration_intercept','calibration_slope']:
 c=ci[ci.metric.eq(metric)][['cohort','model','low','high','B','valid_fits']].rename(columns={v:metric+'_'+v for v in ['low','high','B','valid_fits']});m=m.merge(c,on=['cohort','model'],how='left')
m['reference_type']='Post-hoc evaluation-sample prevalence; not a deployable model';save(m,'E3_probability_reference_calibration.csv')
q=pd.read_csv(X/'结果汇总/conditional_paired.csv');q=q[(q.cohort=='Local100')&(q.model=='M3_observation_LR')&(q.baseline=='M2_routine_LR')];assert len(q)==2;save(q,'E3_existing_paired_differences.csv')
print('E1 and E3 complete');print(b.groupby('exclusion_reason').size().to_string());print(m[['cohort','model','Brier','BSS','expected_observed_ratio']].to_string(index=False))
