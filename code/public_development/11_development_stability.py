from supplement_common import *
d=load_data();pub=d[d.source.eq('MIMIC')&d.AMI&d.y.notna()];rows=[];tuning=[];splitrows=[];localpred=[];st=time.time()
for rep in range(20):
 seed=SEED+100+rep;tr,rest=train_test_split(pub.index.to_numpy(),test_size=.4,stratify=pub.y,random_state=seed);ca,te=train_test_split(rest,test_size=.5,stratify=pub.loc[rest,'y'],random_state=seed+1);sets=standard_sets(d,te);sets['MIMIC_test_no_CABG']=sets['MIMIC_test'][~flag(sets['MIMIC_test'].CABG_any)]
 for split,ids in [('train',tr),('calibration',ca),('test',te)]:splitrows.extend({'repeat':rep,'seed':seed,'split':split,'sid':sid} for sid in ids)
 for name in MAIN:
  obj=fit_model(name,d,tr.tolist(),ca.tolist(),seed=seed,tune=True);tuning.append({'repeat':rep,'model':name,'tuning':json.dumps(obj['tuning']),'calibration_slope':float(obj['calibrator'].coef_[0,0])})
  for group,c in sets.items():
   _,p=predict(obj,c);rows.append({'repeat':rep,'seed':seed,'model':name,'cohort':group,**scores(c.y,p)})
   if group.startswith('CAD'):localpred.append(c[['sid','y']].assign(repeat=rep,model=name,cohort=group,p=p))
  if rep in [0,19]:pickle.dump(obj,(PV/('stability_'+str(rep)+'_'+name+'.pkl')).open('wb'))
 pd.DataFrame(rows).to_csv(O/'repeated_split_metrics.csv',index=False);(O/'11_progress.json').write_text(json.dumps({'completed_repeats':rep+1,'total':20,'seconds':round(time.time()-st,1)}),encoding='utf8');print('REPEAT',rep+1,flush=True)
R=pd.DataFrame(rows);summary=[]
for (group,name),q in R.groupby(['cohort','model']):
 r={'cohort':group,'model':name,'repeats':len(q)}
 for f in ['auc','AP','Brier','mean_prediction','calibration_intercept','calibration_slope']:r[f+'_median']=q[f].median();r[f+'_min']=q[f].min();r[f+'_max']=q[f].max();r[f+'_q025']=q[f].quantile(.025);r[f+'_q975']=q[f].quantile(.975)
 summary.append(r)
pd.DataFrame(summary).to_csv(O/'repeated_split_summary.csv',index=False);pd.DataFrame(tuning).to_csv(O/'repeated_split_tuning.csv',index=False);pd.DataFrame(splitrows).to_csv(PV/'repeated_public_patient_splits.csv.gz',index=False);pd.concat(localpred,ignore_index=True).to_csv(PV/'repeated_split_local_predictions.csv.gz',index=False)
complete('11_stability',{'repeats':20,'each_GB_tuned_only_within_its_training_subset':True,'same_local_patients_reused_not_20_independent_external_cohorts':True,'local_not_used_for_model_selection':True,'seconds':round(time.time()-st,1)})
