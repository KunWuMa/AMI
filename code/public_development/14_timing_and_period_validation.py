from supplement_common import *
all_metrics=[];all_pred=[];flows=[];st=time.time()
def experiment(tag,d,tr,ca,sets):
 assert all(set(tr).isdisjoint(c.index) and set(ca).isdisjoint(c.index) for c in sets.values());objects={}
 for name in MAIN:
  objects[name]=fit_model(name,d,list(tr),list(ca),seed=SEED+900,tune=True);pickle.dump(objects[name],(PV/(tag+'_'+name+'.pkl')).open('wb'))
 m,p=evaluate_models(objects,sets,tag);m.to_csv(O/(tag+'_metrics.csv'),index=False);p.to_csv(PV/(tag+'_predictions.csv.gz'),index=False);ii,dd=bootstrap_fixed(p,B=1000);ii.to_csv(O/(tag+'_intervals.csv'),index=False);dd.to_csv(O/(tag+'_paired.csv'),index=False);all_metrics.append(m);all_pred.append(p);pd.DataFrame([{'sid':sid,'split':split,'experiment':tag} for split,ids in [('train',tr),('calibration',ca)] for sid in ids]+[{'sid':sid,'split':group,'experiment':tag} for group,c in sets.items() for sid in c.sid]).to_csv(PV/(tag+'_patient_splits.csv.gz'),index=False)
 flows.append({'experiment':tag,'status':'completed','train_n':len(tr),'calibration_n':len(ca),**{k+'_n':len(c) for k,c in sets.items()}});print('DONE',tag,flush=True)
d=load_data();meta=pd.read_csv(PV/'MIMIC_timing_metadata.csv.gz').set_index('sid');pub=d[d.source.eq('MIMIC')&d.AMI&d.y.notna()];lo=meta.admit_year_lower.reindex(pub.index);hi=meta.admit_year_upper.reindex(pub.index);early=pub[hi.le(2016)];late=pub[lo.ge(2017)]
if len(early)>=250 and len(late)>=150 and min(early.y.sum(),len(early)-early.y.sum(),late.y.sum(),len(late)-late.y.sum())>=50:
 tr,ca=train_test_split(early.index.to_numpy(),test_size=.25,stratify=early.y,random_state=SEED+901);sets={'MIMIC_late_period':late,'CAD_AMI':d[d.source.eq('CAD')&d.AMI&d.y.notna()],'CAD_all_ACS':d[d.source.eq('CAD')&d.y.notna()]};experiment('MIMIC_period',d,tr,ca,sets)
else:flows.append({'experiment':'MIMIC_period','status':'insufficient pre-fixed support','early_n':len(early),'late_n':len(late)})
period={'earlier_upper_le2016_n':len(early),'earlier_events':int(early.y.sum()),'later_lower_ge2017_n':len(late),'later_events':int(late.y.sum()),'ambiguous_period_excluded':len(pub)-len(early)-len(late),'note':'Real year intervals inferred using anchor group plus relative year offset; not shifted-year chronological sorting'};(O/'MIMIC_period_support.json').write_text(json.dumps(period,indent=2),encoding='utf8')
for tag in ['D3_calendar','strict48h']:
 z=load_data(PV/(tag+'_cohort.csv.gz'));pub=z[z.source.eq('MIMIC')&z.AMI&z.y.notna()];tr,rest=train_test_split(pub.index.to_numpy(),test_size=.4,stratify=pub.y,random_state=SEED+902);ca,te=train_test_split(rest,test_size=.5,stratify=pub.loc[rest,'y'],random_state=SEED+903);sets=standard_sets(z,te)
 if tag=='strict48h':sets={k:c for k,c in sets.items() if k.startswith('MIMIC')}
 experiment(tag,z,tr,ca,sets)
# Fixed original predictions evaluated under shorter/report-based outcome windows or pending-Hb exclusion.
PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');T=pd.read_csv(PV/'timing_fixed_label_sensitivities.csv.gz').set_index('sid');sens=[]
for (group,name),q in PR[PR.model.isin(MAIN)].groupby(['cohort','model']):
 for target in ['y_D2D3','y_D2D5','y_reported_in_window']:
  y=T[target].reindex(q.sid).to_numpy();valid=np.isfinite(y)
  if valid.any():sens.append({'cohort':group,'model':name,'target':target,'unknown_in_original_observed':int((~valid).sum()),'interpretation':'fixed original predictions, changed target window; not a newly trained short-horizon model',**scores(y[valid],q.p.to_numpy()[valid])})
 if group=='MIMIC_test':
  pending=T.pending_early_Hb.reindex(q.sid).fillna(False).to_numpy().astype(bool);valid=~pending;sens.append({'cohort':group,'model':name,'target':'exclude_pending_early_Hb','excluded':int(pending.sum()),**scores(q.y.to_numpy()[valid],q.p.to_numpy()[valid])})
pd.DataFrame(sens).to_csv(O/'fixed_prediction_time_sensitivities.csv',index=False);pd.DataFrame(flows).to_csv(O/'time_period_experiment_flow.csv',index=False)
complete('14_time_period',{'experiments':flows,'different_landmark_populations_not_head_to_head_superiority_tests':True,'local_precise48h_not_fabricated':True,'seconds':round(time.time()-st,1)})
