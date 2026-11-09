from supplement_common import *
d=load_data();s=splits();keep=~flag(d.CABG_any);tr=[i for i in s['train'] if keep.loc[i]];ca=[i for i in s['calibration'] if keep.loc[i]];te=[i for i in s['test'] if keep.loc[i]];datasets=standard_sets(d,te);datasets['MIMIC_test_no_CABG']=datasets.pop('MIMIC_test');objects={}
for name in FEATURES:
 objects[name]=fit_model(name,d,tr,ca);pickle.dump(objects[name],(PV/('nonCABG_'+name+'.pkl')).open('wb'));print('FIT',name,flush=True)
metrics,pred=evaluate_models(objects,datasets,'nonCABG_development');metrics.to_csv(O/'nonCABG_model_metrics.csv',index=False);pred.to_csv(PV/'nonCABG_predictions.csv.gz',index=False);ints,paired=bootstrap_fixed(pred,B=1000);ints.to_csv(O/'nonCABG_performance_intervals.csv',index=False);paired.to_csv(O/'nonCABG_paired_models.csv',index=False)
# Fair direct comparison on the SAME nonCABG public test / local patients, not across different tests.
comp=[];rng=np.random.default_rng(SEED);orig=[]
for name in MAIN:
 obj=pickle.load((V/(name+'.pkl')).open('rb'))
 for group,c in datasets.items():
  _,p=predict(obj,c);new=pred[pred.model.eq(name)&pred.cohort.eq(group)].set_index('sid').loc[c.sid,'p'].to_numpy();y=c.y.to_numpy();a=[]
  for b in range(1000):
   ix=rng.integers(0,len(y),len(y));yy=y[ix]
   if len(np.unique(yy))<2:continue
   a.append([roc_auc_score(yy,new[ix])-roc_auc_score(yy,p[ix]),np.mean((new[ix]-yy)**2-(p[ix]-yy)**2)])
  a=np.asarray(a)
  for k,m in enumerate(['auc','Brier']):lo,hi=np.quantile(a[:,k],[.025,.975]);comp.append({'cohort':group,'model':name,'comparison':'nonCABG development minus all-AMI development','metric':'delta_'+m,'difference':scores(y,new,calibration=False)[m]-scores(y,p,calibration=False)[m],'low':lo,'high':hi})
  orig.append(c[['sid','y']].assign(cohort=group,model=name,p=p))
pd.DataFrame(comp).to_csv(O/'nonCABG_vs_original_development.csv',index=False);pd.concat(orig).to_csv(PV/'nonCABG_original_comparator_predictions.csv.gz',index=False);pd.DataFrame([{'split':k,'n':len(v),'events':int(d.loc[v,'y'].sum())} for k,v in [('train',tr),('calibration',ca),('test',te)]]).to_csv(O/'nonCABG_split_counts.csv',index=False)
complete('10_nonCABG',{'train_n':len(tr),'calibration_n':len(ca),'test_n':len(te),'future_CABG_only_for_retrospective_cohort_selection':True,'local_CABG_exclusion_not_verified':True,'local_labels_never_fit':True,'models':list(FEATURES)})
