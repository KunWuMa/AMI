from supplement_common import *
from sklearn.metrics import roc_curve
PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');PR=PR[PR.cohort.isin(['CAD_AMI','CAD_all_ACS'])&PR.model.isin(MAIN)];rows=[];params=[];folds=[];parts=[];temporal=[];tpred=[]
def updater(p,y,method):
 z=logit(np.clip(np.asarray(p),1e-6,1-1e-6));y=np.asarray(y,dtype=int)
 if method=='intercept':a=brentq(lambda x:expit(x+z).sum()-y.sum(),-40,40);b=1.
 else:
  obj=LogisticRegression(C=1000,max_iter=3000).fit(z.reshape(-1,1),y);a=float(obj.intercept_[0]);b=float(obj.coef_[0,0])
 return a,b
for group,qq in PR.groupby('cohort'):
 ids=qq.drop_duplicates('sid').sort_values('sid');y=ids.y.to_numpy();sid=ids.sid.to_numpy();n=len(ids)
 for rep in range(10):
  cv=StratifiedKFold(5,shuffle=True,random_state=SEED+700+rep)
  for fold,(itr,ite) in enumerate(cv.split(np.zeros(n),y)):
   tr=sid[itr];te=sid[ite];assert set(tr).isdisjoint(te);folds.extend({'cohort':group,'repeat':rep,'fold':fold,'role':role,'sid':x} for role,arr in [('update_train',tr),('update_test',te)] for x in arr)
   for name,q0 in qq.groupby('model'):
    q=q0.set_index('sid');ptr=q.loc[tr,'p'].to_numpy();yt=q.loc[tr,'y'].to_numpy();ptest=q.loc[te,'p'].to_numpy()
    for method in ['frozen','intercept','platt']:
     a,b=(0.,1.) if method=='frozen' else updater(ptr,yt,method);pp=expit(a+b*logit(np.clip(ptest,1e-6,1-1e-6)));parts.append(q.loc[te,['y']].reset_index().assign(cohort=group,repeat=rep,fold=fold,base_model=name,method=method,p=pp));params.append({'cohort':group,'repeat':rep,'fold':fold,'base_model':name,'method':method,'a':a,'b':b,'train_n':len(tr),'train_events':int(yt.sum()),'test_n':len(te),'test_labels_not_used_in_update':True})
 print('CROSSFIT',group,flush=True)
CF=pd.concat(parts,ignore_index=True);CF.to_csv(PV/'local_update_crossfit_predictions.csv.gz',index=False);pd.DataFrame(folds).to_csv(PV/'local_update_fold_membership.csv.gz',index=False);pd.DataFrame(params).to_csv(O/'local_update_fold_parameters.csv',index=False)
for (group,name,method,rep),q in CF.groupby(['cohort','base_model','method','repeat']):assert q.sid.is_unique;rows.append({'cohort':group,'base_model':name,'method':method,'repeat':rep,**scores(q.y,q.p)})
pd.DataFrame(rows).to_csv(O/'local_update_repeated_metrics.csv',index=False);MEAN=CF.groupby(['cohort','base_model','method','sid']).agg(y=('y','first'),p=('p','mean'),repeats=('repeat','nunique')).reset_index();assert MEAN.repeats.eq(10).all();MEAN.to_csv(PV/'local_update_mean_predictions.csv.gz',index=False);m=[]
for (group,name,method),q in MEAN.groupby(['cohort','base_model','method']):m.append({'cohort':group,'base_model':name,'method':method,'evaluation':'10-repeat crossfit averaged predictions, INTERNAL updating assessment',**scores(q.y,q.p)})
pd.DataFrame(m).to_csv(O/'local_update_mean_metrics.csv',index=False);tmp=MEAN.copy();tmp['cohort']=tmp.cohort+'|'+tmp.base_model;ii,dd=bootstrap_fixed(tmp,baseline='frozen',B=1000,model_column='method');ii.to_csv(O/'local_update_conditional_intervals.csv',index=False);dd.to_csv(O/'local_update_paired_intervals.csv',index=False)
# Local historical holdout. The held-out recent outcomes are never used to update.
for group,qq in PR.groupby('cohort'):
 ids=qq.drop_duplicates('sid');earlier=ids[ids.year.lt(2024)];later=ids[ids.year.ge(2024)];supported=len(earlier)>=30 and earlier.y.sum()>=10 and (len(earlier)-earlier.y.sum())>=10 and len(later)>=30 and later.y.sum()>=10
 if not supported:temporal.append({'cohort':group,'status':'insufficient pre-fixed support','train_n':len(earlier),'train_events':int(earlier.y.sum()),'test_n':len(later),'test_events':int(later.y.sum())});continue
 for name,q0 in qq.groupby('model'):
  q=q0.set_index('sid');tr=earlier.sid;te=later.sid
  for method in ['frozen','intercept','platt']:
   a,b=(0.,1.) if method=='frozen' else updater(q.loc[tr,'p'],q.loc[tr,'y'],method);pp=expit(a+b*logit(np.clip(q.loc[te,'p'].to_numpy(),1e-6,1-1e-6)));temporal.append({'cohort':group,'base_model':name,'method':method,'train_n':len(tr),'train_events':int(q.loc[tr,'y'].sum()),'update_a':a,'update_b':b,'status':'historical holdout, update learned before 2024',**scores(q.loc[te,'y'],pp)});tpred.append(q.loc[te,['y']].reset_index().assign(cohort=group+'|'+name,base_model=name,method=method,p=pp))
pd.DataFrame(temporal).to_csv(O/'local_temporal_update_metrics.csv',index=False)
if tpred:
 TP=pd.concat(tpred,ignore_index=True);TP.to_csv(PV/'local_temporal_update_predictions.csv.gz',index=False);ii,dd=bootstrap_fixed(TP,baseline='frozen',B=1000,model_column='method');ii.to_csv(O/'local_temporal_update_intervals.csv',index=False);dd.to_csv(O/'local_temporal_update_paired_intervals.csv',index=False)
complete('13_local_update',{'crossfit_repeats':10,'folds_per_repeat':5,'methods':['frozen','intercept','platt_C1000'],'no_patient_outcome_used_in_their_own_update_fit':True,'local_internal_update_not_independent_external_validation':True,'temporal_holdout_cutoff':'2024-01-01','negative_fold_slopes':int(pd.DataFrame(params).b.lt(0).sum())})
