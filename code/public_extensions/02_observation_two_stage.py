from evaluation_common import *
from sklearn.model_selection import train_test_split
from scipy.special import logit
D=sc.load_data();SP=sc.splits();E=D[D.source.eq('MIMIC')&D.AMI&D.eligible].copy();unknown=E[E.y.isna()].sid.to_numpy();trU,rest=train_test_split(unknown,test_size=.4,random_state=SEED);caU,teU=train_test_split(rest,test_size=.5,random_state=SEED+1);sp={k:list(SP[k])+list(u) for k,u in [('train',trU),('calibration',caU),('test',teU)]};assert set().union(*map(set,sp.values()))==set(E.sid);assert sum(map(len,sp.values()))==len(E)
rows=[]
for split,ids in sp.items():
 c=D.loc[ids];rows.append({'split':split,'n':len(c),'O_observed':int(c.y.notna().sum()),'not_observed':int(c.y.isna().sum()),'joint_recorded_events':int(c.y.sum())})
pd.DataFrame(rows).to_csv(O/'two_stage_public_splits.csv',index=False);pd.concat([pd.DataFrame({'sid':ids,'split':k}) for k,ids in sp.items()]).to_csv(PV/'two_stage_split_ids.csv.gz',index=False)
# O is a separate endpoint, known for every early-normal eligible patient.
DO=D.copy();DO['y']=D.y.notna().astype(int);sets={'Public_AMI_test_eligible':D.loc[sp['test']],'Local_AMI_eligible':D[D.source.eq('CAD')&D.AMI&D.eligible],'Local_ACS_eligible':D[D.source.eq('CAD')&D.eligible]};out=[];private=[];objs={}
for model in ['M1_clinical','M2_routine_LR','M3_observation_LR']:
 om=sc.fit_model(model,DO,sp['train'],sp['calibration'],seed=SEED);objs[model]=om;am=pickle.load((V/(model+'.pkl')).open('rb'));pickle.dump(om,(PV/('observation_'+model+'.pkl')).open('wb'))
 for cohort,c in sets.items():
  _,po=sc.predict(om,c);_,pa=sc.predict(am,c);o=c.y.notna().astype(int).to_numpy();joint=c.y.fillna(0).astype(int).to_numpy();assert np.all(joint<=o)
  for task,y,p in [('O_future_Hb_observed',o,po),('J_joint_recorded_event',joint,po*pa)]:
   out.append({'task':task,'cohort':cohort,'model':model,**sc.scores(y,p)});private.append(c[['sid','pid']].assign(cohort=cohort,model=model,task=task,y=y,p=p))
  assert c.loc[c.y.isna(),'y'].isna().all()
MET=pd.DataFrame(out);PR=pd.concat(private,ignore_index=True);MET.to_csv(O/'two_stage_metrics.csv',index=False);PR.to_csv(PV/'two_stage_predictions.csv.gz',index=False)
for task,q in PR.groupby('task'):
 it,di=paired_bootstrap(q,[('M2_routine_LR','M1_clinical'),('M3_observation_LR','M1_clinical'),('M3_observation_LR','M2_routine_LR')],seed=SEED+9);it.to_csv(O/(task+'_intervals.csv'),index=False);di.to_csv(O/(task+'_paired.csv'),index=False)
# Describe O and J without interpreting J=0 as biological absence.
counts=[]
for name,c in [('Public_AMI_eligible',E),*sets.items()]:
 obs=int(c.y.notna().sum());ev=int(c.y.sum());counts.append({'cohort':name,'eligible_n':len(c),'O_observed_n':obs,'O_probability':obs/len(c),'A_conditional_events':ev,'A_conditional_probability':ev/obs,'J_recorded_events':ev,'J_probability':ev/len(c),'A_unknown_n':len(c)-obs})
pd.DataFrame(counts).to_csv(O/'estimand_counts.csv',index=False)
done('02_two_stage',{'public_splits':rows,'conditional_A_labels_unchanged':True,'local_labels_not_used_for_training_or_calibration':True,'J_zero_for_unmeasured_means_no_recorded_event_only':True})
