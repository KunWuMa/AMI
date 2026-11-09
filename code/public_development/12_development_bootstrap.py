from supplement_common import *
d=load_data();s=splits();sets=standard_sets(d,s['test']);rng=np.random.default_rng(SEED+500);gb=pickle.load((V/'M4_routine_GB.pkl').open('rb'))['model'][-1].get_params();rows=[];draws={};trbase=np.array(s['train']);cabase=np.array(s['calibration']);train_draw=[];cal_draw=[];eval_draw={g:[] for g in sets};st=time.time();samples=[]
for rep in range(300):
 it=rng.integers(0,len(trbase),len(trbase));ic=rng.integers(0,len(cabase),len(cabase));train_draw.append(it);cal_draw.append(ic);ixs={g:rng.integers(0,len(c),len(c)) for g,c in sets.items()}
 for g,ix in ixs.items():eval_draw[g].append(ix)
 for name in MAIN:
  obj=fit_model(name,d,trbase[it].tolist(),cabase[ic].tolist(),seed=SEED+500+rep,tune=False,gbparams=gb if name=='M4_routine_GB' else None)
  for group,c in sets.items():
   _,p=predict(obj,c);ix=ixs[group];rows.append({'replicate':rep,'model':name,'cohort':group,**scores(c.y.to_numpy()[ix],p[ix],calibration=False)})
   if rep in [0,149,299]:samples.append(c[['sid','y']].assign(replicate=rep,cohort=group,model=name,p=p))
  if rep in [0,149,299]:pickle.dump(obj,(PV/('development_boot_'+str(rep)+'_'+name+'.pkl')).open('wb'))
 if (rep+1)%10==0:pd.DataFrame(rows).to_csv(O/'development_bootstrap_metrics.csv',index=False);(O/'12_progress.json').write_text(json.dumps({'completed':rep+1,'total':300,'seconds':round(time.time()-st,1)}),encoding='utf8');print('BOOT',rep+1,flush=True)
R=pd.DataFrame(rows);summ=[];paired=[]
for (group,name),q in R.groupby(['cohort','model']):
 obj=pickle.load((V/(name+'.pkl')).open('rb'));_,p=predict(obj,sets[group]);point=scores(sets[group].y,p,calibration=False)
 for f in ['auc','AP','Brier']:lo,hi=q[f].quantile([.025,.975]);summ.append({'cohort':group,'model':name,'metric':f,'original_point':point[f],'bootstrap_mean':q[f].mean(),'low':lo,'high':hi,'replicates':len(q),'interval':'Development+calibration+evaluation patient bootstrap, fixed GB hyperparameters'})
for group,q in R.groupby('cohort'):
 for f in ['auc','Brier']:
  w=q.pivot(index='replicate',columns='model',values=f)
  for baseline in ['M1_clinical','M2_routine_LR']:
   for name in MAIN:
    if name==baseline:continue
    dif=w[name]-w[baseline];lo,hi=dif.quantile([.025,.975]);paired.append({'cohort':group,'model':name,'baseline':baseline,'metric':'delta_'+f,'bootstrap_mean_difference':dif.mean(),'low':lo,'high':hi})
pd.DataFrame(summ).to_csv(O/'development_bootstrap_intervals.csv',index=False);pd.DataFrame(paired).to_csv(O/'development_bootstrap_paired_intervals.csv',index=False);np.savez_compressed(PV/'development_bootstrap_draws.npz',train=np.asarray(train_draw),calibration=np.asarray(cal_draw),**{k:np.asarray(v) for k,v in eval_draw.items()});pd.concat(samples,ignore_index=True).to_csv(PV/'development_bootstrap_audit_predictions.csv.gz',index=False)
complete('12_development_bootstrap',{'replicates':300,'training_calibration_and_evaluation_resampled':True,'GB_hyperparameter_selection_not_resampled':True,'external_never_training':True,'seconds':round(time.time()-st,1)})
