from evaluation_common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import statsmodels.api as sm
D=sc.load_data();SP=sc.splits();sets=sc.standard_sets(D,SP['test']);pr=pd.read_csv(V/'fixed_external_predictions.csv.gz');pr=pr[pr.cohort.isin(sets)&pr.model.ne('M1D_early_change')][['sid','cohort','model','y','p']];ab=pd.read_csv(V/'exploratory_ablation_predictions.csv.gz');ab=ab[ab.cohort.isin(sets)][['sid','cohort','model','y','p']]
# Correct the real old A4 feature set; the previous wording was inaccurate.
a4=pickle.load((V/'A4_M3_without_alb_process.pkl').open('rb'));assert 'last_alb' in a4['features'];assert 'n_days_alb' not in a4['features'] and 'missing_last_alb' not in a4['features']
new={
 'A6_M3_no_albumin_any':[f for f in sc.FEATURES['M3_observation_LR'] if f not in ['last_alb','n_days_alb','missing_last_alb']],
 'B1_M3_binary_presence':sc.PHYS+['any_'+x for x in sc.COUNTS],
 'B2_M3_repeat_indicators':sc.PHYS+['repeat_'+x for x in sc.COUNTS],
}
for f in sc.COUNTS:D['any_'+f]=D[f].ge(1).astype(int);D['repeat_'+f]=D[f].ge(2).astype(int)
models=[];newpred=[];metrics=[]
for model,feats in new.items():
 sc.FEATURES[model]=feats;obj=sc.fit_model(model,D,SP['train'],SP['calibration'],seed=SEED);pickle.dump(obj,(PV/(model+'.pkl')).open('wb'));models.append({'model':model,'features':feats,'source_fit_only':True})
 for cohort,c0 in sets.items():
  c=D.loc[c0.sid];_,p=sc.predict(obj,c);newpred.append(c[['sid','y']].assign(cohort=cohort,model=model,p=p))
ALL=pd.concat([pr,ab,*newpred],ignore_index=True);assert not ALL[['sid','cohort','model']].duplicated().any();ALL.to_csv(PV/'R1_all_fixed_predictions.csv.gz',index=False)
for (cohort,model),q in ALL.groupby(['cohort','model']):metrics.append({'cohort':cohort,'model':model,**sc.scores(q.y,q.p)})
pd.DataFrame(metrics).to_csv(O/'R1_fixed_model_metrics.csv',index=False)
pairs=[('M1_clinical','M0_Hb'),('M2_routine_LR','M1_clinical'),('M3_observation_LR','M1_clinical'),('M4_routine_GB','M1_clinical'),('M3_observation_LR','M2_routine_LR'),('A3_M2_without_alb','M2_routine_LR'),('A4_M3_without_alb_process','M3_observation_LR'),('A6_M3_no_albumin_any','M3_observation_LR'),('B1_M3_binary_presence','M3_observation_LR'),('B2_M3_repeat_indicators','M3_observation_LR')]
it,di=paired_bootstrap(ALL,pairs);it.to_csv(O/'R1_fixed_model_intervals.csv',index=False);di.to_csv(O/'R1_fixed_paired_differences.csv',index=False)
# Existing frozen models in the same public no-CABG test population; do not relabel this as redevelopment.
q=pr[pr.cohort.eq('MIMIC_test')].copy();q['CABG']=D.loc[q.sid,'CABG_any'].to_numpy().astype(bool);sub=q[~q.CABG].drop(columns='CABG').assign(cohort='Public_nonCABG_frozen');rows=[]
for model,z in sub.groupby('model'):rows.append({'cohort':'Public_nonCABG_frozen','model':model,**sc.scores(z.y,z.p)})
pd.DataFrame(rows).to_csv(O/'public_nonCABG_frozen_metrics.csv',index=False);it,di=paired_bootstrap(sub,pairs);it.to_csv(O/'public_nonCABG_frozen_intervals.csv',index=False);di.to_csv(O/'public_nonCABG_frozen_paired.csv',index=False)
# Can early observation patterns identify the later recorded CABG pathway? Diagnostic association only.
c=D[D.source.eq('MIMIC')&D.AMI&D.y.notna()].copy();y=c.CABG_any.astype(bool).astype(int).to_numpy();cv=StratifiedKFold(5,shuffle=True,random_state=SEED);features={'clinical':sc.FEATURES['M1_clinical'],'physiological':sc.PHYS,'observation_only':sc.COUNTS+sc.MISS,'physiology_plus_observation':sc.FEATURES['M3_observation_LR']};pathmet=[];pathpr=[]
for name,feats in features.items():
 p=np.empty(len(c))
 for tr,te in cv.split(c,y):
  model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=3000));model.fit(c.iloc[tr][feats],y[tr]);p[te]=model.predict_proba(c.iloc[te][feats])[:,1];assert set(c.iloc[tr].sid).isdisjoint(c.iloc[te].sid)
 pathmet.append({'features':name,'endpoint':'CABG_any_during_index_admission','prediction_type':'cross-fitted diagnostic, not a clinical deployment model',**sc.scores(y,p)});pathpr.append(c[['sid']].assign(cohort='Public_recorded_CABG_diagnostic',model=name,y=y,p=p))
pd.DataFrame(pathmet).to_csv(O/'CABG_pathway_classifier_metrics.csv',index=False);pathpr=pd.concat(pathpr,ignore_index=True);pathpr.to_csv(PV/'CABG_pathway_classifier_predictions.csv.gz',index=False);it,_=paired_bootstrap(pathpr,[]);it.to_csv(O/'CABG_pathway_classifier_intervals.csv',index=False)
# Calibration diagnostic uncertainty for the small same-diagnosis local observed sample.
q=ALL[(ALL.cohort=='CAD_AMI')&ALL.model.isin(['M1_clinical','M2_routine_LR','M3_observation_LR'])];calrows=[];rng=np.random.default_rng(SEED+31)
for name,g in q.groupby('model'):
 y=g.y.to_numpy();z=logit(np.clip(g.p.to_numpy(),1e-6,1-1e-6));vals=[]
 for b in range(1000):
  ix=rng.integers(0,len(y),len(y));yy=y[ix]
  if len(np.unique(yy))<2:continue
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');a=sm.GLM(yy,np.ones((len(ix),1)),offset=z[ix],family=sm.families.Binomial()).fit();s=sm.GLM(yy,sm.add_constant(z[ix]),family=sm.families.Binomial()).fit()
   vals.append([a.params[0],s.params[1]])
  except Exception:pass
 for j,m in enumerate(['calibration_intercept','calibration_slope']):
  v=np.asarray(vals)[:,j];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]);calrows.append({'cohort':'CAD_AMI','model':name,'metric':m,'low':lo,'high':hi,'valid_fits':len(v),'B':1000,'conditional_on_public_fitted_model':True})
pd.DataFrame(calrows).to_csv(O/'local_calibration_diagnostic_intervals.csv',index=False)
(O/'R1_feature_audit.json').write_text(json.dumps({'A4_old_actual_features':a4['features'],'A4_retains_albumin_value':True,'new_models':models,'counts_original_support':{f:sorted(map(int,D.loc[SP['train'],f].unique())) for f in sc.COUNTS},'site_specific_label_adaptation':False},indent=2),encoding='utf8')
done('03_ablations_pathways',{'models_total':ALL.model.nunique(),'new_feature_models':len(new),'B':1000,'A4_definition_corrected':True,'no_claim_of_causal_pathway_mediation':True})
