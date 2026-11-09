from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pandas as pd,numpy as np,pickle,json,hashlib,sys
import sklearn,scipy,statsmodels,matplotlib
from sklearn.metrics import roc_auc_score,brier_score_loss
from scipy.special import logit
P=Path(__file__).resolve().parents[1];V=P/'私有数据';O=P/'结果';D=pd.read_csv(V/'harmonized_cohort.csv.gz').set_index('sid',drop=False);SP=pd.read_csv(V/'MIMIC_patient_split.csv');PR=pd.read_csv(V/'fixed_external_predictions.csv.gz');M=pd.read_csv(O/'model_metrics.csv');checks={}
checks['split_patients_unique']=bool(SP.pid.is_unique);checks['public_only_model_split']=bool(SP.sid.str.startswith('M').all());checks['all_split_labels_known']=bool(D.loc[SP.sid,'y'].notna().all());checks['all_predictions_finite']=bool(np.isfinite(PR.p).all());checks['labels_match_final_cleaned_data']=bool((PR.y.to_numpy()==D.loc[PR.sid,'y'].to_numpy()).all());checks['primary_test_n_matches_patient_split']=len(PR[PR.model.eq('M1_clinical')&PR.cohort.eq('MIMIC_test')])==SP.split.eq('test').sum()
tr=SP.loc[SP.split.eq('train'),'sid'];forbidden={'y','AMI','ICU_any','CABG_any','future_hb_days','future_hb_rows','late_min_hb','year','y_drop20','y_lt110'}
for name,q in PR.groupby('model'):
 obj=pickle.load((V/(name+'.pkl')).open('rb'));features=obj['features'];checks['no_future_inputs_'+name]=not bool(set(features)&forbidden)
 X=D.copy()
 for f in features:
  if f.startswith('missing_'):X[f]=X[f[len('missing_'):]].isna().astype(int)
 raw=obj['model'].predict_proba(X.loc[q.sid,features])[:,1];pp=obj['calibrator'].predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1];checks['saved_model_reproduces_predictions_'+name]=bool(np.allclose(pp,q.p,atol=1e-12))
 expected=X.loc[tr,features].median().fillna(0).to_numpy();checks['imputation_uses_training_only_'+name]=bool(np.allclose(expected,obj['model'][0].statistics_))
 for cohort,sub in q.groupby('cohort'):
  row=M[M.model.eq(name)&M.cohort.eq(cohort)&M.version.eq('public_calibrated')].iloc[0];checks['metrics_'+name+'_'+cohort]=abs(roc_auc_score(sub.y,sub.p)-row.auc)<1e-12 and abs(brier_score_loss(sub.y,sub.p)-row.Brier)<1e-12
assert all(checks.values()),[k for k,v in checks.items() if not v]
checks={k:bool(v) for k,v in checks.items()}
manifest={'checks':checks,'passed':True,'environment':{'Python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'matplotlib':matplotlib.__version__},'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((P/'代码').glob('*.py'))},'predictions_sha256':hashlib.sha256((V/'fixed_external_predictions.csv.gz').read_bytes()).hexdigest()};(O/'model_audit.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');(P/'代码/模型运行依赖.txt').write_text('\n'.join(['numpy=='+np.__version__,'pandas=='+pd.__version__,'scikit-learn=='+sklearn.__version__,'scipy=='+scipy.__version__,'statsmodels=='+statsmodels.__version__,'matplotlib=='+matplotlib.__version__]),encoding='utf8');print('MODEL_AUDIT_PASS')
