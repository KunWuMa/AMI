from pathlib import Path
import pickle,pandas as pd,json,hashlib
R=Path(__file__).resolve().parents[1];S=R.parent;O=R/'结果汇总';base=S/'私有数据';ext=S/'论文_BMC_MIDM_LaTeX/审稿意见修订_R1_20261015/新增实验/私有数据';m=pd.read_csv(O/'frozen_model_manifest.csv');rows=[]
for name in m.model:
 obj=pickle.load(open((ext if name.startswith(('A6','B1','B2')) else base)/(name+'.pkl'),'rb'));model=obj['model'];cal=obj['calibrator']
 if not hasattr(model[-1],'coef_'):continue
 for j,f in enumerate(obj['features']):rows.append({'model':name,'feature':f,'training_median_imputation':model[0].statistics_[j],'training_standardization_mean':model[1].mean_[j],'training_standardization_scale':model[1].scale_[j],'coefficient_standardized':model[-1].coef_[0,j],'raw_logistic_intercept':model[-1].intercept_[0],'public_calibration_intercept':cal.intercept_[0],'public_calibration_slope':cal.coef_[0,0]})
pd.DataFrame(rows).to_csv(O/'frozen_logistic_equations.csv',index=False)
# Verify unchanged public predictions and cohort-specific statistical points against saved original predictions.
p=pd.read_csv(R/'私有数据_禁止公开/conditional_predictions.csv.gz');old=pd.read_csv(ext/'R1_all_fixed_predictions.csv.gz');z=p[p.cohort.eq('Public_test')].merge(old[old.cohort.eq('MIMIC_test')][['sid','model','p']],on=['sid','model'],suffixes=('','_old'));delta=(z.p-z.p_old).abs().max();assert delta<1e-12
(R/'public_reproduction.json').write_text(json.dumps({'predictions_compared':len(z),'maximum_probability_difference':delta,'public_models_and_allocation_unchanged':True},indent=2));print('public reproduction',len(z),delta,'equations',len(rows))
