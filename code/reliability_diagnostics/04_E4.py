from analysis_common import *
b,l,p,t=load();l=l[l.observed.eq(1)];o=obj('M3_observation_LR');pipe=o['model'];imp,sc,lr=pipe.steps[0][1],pipe.steps[1][1],pipe.steps[2][1];cal=o['calibrator'];a=float(cal.intercept_[0]);s=float(cal.coef_[0,0]);beta=lr.coef_[0];inter=float(lr.intercept_[0]);features=o['features']
units=dict(age='years',male='1 male; 0 female',last_margin='g/L',last_cr='umol/L',last_alb='g/L',last_wbc='10^9/L',last_platelet='10^9/L',last_na='mmol/L',last_k='mmol/L',last_glucose='mmol/L')
def group(f):return 'Observation days' if f in COUNTS else ('Missingness' if f in MISS else 'Clinical/physiological')
rows=[]
for i,f in enumerate(features):rows.append({'feature':f,'category':group(f),'meaning':('Number of distinct D0-D1 available calendar days: '+f[7:] if f in COUNTS else '1 if pre-imputation '+f[8:]+' is missing, else 0' if f in MISS else 'Latest available early value; margin is Hb minus sex-specific threshold' if f.startswith('last_') else f),'unit_or_encoding':units.get(f,'calendar days (0,1,2)' if f in COUNTS else 'binary 0/1'),'standardized_coefficient':beta[i],'training_imputation_median':imp.statistics_[i],'training_scaler_mean':sc.mean_[i],'training_scaler_scale':sc.scale_[i],'imputation_rule':'Frozen public-training median; indicators defined before imputation','calibrated_log_odds_coefficient':s*beta[i]})
save(rows,'E4_frozen_M3_coefficients.csv');save([{'raw_intercept':inter,'calibration_intercept':a,'calibration_slope':s,'raw_probability_clip_lower':1e-6,'raw_probability_clip_upper':1-1e-6,'combined_intercept':a+s*inter}],'E4_frozen_M3_calibration_transform.csv')
rows=[];patterns=[];qc=[]
for name,d in [('Public_test',t),('Local100',l)]:
 xx=sc.transform(imp.transform(d[features]));contrib=xx*beta;raw_lp=inter+contrib.sum(axis=1);clipped_lp=logit(np.clip(expit(raw_lp),1e-6,1-1e-6));correction=clipped_lp-raw_lp;final_lp=a+s*(raw_lp+correction);err=np.max(np.abs(expit(final_lp)-predict(o,d)));assert err<1e-12
 qc.append({'cohort':name,'N':len(d),'max_abs_probability_reconstruction_error':err,'clipped_patients':int((np.abs(correction)>1e-10).sum()),'passed':True})
 for scale,mult in [('Raw linear predictor',1),('Public-calibrated log odds',s)]:
  for g in ['Clinical/physiological','Observation days','Missingness']:
   values=pd.Series(contrib[:,[i for i,f in enumerate(features) if group(f)==g]].sum(axis=1)*mult);rows.append({'cohort':name,'scale':scale,'component':g,**stats(values)})
  rows.append({'cohort':name,'scale':scale,'component':'Clipping correction',**stats(pd.Series(correction*mult))})
  rows.append({'cohort':name,'scale':scale,'component':'Intercept',**stats(pd.Series(np.repeat(inter if mult==1 else a+s*inter,len(d))))})
 for i,f in enumerate(features):rows.append({'cohort':name,'scale':'Public-calibrated log odds','component':f,**stats(pd.Series(s*contrib[:,i]))})
 for f in ['available_alb','n_days_hb']:
  for val,q in d.groupby(f):patterns.append({'cohort':name,'pattern':f,'value':val,**stats(q.risk_M3_observation_LR),'interpretation':'Descriptive predicted risk only; no subgroup performance estimate'})
save(rows,'E4_prediction_contributions_by_setting.csv');save(patterns,'E4_observation_pattern_risk_summary.csv');save(qc,'E4_decomposition_checks.csv');print(pd.DataFrame(rows).query('component in ["Clinical/physiological","Observation days","Missingness"]')[['cohort','scale','component','mean','median']].to_string(index=False));print('E4 complete')
