from analysis_common import *
b,l,p,t=load();l=l[l.observed.eq(1)];p=p[p.observed.eq(1)];rows=[];performance=[]
for name,d in [('Public_all_observed',p),('Public_test',t)]:
 a=d[d.CABG_any.eq(True)];c=d[~d.CABG_any.eq(True)]
 assert (len(a),int(a.y.sum()),len(c),int(c.y.sum()))==((379,367,1467,472) if name=='Public_all_observed' else (80,78,290,90))
 rows+=compare(a,c,PHYS+AV+COUNTS+MISS+['y'],name,'CABG','No CABG code',BINARY+['y'])
 for label,g in [('CABG',a),('No CABG code',c)]:
  for n in MAIN[1:3]:
   pp=g['risk_'+n];performance.append({'cohort':name,'group':label,'model':n,'N':len(g),'events':int(g.y.sum()),'event_fraction':g.y.mean(),'mean_prediction':pp.mean(),'Brier':np.mean((pp-g.y)**2),'interpretation':'Descriptive mixed development/calibration/test sample' if name=='Public_all_observed' else 'Held-out test sample'})
save(rows,'E2_public_CABG_case_mix.csv');save(performance,'E2_public_CABG_prediction_summary.csv')
rows=compare(t,l,PHYS+AV+TWO+COUNTS+MISS,'Cross-setting','Public_test','Local100',BINARY)
for v in COUNTS:
 for k in [0,1,2]:
  a=t.assign(count_category=t[v].eq(k).astype(int));c=l.assign(count_category=l[v].eq(k).astype(int));rr=compare(a,c,['count_category'],'Cross-setting','Public_test','Local100',['count_category'])
  for r in rr:r['variable']=v+' equals '+str(k)
  rows+=rr
save(rows,'E2_cross_setting_observation_features.csv');print(pd.DataFrame(performance).to_string(index=False));print('E2 complete')
