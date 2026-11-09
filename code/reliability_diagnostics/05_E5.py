from analysis_common import *
b,l,p,t=load();future=pd.read_csv(D/'private_cache/public_future.csv').set_index('sid');labs=pd.read_csv(PV/'labs_confirmed_units.csv.gz',low_memory=False);hb=labs[labs.variable.eq('hb')&labs.day.between(2,7)&labs.value_std.notna()];lf=hb.groupby('sid').agg(first_future_day=('day','min'),derived_future_days=('day','nunique'),derived_future_rows=('day','size'));rows=[];checks=[]
for name,d,f in [('Public',p,future),('Local',l,lf)]:
 for c in ['first_future_day','derived_future_days','derived_future_rows']:d[c]=f[c].reindex(d.index)
 for c in ['derived_future_days','derived_future_rows']:d[c]=d[c].fillna(0)
 for a,z in [('derived_future_days','future_hb_days'),('derived_future_rows','future_hb_rows')]:
  err=np.max(np.abs(d[a]-d[z].fillna(0)));checks.append({'cohort':name,'check':a,'maximum_absolute_error':err,'passed':bool(err==0)});assert err==0
 d['any_future_hb']=d.derived_future_rows.gt(0).astype(int);assert d.any_future_hb.eq(d.observed).all()
 d['first_on_D2']=d.first_future_day.eq(2).astype(int);d['first_on_D3']=d.first_future_day.eq(3).astype(int);d['first_on_D4_D7']=d.first_future_day.between(4,7).astype(int)
 # Inclusive calendar dates with potential inpatient presence, not elapsed 24-hour days.
 d['potential_D2_D7_calendar_dates']=(d.remaining_calendar_days_after_D2+1).clip(upper=6)
 if name=='Public':
  dates=pd.read_csv(D/'private_cache/public_dates.csv').set_index('sid').loc[d.index];end=pd.concat([pd.to_datetime(dates.discharge),pd.to_datetime(dates.deathtime)],axis=1).min(axis=1);d['potential_D2_D7_calendar_dates']=((end.dt.normalize()-pd.to_datetime(dates.D0)).dt.days-1).clip(lower=0,upper=6)
 for label,g in [('All eligible',d),('Future Hb observed',d[d.observed.eq(1)]),('Future Hb unobserved',d[d.observed.eq(0)])]:
  for c in ['remaining_calendar_days_after_D2','potential_D2_D7_calendar_dates','any_future_hb','first_on_D2','first_on_D3','first_on_D4_D7','derived_future_days','derived_future_rows']:
   rows.append({'cohort':name,'group':label,'variable':c,'time_basis':'Specimen calendar date' if name=='Public' else 'Report calendar date',**stats(g[c],c in ['any_future_hb','first_on_D2','first_on_D3','first_on_D4_D7'])})
save(rows,'E5_future_Hb_opportunity.csv');save(checks,'E5_frozen_count_checks.csv');print(pd.DataFrame(rows).query('group == "All eligible"')[['cohort','variable','N','mean','median','q25','q75']].to_string(index=False));print('E5 complete')
