from pathlib import Path
import os,pandas as pd,pickle,hashlib,json
P=Path(os.environ['AMI_STUDY_ROOT']);D=Path(__file__).resolve().parents[2];V=P/'私有数据'
(D/'private_cache').mkdir(parents=True,exist_ok=True)
files=list((P/'投稿版本_20261105').rglob('*'))+list(V.glob('M[1-4]*.pkl'))+list((P/'AMI新数据扩展实验_20261022'/'结果汇总').glob('*.csv'))
manifest={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(D/'private_cache'/'baseline_hashes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
a=pd.read_pickle(V/'MIMIC_index_cohort.pkl');a['sid']='M'+a.hadm_id.astype(str);a[['sid','D0','discharge','deathtime']].to_csv(D/'private_cache'/'public_dates.csv',index=False)
a=pd.read_pickle(V/'future_Hb_arrays.pkl');out=[]
for r in a:
 days=r['day_values'];out.append({'sid':r['sid'],'first_future_day':min(map(int,days)) if days else None,'derived_future_days':len(days),'derived_future_rows':sum(len(x) for x in days.values())})
pd.DataFrame(out).to_csv(D/'private_cache'/'public_future.csv',index=False)
print('Private cache and baseline hashes prepared')
