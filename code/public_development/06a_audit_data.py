from pathlib import Path
import pandas as pd,numpy as np,pickle,json,hashlib
P=Path(__file__).resolve().parents[1];R=P.parent;V=P/'私有数据';O=P/'结果'
D=pd.read_pickle(V/'harmonized_cohort.pkl');U=pd.read_pickle(V/'MIMIC_common_labs.pkl');L=pd.read_pickle(V/'CAD_common_labs.pkl');FA=pickle.load((V/'future_Hb_arrays.pkl').open('rb'));lookup={a['sid']:a for a in FA}
checks={'one_patient_one_index':bool(D.pid.is_unique),'MIMIC_unique_labevent_id':bool(U.labevent_id.is_unique),'known_outcomes_have_future_Hb':bool(D.loc[D.y.notna(),'future_hb_days'].ge(1).all()),'unknown_not_encoded_zero':bool(D.loc[D.eligible&D.future_hb_days.eq(0),'y'].isna().all()),'future_Hb_labels_match_raw_numeric_records':all(int(min(min(v) for v in lookup[c.sid]['day_values'].values())<c.cut)==c.y for c in D[D.y.notna()].itertuples()),'early_normal_guard':bool(D.loc[D.eligible,'early_min_hb'].ge(D.loc[D.eligible,'cut']).all()),'all_ages_capped_90':bool(D.age.le(90).all()),'public_units_known':bool(U.factor.notna().all()),'public_zero_store_missing':bool(U.store.notna().all())}
# Independently reconstruct public daily latest features from eligible pre-landmark data.
U['sid']='M'+U.hadm_id.astype(str);e=U[U.day.le(1)&U.available.lt(U.landmark)].copy();e['feature_day']=(e.available.dt.normalize()-e.D0).dt.days;daily=e.groupby(['sid','variable','feature_day']).value_std.median().dropna().reset_index();latest=daily.sort_values('feature_day').drop_duplicates(['sid','variable'],keep='last').pivot(index='sid',columns='variable',values='value_std');cmp=D[D.source.eq('MIMIC')].set_index('sid')
for v in ['hb','cr','alb','wbc','platelet','na','k','glucose']:
 a=cmp['last_'+v].to_numpy();b=latest[v].reindex(cmp.index).to_numpy();checks['early_feature_reconstruction_'+v]=bool(np.allclose(a,b,equal_nan=True));assert checks['early_feature_reconstruction_'+v]
# Audit structured numeric fields independently of placeholder text.
recovery=U.assign(placeholder_numeric=U.value.fillna('').eq('___')&U.valuenum.notna()).groupby('variable').agg(rows=('value_std','size'),valid=('value_std','count'),placeholder_numeric=('placeholder_numeric','sum'));recovery.to_csv(O/'最终结构化数值审计.csv');checks['placeholder_glucose_not_artificially_missing']=bool(U.loc[U.variable.eq('glucose')&U.value.fillna('').eq('___')&U.valuenum.between(9,1800),'value_std'].notna().all())
# Source binding: compute CAD ZIP checksum anew; MIMIC original complete intake hash + unchanged source stat.
manifest=json.loads((R/'双数据包_选题前探索分析/结果/source_manifest.json').read_text(encoding='utf8'));expected=[s['sha256'] for s in manifest['sources'] if s['group']=='cad'][0];src=R/'普通导出-冠状动脉药物-人维度-970-Empiid（马博）.zip';h=hashlib.sha256()
with src.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
checks['CAD_current_zip_sha_matches']=h.hexdigest()==expected;checks['MIMIC_complete_original_stream_read']=json.loads((O/'extraction_qc.json').read_text(encoding='utf8'))['gzip_EOF'];checks['MIMIC_source_unchanged_during_extraction']=json.loads((O/'extraction_qc.json').read_text(encoding='utf8'))['file_unchanged'];assert all(checks.values()),checks
files={}
for f in [V/'harmonized_cohort.csv.gz',V/'MIMIC_common_labs.pkl',V/'CAD_common_labs.pkl']:
 files[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
(O/'data_audit.json').write_text(json.dumps({'passed':True,'checks':checks,'CAD_zip_sha256':h.hexdigest(),'derived_input_sha256':files,'Python':__import__('sys').version,'pandas':pd.__version__,'numpy':np.__version__},indent=2),encoding='utf8');print('DATA_AUDIT_PASS')
