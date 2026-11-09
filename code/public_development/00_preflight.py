import os
from pathlib import Path
import pandas as pd,json,sys,importlib.util
R=Path(os.environ.get('AMI_WORKSPACE_ROOT', 'workspace'));M=R/'公开数据集/MIMIC-IV/2.2';P=R/'冠脉包_MIMIC_贫血风险_正式研究'
l=pd.read_csv(M/'d_labitems.csv.gz');want=['Hemoglobin','Creatinine','Albumin','White Blood Cells','WBC Count','Platelet Count','Sodium','Potassium','Glucose'];x=l[l.fluid.astype(str).str.lower().eq('blood')&l.label.isin(want)];(P/'结果/可用项目字典.json').write_text(x.to_json(orient='records',force_ascii=False,indent=2),encoding='utf8');print(x[['itemid','label','category']].to_string(index=False));print('PYTHON',sys.version);print('LIBS',{x:bool(importlib.util.find_spec(x)) for x in ['pandas','numpy','sklearn','scipy','statsmodels','matplotlib']})
