from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
for name,table in json.loads((R/'aggregate_results/supporting_results.json').read_text(encoding='utf8')).items():
 p=R/'aggregate_results'/(name+'.csv')
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=table['columns']);w.writeheader();w.writerows(table['rows'])
(R/'figures').mkdir(exist_ok=True)
print('Aggregate tables restored; no models fitted or metrics recalculated.')
