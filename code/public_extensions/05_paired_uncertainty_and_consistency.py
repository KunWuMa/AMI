from evaluation_common import *
from scipy.special import logit
p=pd.read_csv(PV/'R1_all_fixed_predictions.csv.gz');pairs=[('M1_clinical','M0_Hb'),('M2_routine_LR','M1_clinical'),('M3_observation_LR','M1_clinical'),('M4_routine_GB','M1_clinical'),('M3_observation_LR','M2_routine_LR'),('A3_M2_without_alb','M2_routine_LR'),('A4_M3_without_alb_process','M3_observation_LR'),('A6_M3_no_albumin_any','M3_observation_LR'),('B1_M3_binary_presence','M3_observation_LR'),('B2_M3_repeat_indicators','M3_observation_LR'),('B2_M3_repeat_indicators','M2_routine_LR'),('B2_M3_repeat_indicators','M1_clinical')]
it,di=paired_bootstrap(p,pairs);it.to_csv(O/'R1_fixed_model_intervals.csv',index=False);di.to_csv(O/'R1_fixed_paired_differences.csv',index=False)
# Verify B1's presence-only representation is equivalent to old missingness-only A1, apart from numerical optimization.
a=p[p.model=='A1_missing_only'].set_index(['cohort','sid']).p;b=p[p.model=='B1_M3_binary_presence'].set_index(['cohort','sid']).p
check={'B1_vs_A1_max_prediction_difference':float(np.max(np.abs(a.sort_index()-b.sort_index()))),'all_original_hashes_unchanged':True}
orig=json.loads((O/'original_frozen_hashes.json').read_text(encoding='utf8'))
for f,sha in orig.items():assert hashlib.sha256((V/f).read_bytes()).hexdigest()==sha
assert not any(x in sc.FEATURES['M3_observation_LR'] for x in ['CABG_any','y','future_hb_days'])
(O/'consolidation_checks.json').write_text(json.dumps(check,indent=2),encoding='utf8');print(json.dumps(check))

# Paired uncertainty for the already completed non-CABG redevelopment.
nc=pd.read_csv(V/'补充实验/nonCABG_predictions.csv.gz');nc=nc[nc.model.isin(['M0_Hb','M1_clinical','M2_routine_LR','M3_observation_LR','M4_routine_GB'])];it,di=paired_bootstrap(nc,[('M3_observation_LR','M2_routine_LR'),('M3_observation_LR','M1_clinical')]);it.to_csv(O/'nonCABG_redevelopment_intervals_R1.csv',index=False);di.to_csv(O/'nonCABG_redevelopment_paired_R1.csv',index=False)
