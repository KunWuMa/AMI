from pathlib import Path
import pandas as pd,numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
D=Path(__file__).resolve().parents[2];R=D/'aggregate_results/reliability_diagnostics';O=D/'docs/reliability_diagnostics';styles=getSampleStyleSheet();styles['Normal'].fontSize=9;styles['Normal'].leading=12;styles['Heading1'].fontSize=15;styles['Heading2'].fontSize=11
from xml.sax.saxutils import escape
story=[]
def para(t,sty='Normal'):story.append(Paragraph(escape(str(t)),styles[sty]));story.append(Spacer(1,6))
def title(t):para(t,'Heading1')
def table(headers,rows,widths):
 def cell(x):return Paragraph(escape(str(x)),styles['Normal'])
 tab=Table([[cell(x) for x in headers]]+[[cell(x) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#456070')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(tab);story.append(Spacer(1,9))
def read(n):return pd.read_csv(R/n)
def fmt(x):return 'NA' if pd.isna(x) else f'{x:.3f}'
def label(f):
 names={'last_margin':'Hb margin (g/L)','last_hb':'Latest early Hb (g/L)','early_min_hb':'Minimum early Hb (g/L)','calendar_LOS':'Calendar length of stay','male':'Male','age':'Age (years)'}
 return names.get(f,f.replace('risk_','Probability: ').replace('last_','Latest ').replace('available_','Available: ').replace('two_days_','Both days: ').replace('n_days_','Observation days: ').replace('missing_','Missing: '))
def summarize(r):
 if r['type']=='binary':return f"{int(r.positive)}/{int(r.denominator_observed)} ({r.percent_observed:.1f}%)"
 return f"{r['median']:.2f} ({r.q25:.2f}, {r.q75:.2f})" if pd.notna(r['median']) else 'NA'
def comparison(df,features,heading):
 para(heading,'Heading2');groups=list(df.group.unique());a=df[df.group.eq(groups[0])].set_index('variable');b=df[df.group.eq(groups[1])].set_index('variable');rows=[]
 for f in features:
  if f not in a.index:continue
  x,y=a.loc[f],b.loc[f];rows.append([label(f),summarize(x),summarize(y),str(int(x['missing']))+'/'+str(int(y['missing'])),fmt(x.SMD_A_minus_B)])
 table(['Variable',groups[0]+' (N='+str(int(a.N.iloc[0]))+')',groups[1]+' (N='+str(int(b.N.iloc[0]))+')','Missing A/B','SMD'],rows,[148,123,123,65,50])
 para('Values are median (Q1, Q3) or n/observed N (%). SMD is first group minus second; NA denotes unavailable or undefined. Full distributions and missing counts are in the companion aggregate CSV files.')
base=['age','male','last_margin','last_hb','early_min_hb','calendar_LOS']+['last_'+x for x in ['cr','alb','wbc','platelet','na','k','glucose']]+['available_'+x for x in ['hb','cr','alb','wbc','platelet','na','k','glucose']]
title('Post-hoc supplementary analyses: S19-S23');para('AMI anemia prediction across clinical settings | 5 November 2026');para('Independent supplement addition. Primary models and predictions remain frozen. The current manuscript, existing Supplement and main figures have not been replaced. Full methods and machine-readable aggregate results accompany this document.');para('Contents: S19 cohort selection; S20 future-Hb observation and opportunity; S21 CABG case mix and observation patterns; S22 reference probabilities and calibration; S23 frozen M3 parameters and contributions.');para('Analysis population: 280 local primary-AMI index patients, 116 eligible, 100 outcome-observed (46 events); 1,934 public eligible, 1,846 outcome-observed. Public held-out test: 370 patients (168 events).');story.append(PageBreak())
title('Table S19. Local cohort selection');comparison(read('E1_cohort_selection_summary.csv'),base,'Eligible versus ineligible index patients');q=read('E1_exclusion_reasons.csv');table(['Eligibility / exclusion reason','Patients'],q.values.tolist(),[429,80]);para('Early Hb differences are partly determined by the eligibility rule and do not alone demonstrate selection bias. All groups use unique patients.');story.append(PageBreak())
title('Table S20. Future Hb observed versus unobserved');q=read('E1_future_Hb_observation_groups.csv');features=base+['n_days_hb','n_days_alb']+['risk_'+x for x in ['M1_clinical','M2_routine_LR','M3_observation_LR','O3']]
for setting in ['Local','Public']:
 comparison(q[q.analysis.eq(setting)],features,setting);para('Conditional anemia outcomes are unknown without future Hb. Risk distributions do not establish their true anemia status. Full public cohort results include development/calibration patients.');story.append(PageBreak())
para('S20 supporting panel. Opportunity for future Hb observation','Heading1');q=read('E5_future_Hb_opportunity.csv')
for group in ['All eligible']:
 rows=[];z=q[q.group.eq(group)];a=z[z.cohort.eq('Public')].set_index('variable');b=z[z.cohort.eq('Local')].set_index('variable')
 for f in a.index:
  vals=[]
  for r in [a.loc[f],b.loc[f]]:vals.append(f"{int(r.positive)}/{int(r.N)} ({r.percent_observed:.1f}%)" if pd.notna(r.get('positive')) else f"{r['median']:.1f} ({r.q25:.1f}, {r.q75:.1f})")
  rows.append([f.replace('_',' ')]+vals)
 para(group,'Heading2');table(['Measure','Public','Local'],rows,[279,115,115])
para('Remaining days are calendar-date differences, not exact elapsed time. Potential D2-D7 dates include the discharge date (public: also truncated at death). Public uses specimen dates; local uses report dates. Counts are retained interpretable records, not proven distinct blood draws.');story.append(PageBreak())
title('Table S21A. Public CABG case mix');q=read('E2_public_CABG_case_mix.csv')
for setting in ['Public_all_observed','Public_test']:
 comparison(q[q.analysis.eq(setting)],['age','male','last_margin','last_alb']+['available_'+x for x in ['hb','cr','alb','wbc','platelet','na','k','glucose']]+['n_days_'+x for x in ['hb','cr','alb','wbc','platelet','na','k','glucose']]+['y'],setting)
 z=read('E2_public_CABG_prediction_summary.csv');z=z[z.cohort.eq(setting)];table(['Group','Model','Events/N','Mean risk','Brier'],[[r.group,r.model.split('_')[0],f'{r.events}/{r.N}',fmt(r.mean_prediction),fmt(r.Brier)] for r in z.itertuples()],[165,60,95,95,94]);para('Admission-wide CABG coding; descriptive association, not a causal comparison. Full-cohort summaries include training/calibration patients; only Public_test is held out.');story.append(PageBreak())
title('Table S21B. Cross-setting observation patterns');q=read('E2_cross_setting_observation_features.csv');comparison(q,['age','male','last_margin']+['last_'+x for x in ['cr','alb','wbc','platelet','na','k','glucose']]+['available_'+x for x in ['hb','cr','alb','wbc','platelet','na','k','glucose']]+['two_days_'+x for x in ['hb','cr','alb','wbc','platelet','na','k','glucose']],'Public test versus Local100');story.append(PageBreak());comparison(q,['n_days_'+x for x in ['hb','cr','alb','wbc','platelet','na','k','glucose']]+['missing_'+x for x in ['age','male','last_margin','last_cr','last_alb','last_wbc','last_platelet','last_na','last_k','last_glucose']],'S21B continued: day counts and missingness');para('Full 0/1/2-day frequencies are in the companion CSV. Only one local patient had CABG-name evidence; absence of this evidence does not establish CABG-negative status.');story.append(PageBreak())
title('Table S22. Probability reference and calibration');q=read('E3_probability_reference_calibration.csv')
for setting in ['Public_test','Local100']:
 z=q[q.cohort.eq(setting)];r=z.iloc[0];para(f'{setting}: N={r.n}, events={r.events}; reference probability={r.p_ref:.4f}; reference Brier={r.Brier_ref:.4f}.','Heading2');table(['Model','Mean risk','Expected events','E/O','Brier','BSS'],[[r.model.split('_')[0],fmt(r.mean_prediction),f'{r.expected_events:.2f}',fmt(r.expected_observed_ratio),fmt(r.Brier),fmt(r.BSS)] for r in z.itertuples()],[65,85,105,80,87,87]);rows=[]
 for _,r in z.iterrows():
  vals=[]
  for m in ['calibration_intercept','calibration_slope']:
   lo,hi=r[m+'_low'],r[m+'_high'];vals.append(fmt(r[m])+(f' ({lo:.3f}, {hi:.3f})' if pd.notna(lo) else ' (CI unavailable)'))
  rows.append([r.model.split('_')[0]]+vals)
 table(['Model','Calibration intercept (95% interval)','Calibration slope (95% interval)'],rows,[65,222,222])
q=read('E3_existing_paired_differences.csv');table(['Local M3 minus M2','Difference','95% paired interval'],[[r.metric,fmt(r.difference),f'({r.low:.3f}, {r.high:.3f})'] for r in q.itertuples()],[229,100,180]);para('The reference uses evaluation-sample prevalence post hoc; it is not a deployable model. Existing 1,000-patient-bootstrap intervals were reused. Local quintile groups contain approximately 20 patients; observed fractions are imprecise.');story.append(PageBreak())
title('Table S23A. Frozen M3 standardized coefficients');q=read('E4_frozen_M3_coefficients.csv');table(['Feature','Unit/encoding','Median imputation','Mean / scale','Raw beta'],[[r.feature,r.unit_or_encoding,fmt(r.training_imputation_median),fmt(r.training_scaler_mean)+' / '+fmt(r.training_scaler_scale),fmt(r.standardized_coefficient)] for r in q.itertuples()],[145,110,85,110,59]);z=read('E4_frozen_M3_calibration_transform.csv').iloc[0];para(f'Raw intercept b={z.raw_intercept:.8f}; public calibration a={z.calibration_intercept:.8f}, s={z.calibration_slope:.8f}. Final log odds = a + s logit(clip(expit(b + sum beta*z), 0.000001, 0.999999)). Exact parameters and feature definitions are provided in the CSV. Missingness was encoded before imputation.');story.append(PageBreak())
title('Table S23B. Additive prediction contributions');q=read('E4_prediction_contributions_by_setting.csv');q=q[q.component.isin(['Clinical/physiological','Observation days','Missingness','Intercept','Clipping correction'])]
for scale in q.scale.unique():
 para(scale,'Heading2');z=q[q.scale.eq(scale)];table(['Setting','Component','Mean','Median (Q1, Q3)'],[[r.cohort,r.component,fmt(r.mean),f'{r.median:.3f} ({r.q25:.3f}, {r.q75:.3f})'] for r in z.itertuples()],[95,150,65,199])
para('Group contributions, intercept and clipping correction reproduce the frozen prediction. Contributions are mathematical model associations, not causal mechanisms. Per-feature calibrated contributions are in the aggregate CSV.');story.append(PageBreak());title('Table S23C. Observation patterns and predicted risk');q=read('E4_observation_pattern_risk_summary.csv');table(['Setting','Pattern/value','N','Mean risk','Median (Q1, Q3)'],[[r.cohort,r.pattern+' = '+str(int(r.value)),r.N,fmt(r.mean),f'{r.median:.3f} ({r.q25:.3f}, {r.q75:.3f})'] for r in q.itertuples()],[90,155,40,75,149]);para('No subgroup AUROC or Brier estimate was calculated. In particular, the three local albumin-missing patients do not support stable subgroup performance inference.');story.append(PageBreak())
title('Supplementary Methods S2: additional analyses')
for line in (O/'Supplementary_Methods_S2_addition.md').read_text(encoding='utf8').splitlines():
 if line.strip():para(line.lstrip('# '),'Heading2' if line.startswith('#') else 'Normal')
def footer(c,d):c.setFont('Helvetica',8);c.setFillColor(colors.grey);c.drawString(43,26,'Independent post-hoc supplement | frozen primary models');c.drawRightString(552,26,str(d.page))
SimpleDocTemplate(str(O/'Supplement_addendum_S19-S23.pdf'),pagesize=(595.28,841.89),rightMargin=43,leftMargin=43,topMargin=40,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
print('Addendum PDF created')
