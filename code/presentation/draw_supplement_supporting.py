from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R;O=R/'aggregate_results';A=O;G=P/'figures';FD=P/'figure_source_data';M=pd.read_csv(O/'conditional_metrics.csv');I=pd.read_csv(O/'conditional_intervals.csv');D=pd.read_csv(O/'conditional_paired.csv');COL={'M0_Hb':'#999999','M1_clinical':'#333333','M2_routine_LR':'#0072B2','M3_observation_LR':'#D55E00','M4_routine_GB':'#AA4499'}
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
def savefig(fig,n):
 for ext in ['pdf','svg','png']:fig.savefig(G/(n+'.'+ext),bbox_inches='tight',dpi=350,pad_inches=.06)
 plt.close(fig)
def fmt(x,k=3):return '--' if pd.isna(x) else f'{float(x):.{k}f}'
def label(m):return str(m).split('_')[0]
def esc(s):return str(s).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
def get(df,**kw):
 for k,v in kw.items():df=df[df[k].eq(v)]
 assert len(df)==1,(kw,len(df));return df.iloc[0]
# S4: same-patient paired increment for selected sensitivity definitions.
sd=pd.read_csv(O/'sensitivity_paired.csv');names=['y_daily_median','y_last_day','y_lt110','y_drop20','normal_margin_ge10','exclude_recorded_CABG_name','exclude_any_transfusion_evidence','recent_2024_2025'];labs=['Daily median Hb','Last observed Hb day','Hb <110 g/L','Decline >=20 g/L','Early margin >=10 g/L','Exclude CABG-name evidence','Exclude transfusion evidence','Local admissions 2024-2025'];z=sd[sd.model.eq('M3_observation_LR')&sd.baseline.eq('M1_clinical')&sd.metric.eq('delta_auc')].set_index('cohort').loc[names].reset_index();fig,ax=plt.subplots(figsize=(6.8,3.35));fig.subplots_adjust(left=.43,right=.96,bottom=.17,top=.90)
for j,v in z.iterrows():ax.errorbar(v.difference,len(z)-1-j,xerr=[[v.difference-v.low],[v.high-v.difference]],fmt='D',color=COL['M3_observation_LR'],capsize=2,ms=4,lw=.9)
ax.axvline(0,color='#888888',ls='--',lw=.6);ax.set(yticks=range(len(z)),yticklabels=labs[::-1],xlabel='Paired AUROC difference: M3 - M1 (95% interval)',title='Fixed-prediction sensitivities');savefig(fig,'FigureS3_sensitivity_increments');z.to_csv(FD/'sensitivity_increment_intervals.csv',index=False)
# S5: numerical outcome transport.
cm=pd.read_csv(O/'continuous_intervals.csv');fig,axs=plt.subplots(1,3,figsize=(6.8,2.45));fig.subplots_adjust(left=.09,right=.98,bottom=.24,top=.80,wspace=.38)
for ax,target,title in zip(axs,['target_nadir_Hb','target_decline','target_nadir_margin'],['Recorded nadir Hb','Signed Hb decline','Nadir margin']):
 for j,name in enumerate(['R_constant','M1_clinical','M2_routine_LR','M3_observation_LR']):
  v=get(cm,target=target,model=name,metric='MAE');ax.errorbar(v.point,3-j,xerr=[[v.point-v.low],[v.high-v.point]],fmt='o',color=COL.get(name,'#888888'),capsize=2,lw=.8,ms=3.5)
 ax.set(yticks=range(4),yticklabels=['R3','R2','R1','Constant'],xlabel='MAE, g/L (95% interval)',title=title,xlim=(0,27))
savefig(fig,'FigureS4_continuous_outcomes');cm.to_csv(FD/'continuous_error_intervals.csv',index=False)
# S6: the O and J endpoints, each with its own denominator.
tm=pd.read_csv(O/'two_stage_metrics.csv');fig,axs=plt.subplots(1,2,figsize=(6.8,2.6));fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.80,wspace=.36)
for ax,task,title in zip(axs,['O','J'],['Future Hb observed: 100/116','Joint recorded anemia: 46/116']):
 z=tm[tm.cohort.eq('Local_eligible116')&tm.task.eq(task)]
 for j,(_,v) in enumerate(z.iterrows()):ax.bar(j,100*v.mean_prediction,color=COL[v.model],width=.58);ax.text(j,100*v.mean_prediction+1,f'{100*v.mean_prediction:.1f}%',ha='center',fontsize=6.5)
 ax.axhline(100*z.prevalence.iloc[0],color='#333333',ls='--',lw=.7,label='Observed record fraction');ax.set(xticks=range(3),xticklabels=[task+'1',task+'2',task+'3'],ylim=(0,105),ylabel='Probability (%)',title=title);ax.legend(frameon=False,loc='lower right')
savefig(fig,'FigureS5_observation_joint');tm[tm.cohort.eq('Local_eligible116')].to_csv(FD/'observation_joint_probabilities.csv',index=False)
