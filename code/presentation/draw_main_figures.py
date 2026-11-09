from pathlib import Path
import shutil,json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import proportion_confint
R=Path(__file__).resolve().parent;S=R.parent.parent;OLD=R.parent/'期刊精简版_20261015';O=R/'aggregate_results';G=R/'figures';FD=R/'figure_source_data';FD.mkdir(exist_ok=True)
C={'M0_Hb':'#999999','M1_clinical':'#333333','M2_routine_LR':'#0072B2','M3_observation_LR':'#D55E00','M4_routine_GB':'#AA4499'};M=list(C);labels=['M0: Hb margin','M1: clinical','M2: routine labs','M3: observation','M4: boosting']
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.2,'axes.linewidth':.55,'xtick.major.width':.55,'ytick.major.width':.55,'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white','text.color':'#222222','axes.labelcolor':'#222222'})
manifest=[]
def save(fig,name):
 fig.canvas.draw()
 for ext in ['pdf','svg','png']:fig.savefig(G/(name+'.'+ext),dpi=450,bbox_inches='tight',pad_inches=.045)
 manifest.append({'figure':name,'formats':['pdf','svg','png'],'size_inches':list(fig.get_size_inches())});plt.close(fig)
def tag(a,s,x=-.17,y=1.06):a.text(x,y,s,transform=a.transAxes,fontsize=8,fontweight='bold',va='bottom')
def clean(a):a.tick_params(direction='out');a.spines['bottom'].set_position(('outward',2));a.spines['left'].set_position(('outward',2))
# Figure 1: denominator hierarchy, including the explicitly different joint record event.
fig,ax=plt.subplots(figsize=(6.30,3.65));fig.subplots_adjust(left=.01,right=.995,bottom=.01,top=.99);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.01,.98,'a',fontsize=8,fontweight='bold');ax.text(.045,.98,'Population and endpoint denominators',fontweight='bold')
xs=[.025,.355,.685];names=['Public AMI','Selected local AMI'];flow=pd.read_csv(O/'cohort_flow.csv').set_index('cohort');counts=[tuple(int(flow.loc[k,f]) for f in ['at_D2','normal_early_eligible','observed','events']) for k in ['MIMIC_AMI','CAD_AMI']]
for x,name,nums,col in zip(xs,names,counts,['#0072B2','#D55E00','#777777']):
 n,e,o,a=nums;ax.text(x,.91,name,fontweight='bold',fontsize=6.8);ax.plot([x,x+.29],[.89,.89],color=col,lw=1.4)
 for y,title,text in [( .72,'At calendar D2',f'{n:,} primary-diagnosis patients'),(.53,'Early-normal eligible',f'E = {e:,}; future Hb O = {o:,} ({100*o/e:.1f}%)'),(.34,'Recorded anemia',f'A | O: {a:,}/{o:,} ({100*a/o:.1f}%)\nJoint J: {a:,}/{e:,} ({100*a/e:.1f}%)')]:
  ax.add_patch(Rectangle((x,y),.29,.125,fc='#F6F7F8',ec='#B8BEC3',lw=.55));ax.text(x+.01,y+.09,title,fontweight='bold',fontsize=6.6);ax.text(x+.01,y+.039,text,fontsize=6.0,va='center',linespacing=1.3)
 for y1,y2 in [(.72,.655),(.53,.465)]:ax.annotate('',xy=(x+.145,y2),xytext=(x+.145,y1),arrowprops={'arrowstyle':'->','lw':.6,'color':'#555555'})
 ax.text(x,.294,f'No future Hb: {e-o}; A not evaluated',fontsize=6.2)
ax.text(.685,.91,'Three distinct record targets',fontweight='bold',fontsize=6.8)
ax.plot([.685,.975],[.89,.89],color='#777777',lw=1.4)
for yy,title,txt in [(.72,'Observation O','Future Hb present in the record'),(.53,'Conditional A | O','Recorded anemia among O = 1'),(.34,'Joint event J','Recorded anemia in all eligible')]:
 ax.add_patch(Rectangle((.685,yy),.29,.125,fc='#F6F7F8',ec='#B8BEC3',lw=.55));ax.text(.695,yy+.09,title,fontweight='bold',fontsize=6.6);ax.text(.695,yy+.039,txt,fontsize=6.0,va='center')
ax.text(.685,.294,'No future Hb does not imply normal Hb',fontsize=6.0)
ax.text(.01,.225,'b',fontsize=8,fontweight='bold');ax.text(.045,.225,'Known early information and routine-care follow-up',fontweight='bold')
ax.add_patch(Rectangle((.055,.095),.255,.07,fc='#E8F1F6',ec='none'));ax.add_patch(Rectangle((.31,.095),.615,.07,fc='#F9EDE6',ec='none'));ax.plot([.055,.925],[.09,.09],color='#555555',lw=.65)
for x,t in [(.055,'D0'),(.18,'D1'),(.31,'D2'),(.925,'D7')]:ax.plot([x,x],[.087,.074],color='#555555',lw=.55);ax.text(x,.05,t,ha='center',fontsize=6.5)
ax.plot([.31,.31],[.08,.185],color='#333333',lw=.65);ax.text(.31,.188,'Predict',ha='center',fontsize=6.3);ax.text(.18,.128,'Known predictors',ha='center',va='center',fontsize=6.3);ax.text(.62,.128,'Future Hb observation and recorded threshold crossing',ha='center',va='center',fontsize=6.2);save(fig,'Figure1_estimand_design')
# Figure 2: ranking and probability error with interval comparisons, followed by calibration.
met=pd.read_csv(O/'conditional_fixed_model_metrics.csv');its=pd.read_csv(O/'conditional_fixed_model_intervals.csv');bins=pd.read_csv(FD/'calibration_M1_M2_M3.csv')
fig,axs=plt.subplots(2,2,figsize=(6.80,4.55));fig.subplots_adjust(left=.16,right=.98,bottom=.10,top=.91,wspace=.42,hspace=.64)
for ax,metric,title,lim in [(axs[0,0],'auc','Discrimination',(.5,1.0)),(axs[0,1],'Brier','Probability error',(.13,.33))]:
 for j,m in enumerate(M):
  for cohort,shift,mk in [('MIMIC_test',.14,'o'),('CAD_AMI',-.14,'D')]:
   v=its[its.cohort.eq(cohort)&its.model.eq(m)&its.metric.eq(metric)].iloc[0]
   ax.errorbar(v.point,4-j+shift,xerr=[[v.point-v.low],[v.high-v.point]],fmt=mk,ms=3.5,lw=.85,capsize=1.5,color=C[m],mfc='white' if cohort=='MIMIC_test' else C[m])
 ax.set_yticks([4,3,2,1,0],['M0: Hb margin','M1: + age/sex','M2: routine labs','M3: + observation','M4: boosting']);ax.set(xlim=lim,ylim=(-.5,4.5),xlabel='AUROC (95% interval)' if metric=='auc' else 'Brier (95% interval)',title=title);clean(ax)
axs[0,1].set_yticklabels(['M0','M1','M2','M3','M4']);tag(axs[0,0],'a',x=-.38);tag(axs[0,1],'b',x=-.2)
fig.legend([Line2D([0],[0],marker='o',mfc='white',color='#333333',ls='',ms=4),Line2D([0],[0],marker='D',color='#333333',ls='',ms=4)],['Public test: n = 370','Selected local AMI: n = 100'],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.56,1.01),columnspacing=2)
for j,(cohort,title) in enumerate([('MIMIC_test','Public test calibration'),('CAD_AMI','Selected local calibration')]):
 ax=axs[1,j]
 for m in M[1:4]:
  v=bins[bins.cohort.eq(cohort)&bins.model.eq(m)];ax.errorbar(v.prediction,v.observed,yerr=[v.observed-v.low,v.high-v.observed],fmt='o-',ms=3,lw=.7,elinewidth=.6,capsize=1.5,color=C[m],label=m[:2])
 ax.plot([0,1],[0,1],color='#888888',lw=.65,ls=(0,(3,3)),zorder=0);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Recorded event fraction',title=title);ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6));clean(ax);tag(ax,chr(99+j),x=-.38 if j==0 else -.2)
axs[1,0].legend(frameon=False,ncol=3,loc='upper left',handlelength=1.3,columnspacing=.8);save(fig,'Figure2_frozen_performance')
# Supplementary S1/S2: public-test and final-local curves and calibration.
roc=pd.read_csv(FD/'ROC_coordinates.csv');groups=[('MIMIC_test','Public test (n = 370)'),('CAD_AMI','Selected local AMI (n = 100)')]
for stem,calibration in [('FigureS1_ROC',False),('FigureS2_calibration',True)]:
 fig,axs=plt.subplots(1,2,figsize=(6.80,3.0));fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.80,wspace=.42)
 for j,(cohort,title) in enumerate(groups):
  ax=axs[j]
  for m in M[1:4] if calibration else M:
   if calibration:
    v=bins[bins.cohort.eq(cohort)&bins.model.eq(m)];ax.errorbar(v.prediction,v.observed,yerr=[v.observed-v.low,v.high-v.observed],fmt='o-',ms=2.7,lw=.7,capsize=1,color=C[m],label=m[:2])
   else:
    v=roc[roc.cohort.eq(cohort)&roc.model.eq(m)];ax.plot(v.fpr,v.tpr,color=C[m],lw=1,label=m[:2])
  ax.plot([0,1],[0,1],color='#888888',lw=.6,ls=(0,(3,3)),zorder=0);ax.set(xlim=(0,1),ylim=(0,1),title=title,xlabel='Mean prediction' if calibration else 'False-positive rate',ylabel=('Recorded fraction' if calibration else 'True-positive rate') if j==0 else None);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);clean(ax);tag(ax,chr(97+j),x=-.22)
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=len(l),frameon=False,loc='upper center',bbox_to_anchor=(.55,1.0),handlelength=1.4,columnspacing=1.4);save(fig,stem)
# Figure 4: availability, cross-setting mean errors and ranking, and paired local probability error.
met=pd.read_csv(O/'conditional_fixed_model_metrics.csv');its=pd.read_csv(O/'conditional_fixed_model_intervals.csv');dif=pd.read_csv(O/'conditional_fixed_paired_differences.csv');models=['M2_routine_LR','M3_observation_LR','A4_M3_without_alb_process','B2_M3_repeat_indicators'];short=['M2','M3','A4','B2'];ys=np.arange(4)[::-1]
fig=plt.figure(figsize=(6.80,4.85));a=fig.add_axes([.255,.805,.72,.115]);b=fig.add_axes([.30,.375,.27,.285]);c=fig.add_axes([.72,.375,.255,.285]);d=fig.add_axes([.39,.078,.585,.15]);avail=[]
for j,(_,availability) in enumerate(pd.read_csv(FD/'albumin_availability_counts.csv').iterrows()):
 nn=int(availability.n);measured=int(availability.albumin_measured);label=availability.cohort;col=[C[M[2]],C[M[3]]][j]
 y=1-j;p=100*measured/nn;a.barh(y,100,height=.58,color='#E8EEF2');a.barh(y,p,height=.58,color=col)
 a.text(p/2,y,f'{p:.1f}% measured',color='white',ha='center',va='center',fontsize=6.3,fontweight='bold')
 if j==0:a.text(p+(100-p)/2,y,f'{100-p:.1f}% missing',ha='center',va='center',fontsize=6.3,color='#4C5962')
 else:a.text(99,y+.43,'3.0% missing',ha='right',va='center',fontsize=6.1,color='#555555')
 avail.append({'cohort':label,'n':nn,'albumin_measured':measured,'albumin_missing':nn-measured})
a.set_yticks([1,0],['Public AMI\n454/1,846 measured','Selected local AMI\n97/100 measured']);a.tick_params(axis='y',length=0,pad=7);a.set(xlim=(0,100),ylim=(-.55,1.65));a.set_xticks([0,25,50,75,100],['0','25','50','75','100%']);a.spines['left'].set_visible(False);clean(a)
fig.text(.025,.965,'a',fontsize=8,fontweight='bold');fig.text(.255,.965,'Early albumin availability differs sharply by setting',fontweight='bold');fig.text(.255,.94,'All outcome-observed AMI; descriptive counts',fontsize=6.2,color='#555555')
fig.legend([Line2D([0],[0],marker='o',ls='',color=C[M[2]],ms=4),Line2D([0],[0],marker='D',ls='',color=C[M[3]],ms=4)],['Public test: 370 patients / 168 events','Selected local: 100 patients / 46 events'],ncol=2,frameon=False,loc='center',bbox_to_anchor=(.55,.728),columnspacing=1.4,handletextpad=.5,fontsize=6.2)
fig.text(.025,.687,'b',fontsize=8,fontweight='bold');fig.text(.30,.687,'Average risk alignment',fontweight='bold');fig.text(.675,.687,'c',fontsize=8,fontweight='bold');fig.text(.72,.687,'Ranking with uncertainty',fontweight='bold')
descriptions=['M2  Routine labs','M3  + observation inputs','A4  M3 - albumin observation','B2  Two-day indicators'];performance_rows=[]
for y,m,description in zip(ys,models,descriptions):
 if m==M[3]:
  for ax in [b,c]:ax.axhspan(y-.42,y+.42,color='#FBF0E9',zorder=0)
 fy=.375+.285*(y+.5)/4;fig.text(.025,fy,description,va='center',fontsize=6.25,fontweight='bold' if m==M[3] else 'normal')
 pair=[]
 for cohort,col,shift,mk in [('MIMIC_test',C[M[2]],.13,'o'),('CAD_AMI',C[M[3]],-.13,'D')]:
  v=met[met.cohort.eq(cohort)&met.model.eq(m)].iloc[0];bias=100*(v.mean_prediction-v.prevalence);it=its[its.cohort.eq(cohort)&its.model.eq(m)&its.metric.eq('auc')].iloc[0]
  b.plot(bias,y+shift,mk,ms=3.6,color=col,zorder=3);c.errorbar(v.auc,y+shift,xerr=[[v.auc-it.low],[it.high-v.auc]],fmt=mk,ms=3.2,lw=.8,capsize=1.5,color=col,zorder=3)
  if cohort=='CAD_AMI':b.text(bias+.95,y+shift,f'{bias:+.1f}',va='center',fontsize=6.0,color=col)
  pair.append((bias,y+shift));performance_rows.append({'cohort':cohort,'model':m,'n':int(v.n),'events':int(v.events),'mean_prediction':v.mean_prediction,'prevalence':v.prevalence,'mean_error_pp':bias,'auc':v.auc,'auc_low':it.low,'auc_high':it.high,'Brier':v.Brier})
 b.plot([z[0] for z in pair],[z[1] for z in pair],color='#C4CBD0',lw=.65,zorder=1)
for ax in [b,c]:ax.set(ylim=(-.5,3.5));ax.set_yticks([]);ax.spines['left'].set_visible(False);clean(ax)
b.axvline(0,color='#888888',ls=(0,(3,3)),lw=.7);b.set(xlim=(-6,27),xlabel='Mean prediction - event fraction (pp)');b.set_xticks([-5,0,10,20]);b.xaxis.label.set_fontsize(6.3)
c.set(xlim=(.55,.99),xlabel='AUROC (95% interval)');c.set_xticks([.6,.7,.8,.9]);c.xaxis.label.set_fontsize(6.3)
fig.text(.025,.28,'d',fontsize=8,fontweight='bold');fig.text(.39,.28,'Paired local Brier changes versus M3',fontweight='bold');cd=[]
loc=met[met.cohort.eq('CAD_AMI')].set_index('model')
for j,(m,lab) in enumerate(zip(models[2:],short[2:])):
 z=dif[dif.cohort.eq('CAD_AMI')&dif.model.eq(m)&dif.baseline.eq(M[3])&dif.metric.eq('delta_Brier')].iloc[0];y=1-j
 d.errorbar(z.difference,y,xerr=[[z.difference-z.low],[z.high-z.difference]],fmt='D',color=C[M[3]],ms=4,lw=1.0,capsize=2)
 fy=.078+.15*(y+.5)/2;fig.text(.025,fy,f'{lab}   Brier {loc.loc[M[3],"Brier"]:.3f} to {loc.loc[m,"Brier"]:.3f}',va='center',fontsize=6.4)
 d.text(z.difference,y+.20,f'{z.difference:+.3f}',ha='center',fontsize=6.0,color=C[M[3]]);cd.append({'label':lab,**z.to_dict()})
d.axvline(0,color='#888888',ls=(0,(3,3)),lw=.7);d.set_yticks([]);d.spines['left'].set_visible(False);d.set(xlim=(-.14,.025),ylim=(-.5,1.5),xlabel='Brier difference (negative favors alternative; paired 95% interval)');d.xaxis.label.set_fontsize(6.3);d.set_xticks([-.125,-.10,-.075,-.05,-.025,0]);clean(d)
save(fig,'Figure4_albumin_representation');pd.DataFrame(cd).to_csv(FD/'main_representation_Brier_plot.csv',index=False);pd.DataFrame(avail).to_csv(FD/'albumin_availability_counts.csv',index=False);pd.DataFrame(performance_rows).to_csv(FD/'main_representation_performance.csv',index=False)

print('Main figures and complete ROC/calibration figures generated from aggregate data.')
