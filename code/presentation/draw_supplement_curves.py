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

bins=pd.read_csv(FD/'calibration_M1_M2_M3.csv')
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
