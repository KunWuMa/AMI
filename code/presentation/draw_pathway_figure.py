from pathlib import Path
import shutil,json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
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
# Figures 3--4 only. This script needs aggregate CSVs only; no records, predictions or fitted models.
comp=pd.read_csv(FD/'pathway_anemia_bar_counts.csv');inc=pd.read_csv(FD/'pathway_increment_plot.csv');full=frozen=redevelop=inc
levels=[]
for _,z in inc.iterrows():
 for m,key in [('M2_routine_LR','M2_auc'),('M3_observation_LR','M3_auc')]:levels.append({'cohort':z.cohort,'model':m,'auc':z[key]})
met=nm=rm=pd.DataFrame(levels)
fig=plt.figure(figsize=(6.80,4.45))
a=fig.add_axes([.215,.60,.375,.285]);b=fig.add_axes([.755,.60,.22,.285])
paths=['CABG','PCI_without_CABG','no_CABG_or_PCI_code'];pnames=['CABG','PCI without CABG','Neither CABG nor PCI code'];pathrows=[]
for j,(path,label) in enumerate(zip(paths,pnames)):
 z=comp[comp.population.eq('all_observed')&comp.pathway.eq(path)].iloc[0];y=2-j;p=100*z.events/z.n
 a.barh(y,100,height=.55,color='#E8EEF2',zorder=0);a.barh(y,p,height=.55,color=C[M[2]],zorder=1)
 a.text(p/2,y,f'{p:.1f}%',color='white',ha='center',va='center',fontsize=6.5,fontweight='bold')
 pathrows.append({'population':'all_observed','pathway':path,'n':int(z.n),'events':int(z.events),'fraction':p/100})
a.set_yticks([2,1,0],[f'{lab}\n{z["events"]:,}/{z["n"]:,} events' for lab,z in zip(pnames,pathrows)]);a.tick_params(axis='y',length=0,pad=7);a.set(xlim=(0,100),ylim=(-.65,2.65),xlabel='Patients (%)');a.set_xticks([0,25,50,75,100]);a.spines['left'].set_visible(False);clean(a)
fig.text(.025,.96,'a',fontsize=8,fontweight='bold');fig.text(.215,.96,'Recorded anemia by procedure pathway',fontweight='bold');fig.text(.215,.927,'All public outcome-observed AMI (n = 1,846)',fontsize=6.3,color='#555555')
a.legend([Rectangle((0,0),1,1,fc=C[M[2]]),Rectangle((0,0),1,1,fc='#E8EEF2')],['Recorded anemia','No recorded event'],frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.02,-.44),columnspacing=1,handlelength=1.1,fontsize=6.0)
# The test denominator is kept separate from the full public population.
share=[]
for y,(_,share_row) in zip([1,0],pd.read_csv(FD/'pathway_test_event_shares.csv').iterrows()):
 num=int(share_row.CABG_n);den=int(share_row.all_n);label=share_row.denominator
 p=100*num/den;b.barh(y,100,height=.49,color='#8B969E');b.barh(y,p,height=.49,color=C[M[3]])
 b.text(p/2,y,f'{p:.1f}%',ha='center',va='center',color='white',fontsize=6.3,fontweight='bold');b.text(p+(100-p)/2,y,f'{100-p:.1f}%',ha='center',va='center',color='white',fontsize=6.3)
 share.append({'population':'test_observed','denominator':label,'CABG_n':num,'all_n':den,'CABG_fraction':num/den})
b.set_yticks([1,0],['Patients\n80/370 CABG','Events\n78/168 CABG']);b.tick_params(axis='y',length=0,pad=6);b.set(xlim=(0,100),ylim=(-.65,1.65),xlabel='Composition (%)');b.set_xticks([0,50,100]);b.spines['left'].set_visible(False);clean(b)
fig.text(.66,.96,'b',fontsize=8,fontweight='bold');fig.text(.755,.96,'CABG event contribution',fontweight='bold');fig.text(.755,.927,'Full public test sample',fontsize=6.3,color='#555555')
b.legend([Rectangle((0,0),1,1,fc=C[M[3]]),Rectangle((0,0),1,1,fc='#8B969E')],['CABG','Non-CABG'],frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.11,-.44),columnspacing=1,handlelength=1.1,fontsize=6.0)
# Aligned rows distinguish model restriction from redevelopment, and levels from paired gains.
lv=fig.add_axes([.34,.11,.225,.275]);dc=fig.add_axes([.715,.11,.25,.275]);rows=[]
settings=[(full,'MIMIC_test','Full test','Primary full-cohort models',370,168,met),(frozen,'Public_nonCABG_frozen','Non-CABG test','Primary full-cohort models',290,90,nm),(redevelop,'MIMIC_test_no_CABG','Non-CABG test','Redeveloped models',290,90,rm)]
for j,(df,cohort,label,fit,nn,ev,mdf) in enumerate(settings):
 z=df[df.cohort.eq(cohort)&df.model.eq(M[3])&df.baseline.eq(M[2])&df.metric.eq('delta_auc')].iloc[0];v=mdf[mdf.cohort.eq(cohort)]
 if 'version' in v.columns:v=v[v.version.eq('public_calibrated')]
 v=v.set_index('model');x2=float(v.loc[M[2],'auc']);x3=float(v.loc[M[3],'auc']);y=2-j
 lv.plot([x2,x3],[y,y],color='#B6BEC4',lw=1.5,zorder=1)
 for xx,col,dy,mk in [(x2,C[M[2]],.13,'o'),(x3,C[M[3]],-.13,'D')]:
  lv.plot(xx,y,mk,color=col,ms=4,zorder=3);lv.text(xx,y+dy,f'{xx+1e-12:.3f}',ha='center',va='bottom' if dy>0 else 'top',fontsize=6.0,color=col)
 dc.errorbar(z.difference,y,xerr=[[z.difference-z.low],[z.high-z.difference]],fmt='D',ms=4,lw=1,capsize=2,color=C[M[3]])
 dc.text(z.difference,y+.23,f'{z.difference:+.3f}',ha='center',fontsize=6.2,color=C[M[3]])
 fy=.11+.275*(y+.5)/3;fig.text(.025,fy+.009,label,fontsize=6.6,fontweight='bold',va='center');fig.text(.025,fy-.022,f'{fit}\nn = {nn}, events = {ev}',fontsize=5.8,color='#555555',va='top')
 rows.append({'comparison':label+' / '+fit,'n':nn,'events':ev,'M2_auc':x2,'M3_auc':x3,**z.to_dict()})
for ax in [lv,dc]:ax.set(ylim=(-.5,2.5));ax.set_yticks([]);ax.spines['left'].set_visible(False);clean(ax)
lv.set(xlim=(.64,.85),xlabel='AUROC');lv.set_xticks([.65,.70,.75,.80,.85]);dc.set(xlim=(-.08,.15),xlabel='AUROC difference (M3 - M2)');dc.set_xticks([-.05,0,.05,.10]);dc.axvline(0,color='#777777',lw=.7,ls=(0,(3,3)),zorder=0)
fig.text(.025,.455,'c',fontsize=8,fontweight='bold');fig.text(.065,.455,'Observation-feature increment across case compositions',fontweight='bold')
fig.legend([Line2D([0],[0],marker='o',ls='',color=C[M[2]],ms=4),Line2D([0],[0],marker='D',ls='',color=C[M[3]],ms=4)],['M2: routine labs','M3: + observation inputs'],frameon=False,ncol=2,loc='center',bbox_to_anchor=(.455,.427),columnspacing=1.2,handletextpad=.5,fontsize=6.1)
fig.text(.84,.425,'Paired 95% intervals',ha='center',fontsize=6.1,color='#555555');save(fig,'Figure3_pathway_dependence')
pd.DataFrame(rows).to_csv(FD/'pathway_increment_plot.csv',index=False);pd.DataFrame(pathrows).to_csv(FD/'pathway_anemia_bar_counts.csv',index=False);pd.DataFrame(share).to_csv(FD/'pathway_test_event_shares.csv',index=False)

print('Pathway figure generated from aggregate CSVs.')
