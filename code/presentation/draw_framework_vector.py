from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Ellipse,Circle,Arc,FancyArrowPatch
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
f,ax=plt.subplots(figsize=(9,6.75));f.subplots_adjust(0,0,1,1);ax.set(xlim=(0,12),ylim=(0,9));ax.axis('off')
B='#245E87';T='#087F82';O='#B66D2B';I='#143047';G='#536773'
def txt(x,y,s,size=10,color=I,bold=False,ha='left',va='top'):
 ax.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',ha=ha,va=va,linespacing=1.5)
def box(x,y,w,h,col=B,fill='white',lw=.9):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.08',fc=fill,ec=col,lw=lw))
def line(x,y,x2,y2,col=B,lw=1):ax.plot([x,x2],[y,y2],color=col,lw=lw)
def arrow(x,y,x2,y2,col=B):ax.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=14,color=col,lw=1.5))
def icon(kind,x,y,s,c):
 if kind=='hospital':
  ax.add_patch(Rectangle((x+s*.24,y),s*.52,s*.88,fc='white',ec=c,lw=1.5));ax.add_patch(Rectangle((x,y),s*.24,s*.58,fc='white',ec=c,lw=1.5));ax.add_patch(Rectangle((x+s*.76,y),s*.24,s*.58,fc='white',ec=c,lw=1.5))
  line(x+s*.5,y+s*.63,x+s*.5,y+s*.81,c,2);line(x+s*.4,y+s*.72,x+s*.6,y+s*.72,c,2)
  for xx in [.34,.59]:
   for yy in [.17,.36]:ax.add_patch(Rectangle((x+s*xx,y+s*yy),s*.09,s*.1,fc=c,ec='none'))
 elif kind=='db':
  ax.add_patch(Rectangle((x,y+s*.16),s*.85,s*.58,fc=c,ec=c));ax.add_patch(Ellipse((x+s*.425,y+s*.16),s*.85,s*.3,fc=c,ec=c));ax.add_patch(Ellipse((x+s*.425,y+s*.74),s*.85,s*.3,fc=c,ec='white',lw=1))
  for yy in [.34,.53]:ax.add_patch(Arc((x+s*.425,y+s*yy),s*.85,s*.3,theta1=180,theta2=360,color='white',lw=1))
 elif kind=='lock':
  ax.add_patch(Arc((x+s*.5,y+s*.63),s*.5,s*.65,theta1=0,theta2=180,lw=3,color=c));box(x+s*.12,y,s*.76,s*.55,c,c);ax.add_patch(Circle((x+s*.5,y+s*.34),s*.075,fc='white'));line(x+s*.5,y+s*.28,x+s*.5,y+s*.14,'white',2.5)
 elif kind=='calendar':
  box(x,y,s*.8,s*.72,c,'white');line(x,y+s*.51,x+s*.8,y+s*.51,c,1.2)
  for xx in [.2,.6]:line(x+s*xx,y+s*.63,x+s*xx,y+s*.86,c,2)
  for xx in [.18,.46]:
   for yy in [.14,.32]:ax.add_patch(Rectangle((x+s*xx,y+s*yy),s*.13,s*.1,fc=c,ec='none'))
 elif kind=='search':
  ax.add_patch(Circle((x+s*.55,y+s*.55),s*.32,fill=False,ec=c,lw=2));line(x+s*.3,y+s*.3,x,y,c,3)
 elif kind=='tubes':
  for xx in [0,.44]:
   box(x+s*xx,y+s*.1,s*.24,s*.7,c,'white');line(x+s*(xx-.04),y+s*.82,x+s*(xx+.28),y+s*.82,c,2);ax.add_patch(Rectangle((x+s*(xx+.05),y+s*.16),s*.14,s*.3,fc=c,ec='none'))
 elif kind=='document':
  box(x,y,s*.65,s*.85,c,'white')
  for yy in [.25,.45,.65]:line(x+s*.13,y+s*yy,x+s*.52,y+s*yy,c,1.7)
# a: task timeline. Icons indicate concepts; no empirical traces are drawn.
box(.12,6.55,11.76,2.31,B);txt(.3,8.72,'a',17,B,True);txt(.68,8.72,'Shared prediction task',15,I,True)
icon('hospital',.4,7.12,.9,B)
txt(1.64,8.13,'D0-1: early information',12,B,True);txt(1.64,7.79,'Adult primary AMI\nAll known early Hb normal',10.5)
txt(7.65,8.13,'D2-7: recorded anemia',12,O,True);txt(7.65,7.79,'Among patients with future Hb\nBefore discharge or recorded death',10.5)
line(1.7,7.13,5.94,7.13,B,2);arrow(6.45,7.13,11.5,7.13,O);ax.plot([6.18,6.18],[6.93,8.15],ls=(0,(3,3)),color=T,lw=1.5);ax.add_patch(Circle((6.18,7.13),.24,fc='white',ec=T,lw=1.2));icon('lock',6.04,7.01,.28,T)
txt(6.18,8.23,'Start of D2',12,T,True,ha='center');icon('calendar',3.45,6.72,.33,B);icon('calendar',9.36,6.72,.33,O)
txt(6.0,6.77,'No future Hb = unknown conditional label',10.3,I,ha='center')
# b: source training/calibration -> freezing -> two evaluations.
box(.12,2.88,11.76,3.49,B);txt(.3,6.23,'b',17,B,True);txt(.68,6.23,'Public development and frozen cross-setting evaluation',14,I,True)
box(.32,4.76,4.13,1.0,B,'#F2F7FB');icon('db',.58,4.94,.65,B);txt(1.42,5.63,'MIMIC-IV 2.2',12,B,True);txt(1.42,5.29,'Training n = 1,107\nCalibration n = 369',10.5)
box(.32,3.08,4.13,1.53,B,'#F8FAFC');txt(.48,4.49,'Three nested-feature models',10.8,B,True)
for y,label,body,col in [(3.95,'M1','Age + sex + Hb margin',B),(3.54,'M2','M1 + seven laboratory values',T),(3.13,'M3','M2 + observation days + missingness',O)]:
 box(.47,y,3.8,.34,col,'white',.7);txt(.58,y+.27,label,10,col,True);txt(.97,y+.27,body,9.0)
txt(.48,3.035,'Logistic regression',8.2,G)
box(5.15,3.30,1.9,2.15,T,'#F6FBFA');icon('lock',5.76,4.65,.62,T);txt(6.1,4.47,'FREEZE',12,T,True,ha='center');txt(6.1,4.1,'Preprocessing\nModel coefficients\nPublic calibration',10.2,I,ha='center')
arrow(4.5,4.47,5.08,4.47,B)
box(8.12,4.68,3.52,1.06,B,'#F1F6FA');icon('db',8.32,4.88,.6,B);txt(9.07,5.6,'Public test',12,B,True);txt(9.07,5.24,'370 observed / 168 events',10)
box(8.12,3.28,3.52,1.17,T,'#F0F8F6');icon('hospital',8.32,3.56,.58,T);txt(9.07,4.31,'Local hospital',12,T,True);txt(9.07,3.97,'100 observed / 46 events\n116 eligible; 16 unobserved',9.8)
line(7.1,4.66,7.6,4.66,B,1.5);line(7.6,4.66,7.6,5.2,B,1.5);arrow(7.6,5.2,8.08,5.2,B)
line(7.1,4.25,7.6,4.25,T,1.5);line(7.6,4.25,7.6,3.85,T,1.5);arrow(7.6,3.85,8.08,3.85,T)
# c: evidence hierarchy, not additional training stages.
box(.12,.14,11.76,2.56,B);txt(.3,2.56,'c',17,B,True);txt(.68,2.56,'Reliability assessment and explanatory analyses',14,I,True)
for x,title,kind,col,fill,body in [(.32,'Primary evaluation','search',B,'#F2F7FB','M3 vs M2; M1 benchmark\nDiscrimination: AUROC\nProbability error: Brier\nCalibration: level and slope'),(4.23,'Explanatory analyses','tubes',T,'#F0F8F6','Care composition: CABG / PCI\nObservation-feature encoding\nFuture Hb observation\nConditional vs joint event'),(8.14,'Supporting analyses','document',O,'#FBF5EE','Outcome and time-window checks\nImputation and continuous Hb\nRecord sampling and subgroups\nLocal updating assessed separately')]:
 box(x,.54,3.55,1.55,col,fill);icon(kind,x+.16,1.76,.30,col);txt(x+.72,1.95,title,11,col,True);txt(x+.16,1.52,body,9.3)
txt(6,.36,'Retrospective evaluation; explanatory analyses are exploratory.',10,G,ha='center')
for ext in ['pdf','svg','png']:f.savefig(R/'figures'/('Figure1_reliability_vector.'+ext),dpi=600,facecolor='white')
plt.close(f)