import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_v9_download import OUT,OLD
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_velocity_grid import wilson

def main():
 history=json.loads((OUT/'history.json').read_text());rows=json.loads((OUT/'paired_results.json').read_text());old=json.loads((OLD/'fresh_results.json').read_text());allrows=[dict(r,controller='v8') for r in old if r['controller']=='best']+rows;groups=[];paired=[]
 for h,v in [(3.,4.),(2.5,4.),(3.,5.)]:
  for name in ['v8','best','latest']:
   group=[r for r in allrows if r['controller']==name and r['case']['height']==h and r['case']['speed']==v];wins=sum(r['success'] for r in group)
   groups.append(dict(height=h,speed=v,controller=name,successes=wins,count=len(group),rate=wins/len(group),wilson95=wilson(wins,len(group)),failures={reason:sum(r['reason']==reason and not r['success'] for r in group) for reason in sorted(set(r['reason'] for r in group if not r['success']))}))
  lookup={n:{r['case']['id']:r for r in allrows if r['controller']==n and r['case']['height']==h and r['case']['speed']==v} for n in ['v8','best','latest']}
  for a,b in [('v8','best'),('v8','latest'),('best','latest')]:
   paired.append(dict(height=h,speed=v,a=a,b=b,a_only=sum(lookup[a][i]['success'] and not lookup[b][i]['success'] for i in lookup[a]),b_only=sum(lookup[b][i]['success'] and not lookup[a][i]['success'] for i in lookup[a])))
 (OUT/'summary.json').write_text(json.dumps(dict(groups=groups,paired=paired),indent=2))
 grids=history['checkpoint_grids'];fig,axes=plt.subplots(1,2,figsize=(15,4.5))
 for ax,name in zip(axes,['best','latest']):
  e=grids[name];heights=e['heights'];speeds=e['speeds'];lookup={(c['height'],c['speed']):c for c in e['cells'] if c['controller']=='policy'};values=np.array([[lookup[h,v]['success_rate']*100 for v in speeds] for h in heights]);ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
  for i,h in enumerate(heights):
   for j,v in enumerate(speeds):ax.text(j,i,str(lookup[h,v]['successes'])+'/64',ha='center',va='center',fontsize=7,color='black' if values[i,j]>60 else 'white')
  ax.set(xticks=range(len(speeds)),xticklabels=speeds,yticks=range(len(heights)),yticklabels=heights,xlabel='Fixed horizontal speed [m/s]',ylabel='Height [m]',title=f'V9 {name} / step {e["step"]}')
 fig.suptitle('Checkpoint validation: zero initial tilt, noisy velocity');fig.tight_layout();fig.savefig(OUT/'grids.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));x=np.arange(3)
 for j,name in enumerate(['v8','best','latest']):
  group=[g for g in groups if g['controller']==name];ax.bar(x+(j-1)*.25,[g['rate']*100 for g in group],width=.25,label=name)
  for i,g in enumerate(group):ax.text(i+(j-1)*.25,g['rate']*100+1,f"{g['successes']}/64",ha='center',fontsize=8)
 ax.axhline(80,color='black',ls='--');ax.set(xticks=x,xticklabels=['3 m / 4 m/s','2.5 m / 4 m/s','3 m / 5 m/s'],ylabel='Success [%]',ylim=(0,115));ax.legend(ncol=3);fig.tight_layout();fig.savefig(OUT/'paired.png',dpi=160);print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
