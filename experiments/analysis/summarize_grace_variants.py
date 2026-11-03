"""Shared comparison format, initially grace1s; other arms remain pending."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_grace1s_download import OUT,OLD
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_velocity_grid import wilson

def main():
 new=json.loads((OUT/'paired_results.json').read_text());old=json.loads((OLD/'paired_results.json').read_text());rows=[dict(r,controller='v9_'+r['controller']) for r in old]+[dict(r,controller='grace1s') for r in new];groups=[];pairs=[]
 for h,v in [(3.,4.),(2.5,4.),(3.,5.)]:
  for name in ['v9_best','v9_latest','grace1s']:
   g=[r for r in rows if r['controller']==name and r['case']['height']==h and r['case']['speed']==v];k=sum(r['success'] for r in g)
   durations=[r['end_time']-r['first_support'] for r in g if r['success'] and r['first_support'] is not None]
   groups.append(dict(height=h,speed=v,controller=name,successes=k,count=len(g),rate=k/len(g),wilson95=wilson(k,len(g)),success_contact_to_end_median=float(np.median(durations)) if durations else None,success_end_time_median=float(np.median([r['end_time'] for r in g if r['success']])) if k else None,failures={reason:sum(r['reason']==reason and not r['success'] for r in g) for reason in sorted(set(r['reason'] for r in g if not r['success']))}))
  a={r['case']['id']:r for r in rows if r['controller']=='v9_best' and r['case']['height']==h and r['case']['speed']==v};b={r['case']['id']:r for r in rows if r['controller']=='grace1s' and r['case']['height']==h and r['case']['speed']==v}
  pairs.append(dict(height=h,speed=v,v9_only=sum(a[i]['success'] and not b[i]['success'] for i in a),grace1s_only=sum(b[i]['success'] and not a[i]['success'] for i in a),both=sum(a[i]['success'] and b[i]['success'] for i in a)))
 (OUT/'summary.json').write_text(json.dumps(dict(groups=groups,paired=pairs,pending=['grace2s','grace3s']),indent=2))
 history=json.loads((OUT/'history.json').read_text());grid=history['checkpoint_grids']['best'];fig,axes=plt.subplots(1,2,figsize=(14,4.5));heights=grid['heights'];speeds=grid['speeds'];lookup={(c['height'],c['speed']):c for c in grid['cells'] if c['controller']=='policy'};values=np.array([[lookup[h,v]['success_rate']*100 for v in speeds] for h in heights]);axes[0].imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
 for i,h in enumerate(heights):
  for j,v in enumerate(speeds):axes[0].text(j,i,str(lookup[h,v]['successes'])+'/64',ha='center',va='center',fontsize=7,color='black' if values[i,j]>60 else 'white')
 axes[0].set(xticks=range(len(speeds)),xticklabels=speeds,yticks=range(len(heights)),yticklabels=heights,xlabel='Fixed speed [m/s]',ylabel='Height [m]',title='Grace1s checkpoint validation')
 x=np.arange(3)
 for j,name in enumerate(['v9_best','v9_latest','grace1s']):
  g=[r for r in groups if r['controller']==name];axes[1].bar(x+(j-1)*.25,[r['rate']*100 for r in g],width=.25,label=name)
  for i,r in enumerate(g):axes[1].text(i+(j-1)*.25,r['rate']*100+1,f"{r['successes']}/64",ha='center',fontsize=8)
 axes[1].set(xticks=x,xticklabels=['3 m / 4 m/s','2.5 m / 4 m/s','3 m / 5 m/s'],ylabel='Success [%]',ylim=(0,115));axes[1].axhline(80,color='black',ls='--');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
