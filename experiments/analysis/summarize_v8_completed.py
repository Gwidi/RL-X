"""Figures and paired fresh-seed results for completed v8."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_v8_download import OUT
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_velocity_grid import wilson

def main():
 h=json.loads((OUT/'reconstructed_history.json').read_text());fresh=json.loads((OUT/'fresh_results.json').read_text());groups=[]
 for height,speed in [(3.,4.),(2.5,4.),(3.,5.)]:
  for name in ['zero','best','latest']:
   rows=[r for r in fresh if r['controller']==name and r['case']['height']==height and r['case']['speed']==speed]
   if not rows:continue
   k=sum(r['success'] for r in rows);groups.append(dict(height=height,speed=speed,controller=name,successes=k,count=len(rows),rate=k/len(rows),wilson95=wilson(k,len(rows)),failures={reason:sum(r['reason']==reason and not r['success'] for r in rows) for reason in sorted(set(r['reason'] for r in rows if not r['success']))}))
 (OUT/'fresh_summary.json').write_text(json.dumps(groups,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(13,4.7));heights=h['latest']['heights'];speeds=h['latest']['speeds']
 for ax,name,title in zip(axes,['policy','zero'],['PPO v8 best = latest','Zero action / fixed PD']):
  lookup={(c['height'],c['speed']):c for c in h['latest']['cells'] if c['controller']==name};grid=np.array([[lookup[y,v]['success_rate']*100 for v in speeds] for y in heights]);ax.imshow(grid,vmin=0,vmax=100,cmap='viridis',aspect='auto')
  for i,y in enumerate(heights):
   for j,v in enumerate(speeds):ax.text(j,i,str(lookup[y,v]['successes'])+'/64',ha='center',va='center',fontsize=8,color='black' if grid[i,j]>60 else 'white')
  ax.set(xticks=range(len(speeds)),xticklabels=speeds,yticks=range(len(heights)),yticklabels=heights,xlabel='Fixed horizontal speed [m/s]',ylabel='Base height [m]',title=title)
 fig.suptitle('Final checkpoint validation: 64 seeded directions per cell, zero initial tilt');fig.tight_layout();fig.savefig(OUT/'grid.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4))
 for height in heights:ax.step([e['step']/1e6 for e in h['evaluations']],[e['contiguous_speed_boundary'][f'{height:g}'] if e['contiguous_speed_boundary'][f'{height:g}'] is not None else -1 for e in h['evaluations']],where='post',marker='o',label=f'{height:g}m')
 ax.set(xlabel='Training transitions [millions]',ylabel='Contiguous tested speed >80% [m/s]',ylim=(-.2,5.2));ax.grid(alpha=.3);ax.legend(ncol=5);fig.tight_layout();fig.savefig(OUT/'boundaries.png',dpi=160);plt.close(fig)
 print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
