"""Partial snapshots: budget-aligned cluster grids and paired replay results."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_velocity_grid import wilson
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'docs/artifacts';FOLDERS={1:ART/'v10_grace1s_completed_9439587_2026-11-03',2:ART/'v10_grace2s_partial_9441591_step8388608_2026-11-03',3:ART/'v10_grace3s_partial_9441491_step8388608_2026-11-03'};OUT=ART/'grace_partial_comparison_2026-11-03'
def main():
 OUT.mkdir(exist_ok=True);groups=[];motion=[];aligned=[]
 for g,p in FOLDERS.items():
  rows=json.loads((p/'paired_results.json').read_text());step=json.loads((p/'checkpoints.json').read_text())['best']['metrics']['nr_env_steps']
  for h,v in [(3.,4.),(2.5,4.),(3.,5.)]:
   selected=[r for r in rows if r['controller']=='best' and r['case']['height']==h and r['case']['speed']==v];ok=[r for r in selected if r['success']];k=len(ok);duration=[r['end_time']-r['first_support'] for r in ok]
   groups.append(dict(grace=g,checkpoint_step=step,height=h,speed=v,successes=k,count=len(selected),rate=k/len(selected),wilson95=wilson(k,len(selected)),success_contact_to_end_median=float(np.median(duration)) if duration else None,max_success_duration=max(duration) if duration else None,successes_before_grace=sum(t<g for t in duration),failures={reason:sum(r['reason']==reason and not r['success'] for r in selected) for reason in sorted(set(r['reason'] for r in selected if not r['success']))}))
  for r in rows:
   if r['controller']!='best' or r['case']['height']!=3 or r['case']['speed']!=4 or not r['success']:continue
   file=p/(f'trace_{r["case"]["id"]}.json' if g==1 else f'best_trace_{r["case"]["id"]}.json');tr=json.loads(file.read_text());tr=[x for x in tr if x['t']>=r['first_support']];xy=np.array([x['qpos'][:2] for x in tr]);motion.append(dict(grace=g,case_id=r['case']['id'],post_contact_base_path_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),post_contact_base_net_m=float(np.linalg.norm(xy[-1]-xy[0])),time_to_success=r['end_time']-r['first_support']))
  history=json.loads((p/'history.json').read_text());e=next(e for e in history['evaluations'] if e['step']==8388608);aligned.append(dict(grace=g,step=e['step'],cells=e['cells']))
 motion_summary=[dict(grace=g,n=len([r for r in motion if r['grace']==g]),median_base_path_m=float(np.median([r['post_contact_base_path_m'] for r in motion if r['grace']==g])),median_base_net_m=float(np.median([r['post_contact_base_net_m'] for r in motion if r['grace']==g]))) for g in [1,2,3]]
 result=dict(groups=groups,motion=motion_summary,motion_records=motion,budget_aligned_grids=aligned,notes='1s replay model is final30M;2/3s models are8.39M. Equal-budget1s comparison is available only from logged validation, not replay weights. Motion metrics describe successful cases only and base displacement is not a gait or rebound detector.')
 (OUT/'summary.json').write_text(json.dumps(result,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(13,4));x=np.arange(3)
 for j,g in enumerate([1,2,3]):
  selected=[r for r in groups if r['grace']==g];axes[0].bar(x+(j-1)*.25,[r['rate']*100 for r in selected],width=.25,label=f'{g}s / {selected[0]["checkpoint_step"]/1e6:.1f}M')
  for i,r in enumerate(selected):axes[0].text(i+(j-1)*.25,r['rate']*100+1,f'{r["successes"]}/64',ha='center',fontsize=8)
  aligned_cells=next(e['cells'] for e in aligned if e['grace']==g);heights=[1.,1.5,2.,2.5,3.];rates=[next(c['success_rate'] for c in aligned_cells if c['controller']=='policy' and c['height']==h and c['speed']==4) for h in heights];axes[1].plot(heights,np.array(rates)*100,marker='o',label=f'{g}s')
 axes[0].axhline(80,color='black',ls='--');axes[0].set(xticks=x,xticklabels=['3m/4mps','2.5m/4mps','3m/5mps'],ylabel='Success [%]',ylim=(0,116),title='Paired replays, different training budgets');axes[0].legend(fontsize=8)
 axes[1].axhline(80,color='black',ls='--');axes[1].set(xlabel='Height [m]',ylabel='Success at4m/s [%]',ylim=(75,102),title='Cluster validation at equal8.39M budget');axes[1].legend();axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);print(json.dumps(dict(groups=groups,motion=motion_summary),indent=2))
if __name__=='__main__':main()
