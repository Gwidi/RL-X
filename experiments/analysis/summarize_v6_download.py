"""Paired v6 benchmark and diagnostic training curves."""
import sys,json,csv,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_v6_download import OUT
from experiments.analysis.summarize_v4_download import style
from experiments.analysis.summarize_seed606_v3_experiments import wilson
NAMES=['zero','best','latest']
def main():
 rows=[]
 for n in NAMES:rows+=json.loads((OUT/f'{n}_results.json').read_text())
 by={(r['controller'],r['case']['id']):r for r in rows};ids=[r['case']['id'] for r in rows if r['controller']=='zero'];assert len(ids)==314
 assert all(len({by[n,i]['scenario_hash'] for n in NAMES})==1 for i in ids)
 s=dict(paired_cases=314,new_episodes=len(rows),groups=[],speed_grid=[],angle_grid=[],failures={},previous_domain={},notes='Fixed diagnostic suite, one training seed; Wilson intervals concern scenario sampling only.')
 for n in NAMES:
  rr=[r for r in rows if r['controller']==n]
  for kind in sorted({r['case']['kind'] for r in rr}):
   sub=[r for r in rr if r['case']['kind']==kind];k=sum(r['success'] for r in sub);count=len(sub);s['groups'].append(dict(controller=n,kind=kind,successes=k,count=count,rate=k/count,ci95=wilson(k,count)))
  s['failures'][n]=dict(collections.Counter(r['reason'] for r in rr if not r['success']))
  for speed in [1,2,3,4,5]:
   sub=[r for r in rr if r['case']['kind']=='speed_grid' and r['case']['speed']==speed];s['speed_grid'].append(dict(controller=n,speed=speed,successes=sum(r['success'] for r in sub),count=len(sub)))
  for angle in [10,20,30]:
   sub=[r for r in rr if r['case']['kind']=='angle_grid' and abs(r['case']['angle'])==angle];s['angle_grid'].append(dict(controller=n,angle=angle,successes=sum(r['success'] for r in sub),count=len(sub)))
 probe=json.loads((OUT/'previous_domain_probe.json').read_text());s['previous_domain']=dict(count=64,v5_latest=probe['v5_latest_reference_successes'],**{n:sum(r['success'] for r in probe['results'] if r['controller']==n) for n in ['best','latest']})
 movies=json.loads((OUT/'videos.json').read_text());checks=0
 for movie in movies:
  for clip in movie['clips']:
   for r in clip['outcomes']:
    expected=by[r['controller'],clip['case']['id']];assert clip['scenario_hash']==expected['scenario_hash']
    for key in ['success','reason','end_time']:assert r[key]==expected[key]
    checks+=1
 s['video_outcomes_verified']=checks
 (OUT/'test_summary.json').write_text(json.dumps(s,indent=2))
 with (OUT/'paired_results.csv').open('w') as f:
  cols=['id','kind','height','roll','pitch','vx','vy','scenario_hash']+[n+'_success' for n in NAMES];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for i in ids:
   c=by['zero',i]['case'];w.writerow({**{k:c[k] for k in cols[:7]},'scenario_hash':by['zero',i]['scenario_hash'],**{n+'_success':by[n,i]['success'] for n in NAMES}})
 style();p=json.loads((OUT/'parsed.json').read_text());fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for kind,label,color in [('nominal','nominalnie','#a3b1bf'),('angles','same kąty','#84d4c0'),('velocity','sama prędkość','#edbe85'),('perturbed','razem','#c5a9df')]:
  es=p['evaluations'];axs[0,0].plot([r['metrics']['steps/nr_env_steps']/1e6 for r in es],[100*r['metrics'][f'eval/{kind}/policy/success_rate'] for r in es],label=label,color=color)
 axs[0,0].set(title='V6: stała walidacja na 3 m',ylabel='Sukces [%]',ylim=(-2,103));axs[0,0].legend(frameon=False)
 for factor,color in [('angle','#84d4c0'),('velocity','#edbe85')]:axs[0,1].plot([r['metrics']['steps/nr_env_steps']/1e6 for r in p['rows']],[r['metrics'][f'env_info/curriculum/{factor}_difficulty'] for r in p['rows']],label=factor,color=color)
 axs[0,1].set(title='Średnia trudność curriculum (64 środowiska)',ylabel='Trudność',ylim=(-.02,1.03));axs[0,1].legend(frameon=False)
 for n,color in zip(NAMES,['#a3b1bf','#84d4c0','#c5a9df']):
  rr=[r for r in s['speed_grid'] if r['controller']==n];axs[1,0].plot([r['speed'] for r in rr],[100*r['successes']/r['count'] for r in rr],marker='o',label=n,color=color)
  rr=[r for r in s['angle_grid'] if r['controller']==n];axs[1,1].plot([r['angle'] for r in rr],[100*r['successes']/r['count'] for r in rr],marker='o',label=n,color=color)
 axs[1,0].set(title='Osobny test: sama prędkość, 3 m',xlabel='Moduł prędkości [m/s]',ylabel='Udane kierunki [%]',ylim=(-3,103));axs[1,0].legend(frameon=False)
 axs[1,1].set(title='Osobny test: pojedyncze osie przechyłu, 3 m',xlabel='Wartość bezwzględna kąta [°]',ylabel='Udane przypadki [%]',ylim=(-3,103));axs[1,1].legend(frameon=False)
 for ax in axs[0]:ax.set_xlabel('Przejścia środowiska [mln]')
 for ax in axs.flat:ax.grid(alpha=.25)
 for ext in ['svg','png']:fig.savefig(OUT/f'analysis.{ext}',dpi=140)
 plt.close(fig);print(json.dumps(s,indent=2))
if __name__=='__main__':main()
