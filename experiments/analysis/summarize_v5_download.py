"""Paired v4/v5 results, uncertainty, training curves and velocity envelope."""
import sys,json,csv,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_v5_download import OUT
from experiments.analysis.evaluate_v5_download import OLD
from experiments.analysis.summarize_seed606_v3_experiments import wilson
from experiments.analysis.summarize_v4_download import style
NAMES=['zero','v4','v5_best','v5_latest']
LABELS={'zero':'PD / zero','v4':'PPO v4','v5_best':'PPO v5 best','v5_latest':'PPO v5 latest'}
COLORS={'zero':'#a3b1bf','v4':'#edbe85','v5_best':'#84d4c0','v5_latest':'#c5a9df'}
HEIGHTS=[1.,1.5,2.,2.2]
def main():
 rows=json.loads((OLD/'test_results.json').read_text())
 for n in ['best','latest']:rows+=json.loads((OUT/f'{n}_results.json').read_text())
 assert len(rows)==3700
 by={(r['controller'],r['case']['id']):r for r in rows};ids=[r['case']['id'] for r in rows if r['controller']=='zero'];assert len(ids)==925
 assert all(len({by[n,i]['scenario_hash'] for n in NAMES})==1 for i in ids)
 summary=dict(paired_cases=925,total_episodes=3700,new_v5_episodes=1850,groups=[],by_height=[],speed_grid=[],failures={},paired_comparisons={},motor_diagnostics={},notes='95% Wilson intervals concern scenario sampling, not variance across training seeds. Best/latest are correlated checkpoints of one training run. Historical baselines reused after 32 exact replay checks.')
 for n in NAMES:
  rr=[r for r in rows if r['controller']==n]
  for kind in sorted({r['case']['kind'] for r in rr}):
   sub=[r for r in rr if r['case']['kind']==kind];k=sum(r['success'] for r in sub);count=len(sub)
   summary['groups'].append(dict(controller=n,kind=kind,successes=k,count=count,rate=k/count,ci95=wilson(k,count)))
   for h in HEIGHTS:
    hh=[r for r in sub if r['case']['height']==h]
    if hh:summary['by_height'].append(dict(controller=n,kind=kind,height=h,successes=sum(r['success'] for r in hh),count=len(hh)))
  summary['failures'][n]=dict(collections.Counter(r['reason'] for r in rr if not r['success']))
  summary['motor_diagnostics'][n]={key:max(r['telemetry'][key] for r in rr) for key in ['flight_calf_saturation_fraction','flight_calf_sign_flip_fraction']}
  for speed in [.5,1,1.5,2,2.5,3]:
   sub=[r for r in rr if r['case']['kind']=='speed_grid' and r['case']['speed']==speed];summary['speed_grid'].append(dict(controller=n,speed=speed,successes=sum(r['success'] for r in sub),count=len(sub)))
 combined=[i for i in ids if by['zero',i]['case']['kind']=='random_combined']
 for a,b in [('v4','v5_best'),('v5_best','v5_latest')]:
  summary['paired_comparisons'][a+'_vs_'+b]=dict(both_success=sum(by[a,i]['success'] and by[b,i]['success'] for i in combined),a_only=sum(by[a,i]['success'] and not by[b,i]['success'] for i in combined),b_only=sum(not by[a,i]['success'] and by[b,i]['success'] for i in combined),both_fail=sum(not by[a,i]['success'] and not by[b,i]['success'] for i in combined))
 probe=json.loads((OUT/'velocity_zeroed_probe.json').read_text());sub=probe['results'];assert all(r['scenario_hash']==by['v5_best',r['case']['id']]['scenario_hash'] for r in sub)
 summary['zeroed_probe']=dict(successes=sum(r['success'] for r in sub),count=len(sub),note=probe['note'])
 summary['combined_speed_bins']=[]
 for lo,hi in [(0,1),(1,2),(2,3),(3,4.25)]:
  for n in NAMES:
   sub=[r for r in rows if r['controller']==n and r['case']['kind']=='random_combined' and lo<=np.hypot(r['case']['vx'],r['case']['vy'])<hi]
   summary['combined_speed_bins'].append(dict(controller=n,min_inclusive=lo,max_exclusive=hi,successes=sum(r['success'] for r in sub),count=len(sub)))
 summary['best_combined_failures']=[dict(case=r['case'],reason=r['reason'],speed=float(np.hypot(r['case']['vx'],r['case']['vy'])),peak_acceleration=r['peak_acceleration']) for r in rows if r['controller']=='v5_best' and r['case']['kind']=='random_combined' and not r['success']]
 summary['latest_combined_failures']=[dict(case=r['case'],reason=r['reason'],speed=float(np.hypot(r['case']['vx'],r['case']['vy'])),peak_acceleration=r['peak_acceleration']) for r in rows if r['controller']=='v5_latest' and r['case']['kind']=='random_combined' and not r['success']]
 summary['combined_first_forbidden_contacts']={}
 for n in ['v5_best','v5_latest']:
  sub=[r for r in rows if r['controller']==n and r['case']['kind']=='random_combined' and not r['success']]
  summary['combined_first_forbidden_contacts'][n]=dict(collections.Counter(' + '.join(sorted(r['forbidden'][0]['geoms'])) for r in sub if r['forbidden']))
 summary['loads_combined']={}
 for n in NAMES:
  sub=[r for r in rows if r['controller']==n and r['case']['kind']=='random_combined']
  summary['loads_combined'][n]={key:dict(median=float(np.median([r[key] for r in sub])),p95=float(np.percentile([r[key] for r in sub],95)),max=max(r[key] for r in sub)) for key in ['peak_foot_force','peak_acceleration','max_action']}
 movies=json.loads((OUT/'videos.json').read_text())+json.loads((OUT/'best_videos.json').read_text())
 if (OUT/'lateral_videos.json').exists():movies+=json.loads((OUT/'lateral_videos.json').read_text())
 checks=0
 for movie in movies:
  for clip in movie['clips']:
   for r in clip['outcomes']:
    expected=by[r['controller'],clip['case']['id']]
    assert clip['scenario_hash']==expected['scenario_hash']
    for key in ['success','reason','end_time']:assert r[key]==expected[key],(clip['case']['id'],key)
    checks+=1
 summary['video_replay_outcomes_verified']=checks
 (OUT/'test_summary.json').write_text(json.dumps(summary,indent=2))
 with (OUT/'paired_results.csv').open('w') as f:
  cols=['id','kind','height','roll','pitch','vx','vy','scenario_hash']+[n+s for n in NAMES for s in ['_success','_reason']];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for i in ids:
   r=by['zero',i];c=r['case'];row={k:c[k] for k in cols[:7]};row['scenario_hash']=r['scenario_hash']
   for n in NAMES:row.update({n+'_success':by[n,i]['success'],n+'_reason':by[n,i]['reason']})
   w.writerow(row)
 style();fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);p=json.loads((OUT/'parsed.json').read_text());es=p['evaluations'];x=[r['metrics']['steps/nr_env_steps']/1e6 for r in es]
 for kind,label,color in [('nominal','nominalne','#a3b1bf'),('angles','same kąty','#84d4c0'),('velocity','sama prędkość','#edbe85'),('perturbed','kąty + prędkość','#c5a9df')]:
  axs[0,0].plot(x,[100*r['metrics'][f'eval/{kind}/policy/success_rate'] for r in es],label=label,color=color)
 axs[0,0].axhspan(70,80,color='#84d4c0',alpha=.12);axs[0,0].set(title='V5: stały zestaw walidacyjny',ylabel='Sukces [%]',ylim=(-2,103));axs[0,0].legend(frameon=False)
 allx=[r['metrics']['steps/nr_env_steps']/1e6 for r in p['rows']]
 for factor,color in [('angle','#84d4c0'),('velocity','#edbe85')]:axs[0,1].plot(allx,[r['metrics'][f'env_info/curriculum/{factor}_difficulty'] for r in p['rows']],label=factor,color=color)
 axs[0,1].set(title='Curriculum osiąga pełny zakres',ylabel='Trudność',ylim=(-.02,1.03));axs[0,1].legend(frameon=False)
 for run,label,color in [(p,'v5','#84d4c0'),(json.loads((OLD/'parsed.json').read_text()),'v4','#edbe85')]:
  e=run['evaluations'];axs[1,0].plot([r['metrics']['steps/nr_env_steps']/1e6 for r in e],[100*r['metrics']['eval/landing_success_rate'] for r in e],label=label,color=color)
 axs[1,0].set(title='Walidacja: kąty i prędkości, v4 vs v5',ylabel='Sukces [%]',ylim=(-2,103));axs[1,0].legend(frameon=False)
 axs[1,1].plot(allx,[r['metrics']['policy/std_dev'] for r in p['rows']],color='#c5a9df');axs[1,1].set(title='Eksploracja PPO',ylabel='Średnie std akcji')
 for ax in axs.flat:ax.set_xlabel('Przejścia środowiska [mln]');ax.grid(alpha=.25)
 for ext in ['svg','png']:fig.savefig(OUT/f'learning_curves.{ext}',dpi=140)
 plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for ax,h in zip(axs.flat,HEIGHTS):
  for n in NAMES:
   values=[]
   for speed in [.5,1,1.5,2,2.5,3]:
    sub=[r for r in rows if r['controller']==n and r['case']['kind']=='speed_grid' and r['case']['height']==h and r['case']['speed']==speed];values.append(100*sum(r['success'] for r in sub)/len(sub))
   ax.plot([.5,1,1.5,2,2.5,3],values,marker='o',label=LABELS[n],color=COLORS[n])
  ax.set(title=f'h={h:g} m',xlabel='Moduł prędkości poziomej [m/s]',ylabel='Udane kierunki [%]',ylim=(-3,103));ax.grid(alpha=.25)
 axs[0,0].legend(frameon=False,fontsize=9);fig.suptitle('Sama prędkość: osiem kierunków, bez przechyłu')
 for ext in ['svg','png']:fig.savefig(OUT/f'speed_envelope.{ext}',dpi=140)
 plt.close(fig)
 for n in NAMES:
  print(n,[(s['kind'],s['successes'],s['count']) for s in summary['groups'] if s['controller']==n],flush=True)
 print('probe',summary['zeroed_probe']);print('paired',summary['paired_comparisons'])
if __name__=='__main__':main()
