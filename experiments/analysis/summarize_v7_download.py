"""V7 paired benchmark, sensor-intervention matrix and training curves."""
import sys,json,csv,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_v7_download import OUT
from experiments.analysis.evaluate_v7_download import OLD
from experiments.analysis.summarize_v4_download import style
from experiments.analysis.summarize_seed606_v3_experiments import wilson

def group(rows,label):
 out=[]
 for kind in sorted({r['case']['kind'] for r in rows}):
  rr=[r for r in rows if r['case']['kind']==kind];k=sum(r['success'] for r in rr);n=len(rr);out.append(dict(method=label,kind=kind,successes=k,count=n,rate=k/n,ci95=wilson(k,n)))
 return out

def main():
 native=json.loads((OUT/'native_results.json').read_text());assert len(native)==314
 v6={n:json.loads((OLD/f'{n}_results.json').read_text()) for n in ['zero','best','latest']};data={**{'v6_'+n:rr for n,rr in v6.items()},'v7_noisy':native};groups=[]
 for n,rr in data.items():groups+=group(rr,n)
 s=dict(paired_cases=314,new_native_episodes=314,same_v7_policy_weights=True,latest_replays=len(json.loads((OUT/'latest_verification.json').read_text())),groups=groups,speed_grid=[],angle_grid=[],sensor_matrix=[],failures={},notes='One training seed each. One deterministic sensor-noise realization per case. Native v6 has ideal measurement; native v7 has noise+delay. Cross-sensor rows are explicit inference-time interventions, not retraining.')
 reference={r['case']['id']:r for r in v6['zero']}
 for n,rr in data.items():
  assert all(r['scenario_hash']==reference[r['case']['id']]['scenario_hash'] for r in rr)
  s['failures'][n]=dict(collections.Counter(r['reason'] for r in rr if not r['success']))
  for key,values in [('speed',[1,2,3,4,5]),('angle',[10,20,30])]:
   for value in values:
    sub=[r for r in rr if r['case']['kind']==key+'_grid' and (r['case'][key] if key=='speed' else abs(r['case'][key]))==value];s[key+'_grid'].append(dict(method=n,value=value,successes=sum(r['success'] for r in sub),count=len(sub)))
 matrix=[]
 for n in ['best','latest']:
  for r in v6[n]:
   if r['case']['kind']=='random_combined':matrix.append(dict(r,training='v6',checkpoint=n,sensor='clean'))
 matrix.extend(r for r in native if r['case']['kind']=='random_combined')
 extra=[]
 for file in ['v6_noisy_results.json','v7_clean_results.json','sensor_components_results.json']:extra+=json.loads((OUT/file).read_text())
 matrix+=extra;s['intervention_episodes']=len(extra)
 for training,checkpoint,sensor in sorted({(r['training'],r['checkpoint'],r['sensor']) for r in matrix}):
  rr=[r for r in matrix if (r['training'],r['checkpoint'],r['sensor'])==(training,checkpoint,sensor)];assert len(rr)==64;assert all(r['scenario_hash']==reference[r['case']['id']]['scenario_hash'] for r in rr);k=sum(r['success'] for r in rr);s['sensor_matrix'].append(dict(training=training,checkpoint=checkpoint,sensor=sensor,successes=k,count=64,rate=k/64,ci95=wilson(k,64)))
 def select(training,sensor):return {r['case']['id']:r for r in matrix if r['training']==training and r['checkpoint']==('best' if training=='v7' and sensor in ['noisy','clean'] else 'latest') and r['sensor']==sensor}
 comparisons=[]
 for a,b,label in [(select('v6','clean'),select('v7','clean'),'v6/v7 both clean'),(select('v6','noisy'),select('v7','noisy'),'v6/v7 both noisy'),(select('v7','clean'),select('v7','noisy'),'v7 clean/noisy')]:
  assert a.keys()==b.keys();comparisons.append(dict(comparison=label,both_success=sum(a[i]['success'] and b[i]['success'] for i in a),a_only=sum(a[i]['success'] and not b[i]['success'] for i in a),b_only=sum(not a[i]['success'] and b[i]['success'] for i in a),both_fail=sum(not a[i]['success'] and not b[i]['success'] for i in a)))
 s['paired_sensor_comparisons']=comparisons
 (OUT/'test_summary.json').write_text(json.dumps(s,indent=2))
 with (OUT/'paired_results.csv').open('w') as f:
  cols=['id','kind','height','vx','vy','roll','pitch']+list(data);w=csv.DictWriter(f,fieldnames=cols);w.writeheader();by={n:{r['case']['id']:r for r in rr} for n,rr in data.items()}
  for r in native:
   c=r['case'];w.writerow({**{k:c[k] for k in cols[:7]},**{n:by[n][c['id']]['success'] for n in data}})
 style();fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);p=json.loads((OUT/'parsed.json').read_text());old=json.loads((OLD/'parsed.json').read_text())
 for run,label,color in [(old,'v6','#edbe85'),(p,'v7','#84d4c0')]:
  es=run['evaluations'];axs[0,0].plot([r['metrics']['steps/nr_env_steps']/1e6 for r in es],[100*r['metrics']['eval/landing_success_rate'] for r in es],label=label,color=color)
 axs[0,0].set(title='Walidacja na 3 m: połączone zaburzenia',ylabel='Sukces [%]',ylim=(-2,103));axs[0,0].legend(frameon=False)
 for factor,color in [('angle','#84d4c0'),('velocity','#edbe85')]:
  for run,label,ls in [(old,'v6','--'),(p,'v7','-')]:axs[0,1].plot([r['metrics']['steps/nr_env_steps']/1e6 for r in run['rows']],[r['metrics'][f'env_info/curriculum/{factor}_difficulty'] for r in run['rows']],label=label+' '+factor,linestyle=ls,color=color)
 axs[0,1].set(title='Średnia trudność curriculum',ylabel='Trudność',ylim=(-.02,1.03));axs[0,1].legend(frameon=False)
 for key,ax in [('speed',axs[1,0]),('angle',axs[1,1])]:
  for name,label,color in [('v6_latest','v6 latest / ideal','#edbe85'),('v7_noisy','v7 / noisy','#84d4c0')]:
   rr=[r for r in s[key+'_grid'] if r['method']==name];ax.plot([r['value'] for r in rr],[100*r['successes']/r['count'] for r in rr],marker='o',label=label,color=color)
  ax.set(ylabel='Udane przypadki [%]',ylim=(-3,103));ax.legend(frameon=False)
 axs[1,0].set(title='3 m, bez przechyłu: osiem kierunków',xlabel='Moduł prędkości [m/s]');axs[1,1].set(title='3 m, bez prędkości: obie osie i znaki',xlabel='Wartość bezwzględna kąta [°]')
 for ax in axs[0]:ax.set_xlabel('Przejścia środowiska [mln]')
 for ax in axs.flat:ax.grid(alpha=.25)
 for ext in ['svg','png']:fig.savefig(OUT/f'analysis.{ext}',dpi=140)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);sensors=['clean','noise_only','delay_only','noisy'];x=np.arange(4)
 for training,label,color,offset in [('v6','v6 latest','#edbe85',-.17),('v7','v7 best=latest','#84d4c0',.17)]:
  rr=[next(r for r in s['sensor_matrix'] if r['training']==training and r['sensor']==sensor and r['checkpoint']==('best' if training=='v7' and sensor in ['clean','noisy'] else 'latest')) for sensor in sensors]
  ax.bar(x+offset,[100*r['rate'] for r in rr],width=.32,label=label,color=color)
  for xx,r in zip(x+offset,rr):ax.text(xx,100*r['rate']+1,str(r['successes'])+'/64',ha='center',fontsize=10)
 ax.set(xticks=x,xticklabels=['Idealny','Tylko szum','Tylko 20 ms','Szum + 20 ms'],ylabel='Sukces [%]',ylim=(0,55),title='Te same 64 przypadki połączone na 3 m; interwencja w inferencji');ax.legend(frameon=False);ax.grid(axis='y',alpha=.25)
 for ext in ['svg','png']:fig.savefig(OUT/f'sensor_matrix.{ext}',dpi=140)
 plt.close(fig);print(json.dumps(s['groups'],indent=2));print('matrix',s['sensor_matrix']);print('paired',comparisons)
if __name__=='__main__':main()
