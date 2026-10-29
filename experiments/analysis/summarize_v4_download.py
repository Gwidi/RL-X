"""Scientific figures and paired summary for completed v4 and fresh tests."""
import sys,json,csv,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiments.analysis.analyze_v4_download import OUT
from experiments.analysis.summarize_seed606_v3_experiments import wilson
HEIGHTS=[1.,1.5,2.,2.2]
COLORS={'zero':'#a3b1bf','v4':'#84d4c0'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#dce4ec','axes.labelcolor':'#dce4ec','xtick.color':'#a3b1bf','ytick.color':'#a3b1bf','axes.edgecolor':'#34404b','axes.facecolor':'#11171e','figure.facecolor':'#11171e','grid.color':'#34404b','svg.fonttype':'none'})

def learning_curves():
    p=json.loads((OUT/'parsed.json').read_text());style();fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    evals=p['evaluations'];x=np.array([r['metrics']['steps/nr_env_steps'] for r in evals])/1e6
    for condition,label,color in [('nominal','nominalnie','#a3b1bf'),('angles','same kąty','#84d4c0'),('velocity','sama prędkość','#edbe85'),('perturbed','kąty + prędkość','#c5a9df')]:
        y=[np.mean([r['metrics'][f'eval/{condition}/policy/h_{h:g}'.replace('.','p')+'/success_rate'] for h in HEIGHTS]) for r in evals]
        axs[0,0].plot(x,100*np.array(y),label=label,color=color)
    axs[0,0].axhspan(70,80,color='#84d4c0',alpha=.12);axs[0,0].set(title='Stała walidacja: pełny zakres ±10°, vx/vy ±3 m/s',ylabel='Sukces [%]',ylim=(-2,103));axs[0,0].legend(frameon=False,labelcolor='#dce4ec')
    x=np.array([r['metrics']['steps/nr_env_steps'] for r in p['rows']])/1e6
    for factor,color in [('angle','#84d4c0'),('velocity','#edbe85')]:axs[0,1].plot(x,[r['metrics'][f'env_info/curriculum/{factor}_difficulty'] for r in p['rows']],color=color,label=factor)
    axs[0,1].set(title='Niezależne poziomy curriculum',ylabel='Trudność [0–1]',ylim=(-.02,1.03));axs[0,1].legend(frameon=False,labelcolor='#dce4ec')
    for group,color in [('nominal','#a3b1bf'),('angles','#84d4c0'),('velocity','#edbe85'),('combined','#c5a9df')]:axs[1,0].plot(x,[100*r['metrics'][f'rollout/group_{group}_success_rate'] for r in p['rows']],color=color,label=group)
    axs[1,0].set(title='Rollout: zmieniający się rozkład treningowy',ylabel='Sukces [%]',ylim=(-2,103));axs[1,0].legend(frameon=False,labelcolor='#dce4ec')
    for key,color,label in [('flight_calf_saturation_fraction','#edbe85','nasycenie ≥95% limitu'),('flight_calf_sign_flip_fraction','#84d4c0','istotne odwrócenia znaku')]:axs[1,1].plot(x,[100*r['metrics'][f'env_info/metrics/{key}'] for r in p['rows']],color=color,label=label)
    axs[1,1].set(title='Diagnostyka napędów łydki przed kontaktem',ylabel='Udział próbek [%]',ylim=(-.2,5));axs[1,1].legend(frameon=False,labelcolor='#dce4ec');axs[1,1].text(.05,.45,'Oba wskaźniki = 0\nw 229 zapisanych agregatach',transform=axs[1,1].transAxes,color='#84d4c0')
    for ax in axs.flat:ax.set_xlabel('Przejścia środowiska [mln]');ax.grid(alpha=.25)
    fig.savefig(OUT/'learning_curves.svg');fig.savefig(OUT/'learning_curves.png',dpi=140);plt.close(fig)

def main():
    learning_curves();rows=json.loads((OUT/'test_results.json').read_text());assert len(rows)==1850
    summary=dict(actual_main_episodes=len(rows),paired_cases=len(rows)//2,identical_latest_replays=16,groups=[],by_height=[],failures={},telemetry={})
    kinds=['nominal','angle_grid','angle_corner','speed_grid','random_angles','random_velocity','random_combined','limited_combined']
    for name in ['zero','v4']:
        for kind in kinds:
            rr=[r for r in rows if r['controller']==name and r['case']['kind']==kind];k=sum(r['success'] for r in rr);n=len(rr)
            summary['groups'].append(dict(controller=name,kind=kind,successes=k,count=n,rate=k/n,ci95=wilson(k,n)))
            for h in HEIGHTS:
                sub=[r for r in rr if r['case']['height']==h]
                if sub:summary['by_height'].append(dict(controller=name,kind=kind,height=h,successes=sum(r['success'] for r in sub),count=len(sub)))
        summary['failures'][name]=dict(collections.Counter(r['reason'] for r in rows if r['controller']==name and not r['success']))
        rr=[r for r in rows if r['controller']==name]
        summary['telemetry'][name]={k:dict(max=max(r['telemetry'][k] for r in rr),median=float(np.median([r['telemetry'][k] for r in rr]))) for k in ['flight_calf_saturation_fraction','flight_calf_sign_flip_fraction']}
    aa={r['case']['id']:r for r in rows if r['controller']=='zero'};bb={r['case']['id']:r for r in rows if r['controller']=='v4'};assert aa.keys()==bb.keys() and len(aa)==925
    assert all(aa[k]['scenario_hash']==bb[k]['scenario_hash'] for k in aa)
    with (OUT/'paired_results.csv').open('w') as f:
        cols=['id','kind','height','roll','pitch','vx','vy','zero_success','zero_reason','v4_success','v4_reason','scenario_hash'];writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for key in aa:
            c=aa[key]['case'];writer.writerow({**{k:c[k] for k in cols[:7]},'zero_success':aa[key]['success'],'zero_reason':aa[key]['reason'],'v4_success':bb[key]['success'],'v4_reason':bb[key]['reason'],'scenario_hash':aa[key]['scenario_hash']})
    style();fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,h in zip(axs.flat,HEIGHTS):
        for n in ['zero','v4']:
            values=[]
            for speed in [.5,1,1.5,2,2.5,3]:
                r=[r for r in rows if r['controller']==n and r['case']['kind']=='speed_grid' and r['case']['height']==h and r['case']['speed']==speed];values.append(sum(x['success'] for x in r)/8)
            ax.plot([.5,1,1.5,2,2.5,3],100*np.array(values),marker='o',color=COLORS[n],label='PD / zero' if n=='zero' else 'PPO best=latest')
        ax.set(title=f'h={h:g} m',xlabel='Moduł v_xy [m/s]',ylabel='Udane kierunki [%]',ylim=(-3,103));ax.grid(alpha=.3)
    axs[0,0].legend(frameon=False,labelcolor='#dce4ec');fig.suptitle('Sama prędkość: roll=pitch=0; osiem kierunków na punkt')
    fig.savefig(OUT/'speed_envelope.svg');fig.savefig(OUT/'speed_envelope.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True);angles=[-30,-20,-10,-3,0,3,10,20,30]
    for ax,axis in zip(axs,['roll','pitch']):
        grid=[];labels=[]
        for h in HEIGHTS:
            for n in ['zero','v4']:
                grid.append([next(r['success'] for r in rows if r['controller']==n and r['case']['height']==h and (r['case']['kind']=='nominal' if a==0 else r['case']['kind']=='angle_grid' and r['case']['axis']==axis and r['case']['angle']==a)) for a in angles]);labels.append(f'{h:g} m / '+('PD' if n=='zero' else 'PPO'))
        ax.imshow(grid,cmap=ListedColormap(['#49353a','#64b7a1']),aspect='auto',vmin=0,vmax=1);ax.set(xticks=range(9),xticklabels=angles,yticks=range(8),yticklabels=labels,title=axis,xlabel='Początkowy kąt [°]')
        for i in range(8):
            for j in range(9):ax.text(j,i,'✓' if grid[i][j] else '×',ha='center',va='center',color='white')
    fig.suptitle('Same przechyły: ±20° i ±30° poza zakresem treningu; jedna próba na komórkę')
    fig.savefig(OUT/'angle_map.svg');fig.savefig(OUT/'angle_map.png',dpi=140);plt.close(fig)
    (OUT/'test_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['groups'],indent=2))
if __name__=='__main__':main()
