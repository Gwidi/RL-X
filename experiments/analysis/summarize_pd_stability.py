"""Reproducible summary of gain ablations and stable-integrator tests."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.check_pd_stability import OUT,VARIANTS
ROOT=Path(__file__).resolve().parents[2]

def main():
    replay=json.loads((OUT/'replay_results.json').read_text());rows=[];summaries=[]
    original=json.loads((ROOT/'docs/artifacts/tilt_control_diagnostic/results.json').read_text())
    for r in original:r['variant']='original'
    rows.extend(original)
    for path in sorted((OUT/'closed_loop').glob('*/results.json')):rows.extend(json.loads(path.read_text()))
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    initial_hashes={}
    for r in rows:
        if r['variant']=='original':continue
        cid=r['case']['id'];h=r['initial_state_hash']
        if cid in initial_hashes:assert initial_hashes[cid]==h
        initial_hashes[cid]=h
    for variant in variants:
        selected=[r for r in rows if r['variant']==variant];assert len(selected)==204
        for name in ['zero','best','latest']:
            rr=[r for r in selected if r['controller']==name];nom=[r for r in rr if r['case']['axis']=='nominal']
            summaries.append(dict(variant=variant,controller=name,count=len(rr),successes=sum(r['success'] for r in rr),nominal_successes=sum(r['success'] for r in nom),nominal_count=len(nom),by_angle={str(a):dict(successes=sum(r['success'] for r in rr if abs(r['case']['angle'])==a),count=sum(abs(r['case']['angle'])==a for r in rr)) for a in [0,3,10,20,30]},median_flight_saturation=float(np.median([r['metrics']['flight']['motor_saturation_fraction'] for r in rr])),median_flight_rms_deg=float(np.median([r['metrics']['flight']['tracking_rms_deg'] for r in rr]))))
    screen=[]
    for path in sorted((OUT/'screen').glob('*/results.json')):
        rr=json.loads(path.read_text());screen.append(dict(variant=path.parent.name,episodes=len(rr),controllers={n:dict(successes=sum(r['success'] for r in rr if r['controller']==n),count=sum(r['controller']==n for r in rr),nominal_successes=sum(r['success'] for r in rr if r['controller']==n and r['case']['axis']=='nominal')) for n in ['zero','best','latest']}))
    convergence=[]
    for name in ['zero','best','latest']:
        a={r['case']['id']:r for r in rows if r['variant']=='implicit_2p5ms' and r['controller']==name};b={r['case']['id']:r for r in rows if r['variant']=='implicit_1p25ms' and r['controller']==name}
        if not b:continue
        changes=sum(a[k]['success']!=b[k]['success'] for k in a)
        angle_diff=[np.max(np.abs(np.array(a[k]['rpy_before_contact'][:2])-b[k]['rpy_before_contact'][:2])) for k in a]
        convergence.append(dict(controller=name,classification_changes=changes,count=len(a),median_max_rp_difference_at_respective_contact_deg=float(np.median(angle_diff)),max_rp_difference_at_respective_contact_deg=float(np.max(angle_diff))))
    summary=dict(variants=VARIANTS,full=summaries,screen=screen,replay=replay['rows'],refinement=convergence,full_episodes=sum(r['count'] for r in summaries),new_full_episodes=sum(r['count'] for r in summaries if r['variant']!='original'),screen_episodes=sum(r['episodes'] for r in screen),replay_count=len(replay['rows']),notes=['Original 204 episodes reused from Etap6. Screen and full repeat some identical cases; do not treat repetitions as independent evidence.','Pure deterministic tilt grid; no statistical success-probability claim or training-seed comparison.','Old checkpoints transferred to changed physics without retraining.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'comparison.csv').open('w') as f:
        fields=['variant','controller','successes','count','nominal_successes','nominal_count','median_flight_saturation','median_flight_rms_deg'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summaries)
    plt.rcParams.update({'text.color':'#dce4ec','axes.labelcolor':'#dce4ec','xtick.color':'#a3b1bf','ytick.color':'#a3b1bf','axes.edgecolor':'#34404b','axes.facecolor':'#11171e','figure.facecolor':'#11171e','grid.color':'#34404b','svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    labels=['original','kp_half','kd_075','kd_half','kd_quarter','calf_kd_half','implicit_5ms','implicit_2p5ms','implicit_1p25ms']
    vals=[100*np.mean([r['calf_saturation_fraction'] for r in replay['rows'] if r['variant']==v]) for v in labels]
    axs[0].barh(range(len(labels)),vals,color='#84d4c0');axs[0].set(yticks=range(len(labels)),yticklabels=labels,xlabel='Próbki ≥95% limitu momentu łydki [%]',title='Identyczne komendy w locie, średnia 4 replayów');axs[0].invert_yaxis()
    for i,name in enumerate(['zero','best','latest']):
        values=[next(r['successes']/r['count']*100 for r in summaries if r['variant']==v and r['controller']==name) for v in variants]
        axs[1].bar(np.arange(len(variants))+(i-1)*.25,values,.25,color=['#a3b1bf','#84d4c0','#edbe85'][i],label={'zero':'PD / zero','best':'best','latest':'latest'}[name])
    axs[1].set(xticks=range(len(variants)),xticklabels=variants,ylabel='Sukcesy w 68 stanach [%]',title='Zamknięta pętla: stare wagi, zmieniona fizyka');axs[1].tick_params(axis='x',rotation=35);axs[1].legend(frameon=False,labelcolor='#dce4ec');axs[1].grid(axis='y',alpha=.25)
    fig.savefig(OUT/'comparison.svg');fig.savefig(OUT/'comparison.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    for v,color in [('original','#edbe85'),('kd_075','#c4a6df'),('calf_kd_half','#a3b1bf'),('implicit_2p5ms','#84d4c0')]:
        r=next(r for r in replay['rows'] if r['variant']==v and r['controller']=='latest' and r['axis']=='roll');z=np.load(OUT/r['trace']);axs[0].plot(z['t'],z['tau'][:,12],label=v,color=color);axs[1].plot(z['t'],np.degrees(z['q'][:,12]),color=color)
    axs[0].set_ylabel('Moment silnika FL_calf [Nm]');axs[1].set_ylabel('Pozycja FL_calf [°]');axs[0].legend(frameon=False,labelcolor='#dce4ec')
    for ax in axs:ax.set_xlim(.05,.3);ax.set_xlabel('Czas [s]');ax.grid(alpha=.3)
    fig.suptitle('Te same komendy latest / roll +30°: mniejsze Kd i implicitfast usuwają oscylacje')
    fig.savefig(OUT/'gain_replay.svg');fig.savefig(OUT/'gain_replay.png',dpi=140);plt.close(fig)
    print(json.dumps(summary['full'],indent=2));print('REFINEMENT',convergence)
if __name__=='__main__':main()
