"""Summarize and plot the diagnostic tilt sweep; no training changes."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiments.analysis.diagnose_tilt_control import OUT,NAMES

COLORS={'zero':'#a3b1bf','best':'#84d4c0','latest':'#edbe85'}
LABELS={'zero':'PD / zero','best':'PPO best','latest':'PPO latest'}


def onset(t,x,limit,deadline):
    above=x>=limit
    for k in range(len(t)-2):
        if t[k+2]<deadline and above[k:k+3].all():return float(t[k])
    return None


def main():
    rows=json.loads((OUT/'results.json').read_text());assert len(rows)==204
    mapping={(r['controller'],r['case']['id']):r for r in rows}
    for name in NAMES:
        for h in [1.,1.5,2.,2.5]:
            base=next(r for r in rows if r['controller']==name and r['case']['height']==h and r['case']['axis']=='nominal')
            n=np.load(OUT/base['trace'])
            for r in [r for r in rows if r['controller']==name and r['case']['height']==h]:
                z=np.load(OUT/r['trace']);tc=r['first_contact']['time'] if r['first_contact'] else r['end_time'];t=z['control_t'];mask=t<tc
                nominal=np.stack([np.interp(t,n['control_t'],n['control_target'][:,j]) for j in range(13)],axis=1)
                diff=np.degrees(np.max(np.abs(z['control_target']-nominal),axis=1))
                r['nominal_comparison']=dict(peak_target_difference_flight_deg=float(diff[mask].max()),mean_target_difference_flight_deg=float(diff[mask].mean()),onset_5deg_3steps=onset(t,diff,5.,tc),note='Max across joints vs nominal trajectory at same height; descriptive, resets have different seeds. Threshold sustained for 3 control samples; not a physiological latency.')
    groups=[]
    for a in [0,3,10,20,30]:
        for n in NAMES:
            selected=[r for r in rows if r['controller']==n and abs(r['case']['angle'])==a]
            groups.append(dict(angle=a,controller=n,successes=sum(r['success'] for r in selected),count=len(selected),median_flight_tracking_rms_deg=float(np.median([r['metrics']['flight']['tracking_rms_deg'] for r in selected])),median_flight_saturation_any_fraction=float(np.median([r['metrics']['flight']['motor_saturation_any_fraction'] for r in selected])),median_flight_peak_action=float(np.median([r['metrics']['flight']['max_abs_action'] for r in selected])),median_target_difference_vs_nominal_deg=float(np.median([r['nominal_comparison']['peak_target_difference_flight_deg'] for r in selected]))))
    summary=dict(episodes=len(rows),paired_cases=len(rows)//3,groups=groups,results=rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'metrics.csv').open('w') as f:
        columns=['controller','height','axis','angle','success','reason','contact_time','flight_rms_deg','flight_saturation_any_fraction','flight_peak_action','target_difference_vs_nominal_deg','reaction_onset_5deg','roll_before_contact','pitch_before_contact']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:
            c=r['case'];m=r['metrics']['flight'];nc=r['nominal_comparison']
            writer.writerow(dict(controller=r['controller'],height=c['height'],axis=c['axis'],angle=c['angle'],success=r['success'],reason=r['reason'],contact_time=r['first_contact']['time'] if r['first_contact'] else None,flight_rms_deg=m['tracking_rms_deg'],flight_saturation_any_fraction=m['motor_saturation_any_fraction'],flight_peak_action=m['max_abs_action'],target_difference_vs_nominal_deg=nc['peak_target_difference_flight_deg'],reaction_onset_5deg=nc['onset_5deg_3steps'],roll_before_contact=r['rpy_before_contact'][0],pitch_before_contact=r['rpy_before_contact'][1]))
    plt.rcParams.update({'text.color':'#dce4ec','axes.labelcolor':'#dce4ec','xtick.color':'#a3b1bf','ytick.color':'#a3b1bf','axes.edgecolor':'#34404b','axes.facecolor':'#11171e','figure.facecolor':'#11171e','grid.color':'#34404b','svg.fonttype':'none','font.size':10})
    angles=[-30,-20,-10,-3,0,3,10,20,30];heights=[1.,1.5,2.,2.5]
    fig,axs=plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
    for ax,axis in zip(axs,['roll','pitch']):
        matrix=[];labels=[]
        for h in heights:
            for name in NAMES:
                matrix.append([next(r['success'] for r in rows if r['controller']==name and r['case']['height']==h and r['case']['axis']==('nominal' if a==0 else axis) and r['case']['angle']==a) for a in angles]);labels.append(f'{h:g} m / {LABELS[name]}')
        grid=np.array(matrix);ax.imshow(grid,cmap=ListedColormap(['#49353a','#64b7a1']),aspect='auto',vmin=0,vmax=1)
        ax.set(xticks=range(9),xticklabels=angles,yticks=range(12),yticklabels=labels,xlabel='Początkowy kąt [°]',title=axis)
        for y in range(12):
            for x in range(9):ax.text(x,y,'✓' if grid[y,x] else '×',ha='center',va='center',color='white')
        for y in [2.5,5.5,8.5]:ax.axhline(y,color='#dce4ec',lw=1)
    fig.suptitle('Test istniejących wag: same przechyły, v_xy=0 · jedna próba na komórkę',fontsize=15)
    fig.savefig(OUT/'success_map.svg');fig.savefig(OUT/'success_map.png',dpi=140);plt.close(fig)
    # Detailed representative cases chosen by condition, not by outcome.
    examples=[]
    protocol=json.loads((OUT/'protocol.json').read_text());joint_names=['spine','RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf','FR_hip','FR_thigh','FR_calf','FL_hip','FL_thigh','FL_calf']
    for h in [1.,2.5]:
        for axis in ['roll','pitch']:
            for angle in [10,30]:
                c=next(c for c in protocol['cases'] if c['height']==h and c['axis']==axis and c['angle']==angle)
                fig,axs=plt.subplots(4,2,figsize=(14,13),constrained_layout=True)
                for name in NAMES:
                    r=mapping[name,c['id']];z=np.load(OUT/r['trace']);t=z['t'];tc=r['first_contact']['time'] if r['first_contact'] else r['end_time'];color=COLORS[name]
                    for idx in [0,1]:axs[0,idx].plot(t,z['rpy'][:,idx],color=color,label=LABELS[name])
                    axs[1,0].plot(z['control_t'],np.max(np.abs(z['control_action']),axis=1),color=color,label=LABELS[name],drawstyle='steps-post')
                    axs[1,1].plot(t,np.degrees(np.sqrt(np.mean((z['target']-z['q'])**2,axis=1))),color=color)
                    axs[2,0].plot(t,np.max(np.abs(z['motor_torque'])/z['motor_limit'],axis=1),color=color)
                    axs[2,1].plot(t,np.max(np.abs(z['joint_torque']),axis=1),color=color)
                    if name!='zero':
                        ax=axs[3,0 if name=='best' else 1];j=int(np.argmax(r['metrics']['flight']['per_joint_rms_deg']))
                        ax.plot(t,np.degrees(z['target'][:,j]),'--',color=color,label='zadana');ax.plot(t,np.degrees(z['q'][:,j]),color=color,label='rzeczywista');ax.set_title(f'{LABELS[name]}: największy RMS w locie — {joint_names[j]}');ax.set_ylabel('Pozycja stawu [°]');ax.legend(frameon=False,labelcolor='#dce4ec');ax.axvline(tc,color=color,ls=':',lw=1.5)
                    for ax in axs[:3].flat:ax.axvline(tc,color=color,ls=':',lw=1)
                titles=['Roll korpusu [°]','Pitch korpusu [°]','Maks. |akcja|','RMS błędu śledzenia, 13 stawów [°]','Maks. wykorzystanie limitu momentu silnika','Maks. |moment stawu| [Nm]']
                deadline=max(mapping[n,c['id']]['first_contact']['time'] for n in NAMES)+.3
                for ax,title in zip(axs[:3].flat,titles):ax.set_title(title)
                for ax in axs.flat:ax.set_xlabel('Czas [s]');ax.set_xlim(0,deadline);ax.grid(alpha=.3)
                axs[0,0].legend(frameon=False,labelcolor='#dce4ec');axs[2,0].axhline(.95,color='white',ls='--',lw=.7)
                fig.suptitle(f'h={h:g} m, {axis}={angle:+}°, v_xy=0 · pionowe linie: pierwszy kontakt\nWykres do kontaktu +0,3 s; przebieg kończy się z epizodem. Pełne dane w NPZ.',fontsize=14)
                filename=f'detail_{c["id"]}.svg';fig.savefig(OUT/filename);plt.close(fig);examples.append(dict(case=c,figure=filename))
    summary['examples']=examples;(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
