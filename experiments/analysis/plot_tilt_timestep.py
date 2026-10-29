import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path('docs/artifacts/tilt_control_diagnostic');rows=json.loads((p/'timestep_probe.json').read_text())['rows'];results=json.loads((p/'results.json').read_text());r=next(r for r in results if r['controller']=='latest' and r['case']['height']==2.5 and r['case']['axis']=='roll' and r['case']['angle']==30);original=np.load(p/r['trace'])
plt.rcParams.update({'text.color':'#dce4ec','axes.labelcolor':'#dce4ec','xtick.color':'#a3b1bf','ytick.color':'#a3b1bf','axes.edgecolor':'#34404b','axes.facecolor':'#11171e','figure.facecolor':'#11171e','grid.color':'#34404b','svg.fonttype':'none'})
fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
for dt,color in zip([.005,.0025,.00125],['#edbe85','#84d4c0','#a3b1bf']):
 r=next(r for r in rows if r['controller']=='latest' and r['axis']=='roll' and r['dt']==dt);z=np.load(p/r['trace']);axs[0].plot(z['t'],z['motor_torque'][:,12],label=f'{dt*1000:g} ms',color=color);axs[1].plot(z['t'],np.degrees(z['q'][:,12]),color=color)
axs[1].step(original['control_t'],np.degrees(original['control_target'][:,12]),where='post',color='white',ls='--',label='zadana, wspólna')
axs[0].set_ylabel('Moment silnika FL_calf [Nm]');axs[1].set_ylabel('Pozycja FL_calf [°]')
for ax in axs:ax.set_xlim(.05,.3);ax.set_xlabel('Czas [s]');ax.grid(alpha=.3);ax.legend(frameon=False,labelcolor='#dce4ec')
fig.suptitle('Te same komendy, różny krok fizyki · latest / roll +30° / h=2,5 m')
fig.savefig(p/'timestep_probe.svg');fig.savefig(p/'timestep_probe.png',dpi=140)
