"""Physics-step replays of every fresh v8 3m/4mps case, without outcome selection."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.analyze_v8_download import OUT,V8Experiment
from experiments.analysis.report_landing_seed606 import load_policy

def main():
 rows=json.loads((OUT/'fresh_results.json').read_text());cases=[r for r in rows if r['controller']=='best' and r['case']['height']==3 and r['case']['speed']==4]
 ex=V8Experiment();action=load_policy(OUT/'best.model',ex.wrapper.env)[0];records=[];traces={}
 try:
  for i,expected in enumerate(cases):
   obs=ex.reset(expected['case']);env=ex.wrapper.env;s=env.internal_state;m=s['mj_model'];mon=s['landing_safety'];original=mon.update;trace=[]
   def update(model,d):
    forces=original(model,d)
    trace.append(dict(t=float(d.time),acc=float(mon.acceleration),raw_acc=float(np.linalg.norm(d.qacc[:3])),foot=forces[2].tolist(),z=float(d.qpos[2]),vz=float(d.qvel[2]),vxy=float(np.linalg.norm(d.qvel[:2])),joints=d.qpos[7:].tolist(),joint_velocity=d.qvel[6:].tolist(),motor=d.actuator_force.tolist()))
    return forces
   mon.update=update
   for _ in range(env.horizon):
    obs,_,term,trunc,_=ex.wrapper.step(action(obs))
    if term or trunc:break
   assert bool(s['landing_success'])==expected['success'] and s['landing_failure_reason']==expected['reason']
   assert float(s['data'].time)==expected['end_time'] and ex.wrapper.case_info['scenario_hash']==expected['scenario_hash']
   first=next(k for k,r in enumerate(trace) if max(r['foot'])>1);peak=max(range(len(trace)),key=lambda k:trace[k]['acc']);onset=trace[first];top=trace[peak]
   contacts=[next((r['t'] for r in trace if r['foot'][j]>1),None) for j in range(4)]
   rec=dict(case=expected['case'],success=expected['success'],reason=expected['reason'],peak_acc=top['acc'],first_loaded_foot_time=onset['t'],peak_time=top['t'],peak_after_contact_ms=1000*(top['t']-onset['t']),first_contact_foot_times=contacts,feet_at_peak=sum(f>1 for f in top['foot']),foot_forces_peak=top['foot'],z_at_contact=onset['z'],z_at_peak=top['z'],vz_at_contact=onset['vz'],vz_at_peak=top['vz'],speed_at_contact=onset['vxy'],speed_at_peak=top['vxy'],joint_change_contact_to_peak_deg=(np.degrees(np.array(top['joints'])-onset['joints'])).tolist(),max_leg_motor_torque=max(abs(v) for r in trace for v in r['motor'][1:]))
   records.append(rec);traces[str(expected['case']['id'])]=trace
   if (i+1)%16==0:print(i+1,flush=True)
 finally:ex.close()
 (OUT/'impact_diagnostics.json').write_text(json.dumps(dict(records=records,foot_order=mon.contacts.foot_legs,sample_dt=.0025,contact_definition='first foot force >1N; time before/at 50ms safety window maximum',note='All64 matched original outcome, time and physical hash'),indent=2))
 (OUT/'impact_traces.json').write_text(json.dumps(traces))
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for r in records:
  trace=traces[str(r['case']['id'])];t=np.array([x['t'] for x in trace])-r['first_loaded_foot_time'];sel=(t>=-.02)&(t<=.25);color='tab:blue' if r['success'] else 'tab:red'
  axes[0].plot(t[sel]*1000,np.array([x['acc'] for x in trace])[sel],color=color,alpha=.35)
  axes[1].scatter(r['case']['direction'],r['peak_acc'],color=color,s=22)
  axes[2].scatter(r['peak_after_contact_ms'],r['z_at_contact']-r['z_at_peak'],color=color,s=22)
 axes[0].axhline(98.1,color='black',ls='--');axes[0].set(xlabel='Time since first foot load [ms]',ylabel='Safety acceleration, 50 ms [m/s²]')
 axes[1].axhline(98.1,color='black',ls='--');axes[1].set(xlabel='Initial direction [deg]',ylabel='Peak safety acceleration [m/s²]')
 axes[2].set(xlabel='Delay of peak after first foot load [ms]',ylabel='Base vertical travel to peak [m]')
 fig.suptitle('64 fresh cases at 3 m / 4 m/s: blue success, red failure');fig.tight_layout();fig.savefig(OUT/'impact_diagnostics.png',dpi=150)
 summary={}
 for success in [True,False]:
  g=[r for r in records if r['success']==success];summary[str(success)]=dict(count=len(g),**{key:dict(min=min(r[key] for r in g),median=float(np.median([r[key] for r in g])),max=max(r[key] for r in g)) for key in ['peak_acc','peak_after_contact_ms','feet_at_peak','z_at_contact','vz_at_contact','max_leg_motor_torque']})
 (OUT/'impact_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
