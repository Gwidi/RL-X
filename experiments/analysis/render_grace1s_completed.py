"""Same noisy sensor for both v6 and v7, no outcome-based video selection."""
import sys,json,contextlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.analysis.analyze_v9_download import V9Experiment
from experiments.analysis.analyze_grace1s_download import OUT,OLD,Grace1sExperiment
from experiments.analysis.report_landing_seed606 import load_policy
from experiments.analysis.evaluate_v6_download import scenarios
from experiments.analysis.report_landing_seed606 import sha,FONT,configure_landing_collider_view,update_landing_collider_view
MEDIA=Path(__file__).resolve().parents[2]/'docs/report/media'
def render_montage(cases,label,actions,controllers):
    path=MEDIA/f'etap22-{label}.mp4';font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,18);title=ImageFont.truetype(FONT,29)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1400x820','-r','25','-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    clips=[];frame_count=0
    try:
        for clip,c in enumerate(cases):
            exs=[V9Experiment(),Grace1sExperiment()];obs=[e.reset(c) for e in exs];hashes=[e.wrapper.case_info['scenario_hash'] for e in exs];assert hashes[0]==hashes[1]
            done=[False,False];outcomes=[None,None];start=frame_count/25
            try:
                with contextlib.ExitStack() as stack:
                    renderers=[];cameras=[]
                    for ex in exs:
                        m=ex.wrapper.env.internal_state['mj_model'];configure_landing_collider_view(m);m.vis.global_.offwidth=700;m.vis.global_.offheight=520
                        renderers.append(stack.enter_context(mujoco.Renderer(m,height=520,width=700)))
                        cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=1.25;cam.azimuth=120;cam.elevation=-13;cameras.append(cam)
                    for step in range(250):
                        image=Image.new('RGB',(1400,820),(17,23,30));draw=ImageDraw.Draw(image)
                        heading='Body motion penalty: V9 (0.5 s) vs V10 (1 s)'
                        draw.text((22,14),heading,font=title,fill='white')
                        draw.text((22,57),f'{clip+1}/{len(cases)}  h={c["height"]:g} m   roll={c["roll"]:+.1f}°   pitch={c["pitch"]:+.1f}°   vx={c["vx"]:+.2f}   vy={c["vy"]:+.2f} m/s',font=font,fill='#84d4c0')
                        draw.text((22,94),'implicitfast 2,5 ms • sterowanie 50 Hz • film 4× wolniej • obraz zatrzymuje się po wyniku próby',font=small,fill='#a3b1bf')
                        for i,(ex,renderer,cam) in enumerate(zip(exs,renderers,cameras)):
                            env=ex.wrapper.env
                            if not done[i]:
                                a=np.zeros(13) if actions[i] is None else actions[i](obs[i]);obs[i],_,term,trunc,_=ex.wrapper.step(a);done[i]=bool(term or trunc)
                                if done[i]:outcomes[i]=dict(controller=controllers[i],success=bool(env.internal_state['landing_success']),reason=env.internal_state['landing_failure_reason'],end_time=float(env.internal_state['data'].time))
                            m=env.internal_state['mj_model'];d=env.internal_state['data'];mon=env.internal_state['landing_safety'];update_landing_collider_view(m,d);cam.lookat[:]=d.qpos[:3]+[0,0,-.05]
                            renderer.update_scene(d,camera=cam);image.paste(Image.fromarray(renderer.render()),(700*i,172))
                            draw.text((700*i+22,134),{'v9':'V9 best / kara od 0,5 s','grace1s':'V10 grace1s / best = latest'}[controllers[i]],font=font,fill='white')
                            status=('SUKCES' if env.internal_state['landing_success'] else 'PORAŻKA: '+env.internal_state['landing_failure_reason']) if done[i] else 'W TRAKCIE'
                            draw.text((700*i+22,707),status,font=font,fill='#84d4c0' if env.internal_state['landing_success'] else '#edbe85')
                            draw.text((700*i+22,746),f't={d.time:.3f} s   stopa: {mon.peak_foot.max():.0f} N / 10 ms   acc: {mon.peak_acceleration:.1f} m/s²',font=small,fill='#a3b1bf')
                        draw.text((22,789),'Zielony: stopy • niebieski: dolna 1/3 łydki • pomarańczowy/czerwony: górna 2/3 łydki',font=small,fill='#a3b1bf')
                        for _ in range(2):proc.stdin.write(np.asarray(image).tobytes());frame_count+=1
                        if all(done):break
                    for _ in range(25):proc.stdin.write(np.asarray(image).tobytes());frame_count+=1
                    if clip==0:image.save(path.with_suffix('.png'))
            finally:
                for ex in exs:ex.close()
            clips.append(dict(case=c,start_seconds=start,outcomes=outcomes,scenario_hash=hashes[0]));print(label,c['height'],outcomes,flush=True)
    finally:
        proc.stdin.close();code=proc.wait()
        if code:raise RuntimeError('ffmpeg failed')
    return dict(label=label,src='media/'+path.name,poster='media/'+path.with_suffix('.png').name,sha256=sha(path),duration_seconds=frame_count/25,clips=clips)

def main():
 old=json.loads((OLD/'paired_results.json').read_text());new=json.loads((OUT/'paired_results.json').read_text());cases=json.loads((OUT/'paired_protocol.json').read_text())['cases']
 native=[V9Experiment(),Grace1sExperiment()]
 try:actions=[load_policy(OLD/'best.model',native[0].wrapper.env)[0],load_policy(OUT/'best.model',native[1].wrapper.env)[0]]
 finally:
  for e in native:e.close()
 first4=next(r['case'] for r in old if r['controller']=='best' and r['case']['height']==3 and r['case']['speed']==4 and r['success'])
 failed=next(r['case'] for r in old if r['controller']=='best' and r['case']['height']==3 and r['case']['speed']==4 and not r['success'])
 first5=next(c for c in cases if c['height']==3 and c['speed']==5)
 success5=next(r['case'] for r in old if r['controller']=='best' and r['case']['height']==3 and r['case']['speed']==5 and r['success'])
 assert first4['id']!=failed['id'] and first5['id']!=success5['id']
 movies=[]
 for label,selected in [('grace1s-4mps',[first4,failed]),('grace1s-5mps',[first5,success5])]:
  movie=render_montage(selected,label,actions,['v9','grace1s'])
  for clip in movie['clips']:
   for name,factory,callback,source in [('v9',V9Experiment,actions[0],old),('grace1s',Grace1sExperiment,actions[1],new)]:
    e=factory()
    try:r=e.trial(clip['case'],callback)
    finally:e.close()
    expected=next(row for row in source if row['controller']=='best' and row['case']['id']==clip['case']['id']);shown=next(o for o in clip['outcomes'] if o['controller']==name)
    for key in ['success','reason','end_time']:assert r[key]==expected[key]==shown[key]
    assert r['scenario_hash']==expected['scenario_hash']==clip['scenario_hash']
  movie['verified']=True;movie['selection']='First v9-best success and failure at3m/4; first predeclared3m/5 case and first v9-best success at3m/5. No filtering on grace1s outcome; illustrations not success-rate evidence.';movies.append(movie)
 (OUT/'videos.json').write_text(json.dumps(movies,indent=2))
if __name__=='__main__':main()
