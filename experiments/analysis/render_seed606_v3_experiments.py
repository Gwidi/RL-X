"""Matched three-panel examples selected from the complete controlled grid."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.evaluate_seed606_v3 import Experiment, OUT
from experiments.analysis.report_landing_seed606 import load_policy, make_report_env, configure_landing_collider_view,update_landing_collider_view,sha,FONT
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def render(case,label,actions):
    names=['zero','best','latest'];experiments=[Experiment() for _ in names]
    observations=[e.reset(case) for e in experiments]
    hashes=[e.wrapper.case_info['scenario_hash'] for e in experiments]
    assert len(set(hashes))==1
    slug='example_'+label;path=OUT/(slug+'.mp4')
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1500x720','-r','40','-i','-','-an','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    done=[False]*3;outcomes=[None]*3;next_frame=0.;font=ImageFont.truetype(FONT,19);small=ImageFont.truetype(FONT,16)
    try:
        with contextlib.ExitStack() as stack:
            renderers=[];cameras=[]
            for ex in experiments:
                model=ex.wrapper.env.internal_state['mj_model'];configure_landing_collider_view(model)
                model.vis.global_.offwidth=500;model.vis.global_.offheight=500
                renderers.append(stack.enter_context(mujoco.Renderer(model,height=500,width=500)))
                camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);cameras.append(camera)
            for step in range(250):
                frame=Image.new('RGB',(1500,720),(17,23,30));draw=ImageDraw.Draw(frame)
                draw.text((18,9),f'PD606 / BEST / LATEST  |  h={case["height"]:g} m  vx={case["vx"]:.2f}  vy={case["vy"]:.2f} m/s  roll={case["roll"]:.1f}  pitch={case["pitch"]:.1f} deg',font=font,fill='white')
                draw.text((18,39),'Ten sam model, PD i stan poczatkowy. Film 4x wolniej. Obraz zatrzymany po rozstrzygnieciu epizodu.',font=small,fill=(163,177,191))
                for i,(name,ex,renderer,camera) in enumerate(zip(names,experiments,renderers,cameras)):
                    env=ex.wrapper.env
                    if not done[i]:
                        a=np.zeros(13) if name=='zero' else actions[name](observations[i])
                        observations[i],_,terminated,truncated,_=ex.wrapper.step(a)
                        done[i]=bool(terminated or truncated)
                        if done[i]:outcomes[i]=dict(controller=name,success=bool(env.internal_state['landing_success']),reason=env.internal_state['landing_failure_reason'],end_time=float(env.internal_state['data'].time))
                    model=env.internal_state['mj_model'];data=env.internal_state['data'];monitor=env.internal_state['landing_safety']
                    update_landing_collider_view(model,data)
                    camera.lookat[:]=data.qpos[:3]+np.array([0,0,-.05]);camera.distance=1.15;camera.azimuth=120;camera.elevation=-12
                    renderer.update_scene(data,camera=camera);frame.paste(Image.fromarray(renderer.render()),(500*i,94))
                    draw.text((500*i+15,68),['PD / zerowa akcja','PPO / best','PPO / latest'][i],font=font,fill=(132,212,192))
                    status=('SUKCES' if env.internal_state['landing_success'] else 'PORAZKA: '+env.internal_state['landing_failure_reason']) if done[i] else 'W TRAKCIE'
                    draw.text((500*i+15,607),status,font=small,fill=(132,212,192) if env.internal_state['landing_success'] else (237,190,133))
                    draw.text((500*i+15,636),f't={data.time:.3f} s  stopa={monitor.peak_foot.max():.0f} N / 10 ms',font=small,fill='white')
                    draw.text((500*i+15,665),f'acc={monitor.peak_acceleration:.1f} m/s2 / 50 ms',font=small,fill=(163,177,191))
                while next_frame<=(step+1)*.02:
                    process.stdin.write(np.asarray(frame).tobytes());next_frame+=.25/40
                if all(done):break
            for _ in range(80):process.stdin.write(np.asarray(frame).tobytes())
            frame.save(OUT/(slug+'.png'))
    finally:
        process.stdin.close();code=process.wait()
        for ex in experiments:ex.close()
        if code:raise RuntimeError('ffmpeg failed')
    return dict(label=label,case=case,video=path.name,poster=slug+'.png',video_sha256=sha(path),scenario_hash=hashes[0],outcomes=outcomes)


def main():
    rows=json.loads((OUT/'results.json').read_text());by={(r['controller'],r['case']['id']):r for r in rows}
    wins=[r for r in rows if r['controller']=='best' and r['case']['kind']=='speed' and r['success'] and not by[('zero',r['case']['id'])]['success']]
    assert wins,'No successful policy-only example in controlled grid'
    good=max(wins,key=lambda r:(r['case']['height'],r['case']['speed']))['case']
    failures=[r for r in rows if r['controller']=='best' and r['case']['kind']=='speed' and not r['success'] and r['case']['height']==good['height']]
    bad=max(failures,key=lambda r:(r['case']['speed'],r['case']['direction']==good['direction']))['case']
    loader=make_report_env();actions={n:load_policy(OUT/f'{n}.model',loader.env)[0] for n in ['best','latest']};loader.close()
    videos=[]
    for case,label in [(good,'policy_improvement'),(bad,'policy_limit')]:
        result=render(case,label,actions)
        for outcome in result['outcomes']:
            expected=by[(outcome['controller'],case['id'])]
            assert outcome['success']==expected['success']
            assert abs(outcome['end_time']-expected['end_time'])<1e-9
            assert result['scenario_hash']==expected['scenario_hash']
        videos.append(result);print(label,case,result['outcomes'],flush=True)
    (OUT/'videos.json').write_text(json.dumps(videos,indent=2))


if __name__=='__main__':main()
