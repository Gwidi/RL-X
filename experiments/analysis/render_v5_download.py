"""Matched v4/v5 video montages on four heights."""
import sys,json,contextlib,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.analysis.evaluate_v4_download import V4Experiment,HEIGHTS
from experiments.analysis.analyze_v5_download import OUT
from experiments.analysis.evaluate_v5_download import V5Experiment,OLD
from experiments.analysis.report_landing_seed606 import load_policy,sha,FONT,configure_landing_collider_view,update_landing_collider_view
MEDIA=Path(__file__).resolve().parents[2]/'docs/report/media'

def render_montage(cases,label,actions,checkpoint):
    path=MEDIA/f'etap11-{checkpoint}-{label}.mp4';font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,18);title=ImageFont.truetype(FONT,29)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1400x820','-r','25','-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    clips=[];frame_count=0
    try:
        for clip,c in enumerate(cases):
            exs=[V4Experiment(),V5Experiment()];obs=[e.reset(c) for e in exs];hashes=[e.wrapper.case_info['scenario_hash'] for e in exs];assert hashes[0]==hashes[1]
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
                        heading='PPO v4 bez prędkości i PPO v5 z prędkością — identyczny stan'
                        draw.text((22,14),heading,font=title,fill='white')
                        draw.text((22,57),f'{clip+1}/{len(cases)}  h={c["height"]:g} m   roll={c["roll"]:+.1f}°   pitch={c["pitch"]:+.1f}°   vx={c["vx"]:+.2f}   vy={c["vy"]:+.2f} m/s',font=font,fill='#84d4c0')
                        draw.text((22,94),'implicitfast 2,5 ms • sterowanie 50 Hz • film 4× wolniej • obraz zatrzymuje się po wyniku próby',font=small,fill='#a3b1bf')
                        for i,(ex,renderer,cam) in enumerate(zip(exs,renderers,cameras)):
                            env=ex.wrapper.env
                            if not done[i]:
                                a=actions[i](obs[i]);obs[i],_,term,trunc,_=ex.wrapper.step(a);done[i]=bool(term or trunc)
                                if done[i]:outcomes[i]=dict(controller=['v4','v5_'+checkpoint][i],success=bool(env.internal_state['landing_success']),reason=env.internal_state['landing_failure_reason'],end_time=float(env.internal_state['data'].time))
                            m=env.internal_state['mj_model'];d=env.internal_state['data'];mon=env.internal_state['landing_safety'];update_landing_collider_view(m,d);cam.lookat[:]=d.qpos[:3]+[0,0,-.05]
                            renderer.update_scene(d,camera=cam);image.paste(Image.fromarray(renderer.render()),(700*i,172))
                            draw.text((700*i+22,134),['PPO v4 / bez prędkości',f'PPO v5 / {checkpoint} / z prędkością'][i],font=font,fill='white')
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
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',choices=['best','latest'],default='latest');args=parser.parse_args();checkpoint=args.checkpoint
    cases=json.loads((OLD/'test_protocol.json').read_text())['cases']
    loaders=[V4Experiment(),V5Experiment()]
    actions=[load_policy(path,ex.wrapper.env)[0] for path,ex in zip([OLD/'best.model',OUT/f'{checkpoint}.model'],loaders)]
    for ex in loaders:ex.close()
    movies=[]
    for kind,label in [('speed_grid','predkosc-3ms'),('random_combined','pierwsze-proby-laczone')]:
        selected=[next(c for c in cases if c['height']==h and c['kind']==kind and (kind!='speed_grid' or c['speed']==3 and c['direction']==0)) for h in HEIGHTS]
        v=render_montage(selected,label,actions,checkpoint)
        v['selection']='First predefined scenario per height; speed grid uses 3m/s forward, zero tilt. Combined uses first archived random case; no selection by outcome.'
        movies.append(v)
    (OUT/f'{checkpoint}_videos.json').write_text(json.dumps(movies,indent=2))
    if checkpoint=='latest':(OUT/'videos.json').write_text(json.dumps(movies,indent=2))
if __name__=='__main__':main()
