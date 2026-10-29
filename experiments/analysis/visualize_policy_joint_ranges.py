"""Kinematic suspended-robot sweep of the exact policy target mapping.
No integration or policy inference: isolates available joint-position commands.
"""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.analysis.report_landing_seed606 import FONT
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/policy_joint_ranges'
MEDIA=ROOT/'docs/report/media'
FPS=20
SECONDS=6
W,H=1600,900


def sweep(t):
    # Home -> -1, hold -> +1, hold -> home.
    times=[0,.6,1.6,2.1,4.1,4.6,5.6,6.]
    values=[0,0,-1,-1,1,1,0,0]
    for k in range(len(times)-1):
        if t<=times[k+1]:
            u=np.clip((t-times[k])/(times[k+1]-times[k]),0,1)
            return values[k]+(values[k+1]-values[k])*(1-np.cos(np.pi*u))/2
    return 0.


def main():
    OUT.mkdir(parents=True,exist_ok=True);MEDIA.mkdir(parents=True,exist_ok=True)
    ex=Experiment();ex.reset(dict(id=0,height=2.5,roll=0,pitch=0,vx=0,vy=0))
    e=ex.wrapper.env;s=e.internal_state;m=s['mj_model'];d=s['data']
    joints=np.asarray(e.actuator_joint_mask_joints);addresses=m.jnt_qposadr[joints]
    home=s['actuator_joint_nominal_positions'].copy();scale=np.broadcast_to(s['scaling_factor'],home.shape)
    offsets=s['position_offsets'].copy();assert np.all(offsets==0)
    gear=m.actuator_gear[:,0];assert len(gear)==len(joints) and np.all(gear!=0)
    def target(a):return e.control_function.process_action(a)/gear
    lo=target(-np.ones(13));hi=target(np.ones(13));qhome=target(np.zeros(13))
    expected_lo=np.where(m.jnt_limited[joints],np.clip(home-scale,m.jnt_range[joints,0],m.jnt_range[joints,1]),home-scale)
    expected_hi=np.where(m.jnt_limited[joints],np.clip(home+scale,m.jnt_range[joints,0],m.jnt_range[joints,1]),home+scale)
    np.testing.assert_allclose(lo,expected_lo);np.testing.assert_allclose(hi,expected_hi)
    records=[]
    for i,j in enumerate(joints):
        records.append(dict(index=i,joint=m.joint(int(j)).name,home_rad=float(qhome[i]),min_rad=float(lo[i]),max_rad=float(hi[i]),mechanical_min_rad=float(m.jnt_range[j,0]),mechanical_max_rad=float(m.jnt_range[j,1]),scale_rad=float(scale[i]),clipped_low=bool(home[i]-scale[i]<lo[i]-1e-9),clipped_high=bool(home[i]+scale[i]>hi[i]+1e-9),start_seconds=i*SECONDS))
    (OUT/'ranges.json').write_text(json.dumps(dict(mode='kinematic; fixed free-base; no dynamics; one joint at a time',fps=FPS,joint_duration_seconds=SECONDS,joints=records),indent=2))
    m.vis.global_.offwidth=600;m.vis.global_.offheight=570
    # Hide contact spheres that obscure the robot's visual surfaces.
    cameras=[]
    for azimuth in [120,300]:
        c=mujoco.MjvCamera();mujoco.mjv_defaultCamera(c);c.lookat[:]=d.qpos[:3]+[0,0,-.10];c.distance=1.55;c.azimuth=azimuth;c.elevation=-18;cameras.append(c)
    initial=d.qpos.copy();font=ImageFont.truetype(FONT,23);small=ImageFont.truetype(FONT,18);title=ImageFont.truetype(FONT,32)
    dest=MEDIA/'policy_joint_ranges.mp4'
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','22','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(m,height=570,width=600) as renderer:
            for i,record in enumerate(records):
                for frame_id in range(FPS*SECONDS):
                    a=np.zeros(13);a[i]=sweep(frame_id/FPS);q=target(a)
                    d.qpos[:]=initial;d.qpos[addresses]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
                    np.testing.assert_array_equal(d.qpos[:7],initial[:7])
                    frame=Image.new('RGB',(W,H),(17,23,30));draw=ImageDraw.Draw(frame)
                    draw.text((24,18),'Zakres ruchu dostępny polityce PPO',font=title,fill='white')
                    draw.text((24,64),'Korpus zawieszony • stawy ustawiane kinematycznie • pozostałe stawy w home',font=font,fill='#a3b1bf')
                    draw.text((24,105),f'{i+1:02d}/13  {record["joint"]}     akcja: {a[i]:+.2f}     pozycja: {np.degrees(q[i]):+.1f}°',font=font,fill='#84d4c0')
                    for view,camera in enumerate(cameras):
                        renderer.update_scene(d,camera=camera)
                        idx=renderer.scene.ngeom
                        if idx<renderer.scene.maxgeom:
                            mujoco.mjv_initGeom(renderer.scene.geoms[idx],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.018]*3),d.xanchor[joints[i]],np.eye(3).ravel(),np.array([1.,.65,.15,1.]));renderer.scene.ngeom+=1
                        frame.paste(Image.fromarray(renderer.render()),(view*600,155))
                        draw.text((24+600*view,161),f'Widok {view+1} • przeciwne strony',font=small,fill='white')
                    draw.text((1220,164),'STAWY / ZAKRES POLITYKI',font=small,fill='#a3b1bf')
                    for k,r in enumerate(records):
                        y=202+40*k;fill='#84d4c0' if k==i else '#a3b1bf'
                        if k==i:draw.rectangle((1210,y-3,1590,y+35),fill='#263a40')
                        name=r['joint'].replace('_joint','')
                        draw.text((1220,y),name,font=small,fill=fill)
                        draw.text((1390,y),f'{np.degrees(r["min_rad"]):+.0f}…{np.degrees(r["max_rad"]):+.0f}°',font=small,fill=fill)
                    left,right=55,1140;y=775
                    mn,mx=m.jnt_range[joints[i]]
                    def xx(value):return left+(right-left)*(value-mn)/(mx-mn)
                    draw.line((left,y,right,y),fill='#54616d',width=9)
                    draw.line((xx(lo[i]),y,xx(hi[i]),y),fill='#84d4c0',width=13)
                    draw.line((xx(qhome[i]),y-18,xx(qhome[i]),y+18),fill='white',width=3)
                    draw.ellipse((xx(q[i])-9,y-9,xx(q[i])+9,y+9),fill='#edbe85')
                    draw.text((30,731),f'Limity stawu: {np.degrees(mn):+.1f}° … {np.degrees(mx):+.1f}°   |   Home: {np.degrees(qhome[i]):+.1f}°',font=small,fill='#a3b1bf')
                    draw.text((30,808),f'Zakres polityki: {np.degrees(lo[i]):+.1f}° … {np.degrees(hi[i]):+.1f}°   |   zielony: dostępny zakres; biała kreska: home',font=small,fill='#84d4c0')
                    draw.text((30,850),'To geometria pozycji zadanych, nie test siły, prędkości ani wykonalności ruchu w locie. Żółta kropka wskazuje aktywny staw.',font=small,fill='#a3b1bf')
                    proc.stdin.write(np.asarray(frame).tobytes())
                    if frame_id==82:
                        frame.save(OUT/f'joint_{i:02d}.png')
                        if i==0:frame.save(MEDIA/'policy_joint_ranges.png')
                print(record['joint'],f'{np.degrees(lo[i]):.1f} to {np.degrees(hi[i]):.1f} deg',flush=True)
    finally:
        proc.stdin.close();code=proc.wait();ex.close()
        if code:raise RuntimeError('Video encoding failed')
    print(dest,flush=True)

if __name__=='__main__':main()
