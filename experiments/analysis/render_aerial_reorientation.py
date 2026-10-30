"""Render recorded free-flight states; camera follows the freely falling base."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.analysis.probe_aerial_reorientation import Probe,OUT,ROOT
from experiments.analysis.report_landing_seed606 import FONT

def main():
    data=json.loads((OUT/'results.json').read_text());p=Probe()
    keys=['zero','roll_hold','pitch_hold','roll'];labels=['Zero action','Roll: move and hold','Pitch: move and hold','Roll: cycle back home']
    traces=[];positions=[]
    for key in keys:
        c=data['rows'][0]['candidate'] if key=='zero' else data['selected'][key]['candidate']
        result,t,q=p.trial(c);traces.append(t);positions.append(q)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for i,ax in enumerate(axes):
        for label,t in zip(labels,traces):ax.plot(t[:,0],t[:,1+i],label=label)
        ax.set(xlabel='Time [s]',ylabel=['Rotation vector X [deg]','Rotation vector Y [deg]','Rotation vector Z [deg]'][i]);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'orientation.png',dpi=150);plt.close(fig)
    media=ROOT/'docs/report/media';media.mkdir(exist_ok=True)
    dest=media/'branch-aerial-feasibility.mp4';font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,16)
    m,d=p.model,p.data;m.vis.global_.offwidth=640;m.vis.global_.offheight=330
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=1.65;cam.azimuth=125;cam.elevation=-20
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x860','-r','12.5','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(m,height=330,width=640) as renderer:
            for k in list(range(len(traces[0])))+[len(traces[0])-1]*12:
                frame=Image.new('RGB',(1280,860),(17,23,30));draw=ImageDraw.Draw(frame)
                draw.text((20,12),'Free base / gravity ON / vacuum / actual PD / 4x slow motion',font=font,fill='white')
                draw.text((20,43),'Camera follows falling robot. This is not a suspension or a successful landing.',font=small,fill='#a3b1bf')
                for j,label in enumerate(labels):
                    x=(j%2)*640;y=80+(j//2)*385
                    d.qpos[:]=positions[j][k];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3]+[0,0,-.1]
                    renderer.update_scene(d,camera=cam);frame.paste(Image.fromarray(renderer.render()),(x,y+44))
                    t=traces[j][k];draw.text((x+15,y),label,font=font,fill='#84d4c0')
                    draw.text((x+15,y+26),f't={t[0]:.2f}s | rot=[{t[1]:+.1f}, {t[2]:+.1f}, {t[3]:+.1f}] deg | w={t[4]:.2f} rad/s',font=small,fill='white')
                proc.stdin.write(np.asarray(frame).tobytes())
                if k==len(traces[0])-1:frame.save(media/'branch-aerial-feasibility.png')
    finally:
        proc.stdin.close();code=proc.wait();p.close()
        if code:raise RuntimeError('ffmpeg failed')
    print(dest)
if __name__=='__main__':main()
