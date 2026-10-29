"""Summarize paired v3 experiments and draw scientific figures."""
import collections
import json
from pathlib import Path
import math
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v3_checkpoint_experiments_2026-10-26'
CONTROLLERS=['zero','best','latest']
HEIGHTS=[1.,1.5,2.,2.5]


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def main():
    rows=json.loads((OUT/'results.json').read_text())
    assert len(rows)==1488
    refinement=json.loads((OUT/'refinement_results.json').read_text())
    region=json.loads((OUT/'region_results.json').read_text())
    assert len(refinement)==96 and len(region)==384
    by=collections.defaultdict(list)
    for row in rows:by[(row['controller'],row['case']['kind'])].append(row)
    summary={'total_episodes':len(rows)+len(refinement)+len(region),'main_episodes':len(rows),'refinement_episodes':len(refinement),'region_episodes':len(region),'per_controller':{},'speed_grid':[], 'heldout':[], 'limited_region':[], 'pairing':{}}
    for name in CONTROLLERS:
        summary['per_controller'][name]={}
        for kind in ['nominal','speed','angle','heldout']:
            selected=by[(name,kind)];k=sum(r['success'] for r in selected)
            summary['per_controller'][name][kind]=dict(successes=k,count=len(selected),rate=k/len(selected))
        for h in HEIGHTS:
            selected=[r for r in by[(name,'heldout')] if r['case']['height']==h];k=sum(r['success'] for r in selected)
            summary['heldout'].append(dict(controller=name,height=h,successes=k,count=len(selected),ci95=wilson(k,len(selected))))
            for speed in [.5,1.,1.5,2.,2.5,3.]:
                selected=[r for r in by[(name,'speed')] if r['case']['height']==h and r['case']['speed']==speed]
                summary['speed_grid'].append(dict(controller=name,height=h,speed=speed,successes=sum(r['success'] for r in selected),count=len(selected),successful_directions=[r['case']['direction'] for r in selected if r['success']]))
            selected=[r for r in refinement if r['controller']==name and r['case']['height']==h]
            summary['speed_grid'].append(dict(controller=name,height=h,speed=1.25,successes=sum(r['success'] for r in selected),count=len(selected),successful_directions=[r['case']['direction'] for r in selected if r['success']]))
            selected=[r for r in region if r['controller']==name and r['case']['height']==h];k=sum(r['success'] for r in selected)
            summary['limited_region'].append(dict(controller=name,height=h,successes=k,count=len(selected),ci95=wilson(k,len(selected))))
        for kind,extra in [('limited_region',region)]:
            selected=[r for r in extra if r['controller']==name];k=sum(r['success'] for r in selected)
            summary['per_controller'][name][kind]=dict(successes=k,count=len(selected),rate=k/len(selected),ci95=wilson(k,len(selected)))
        failures=[r for r in rows if r['controller']==name and not r['success']]
        summary['per_controller'][name]['failure_reasons']=dict(collections.Counter(r['reason'] for r in failures))
        summary['per_controller'][name]['first_forbidden_geoms']=dict(collections.Counter(' / '.join(r['forbidden'][0]['geoms']) for r in failures if r['forbidden']))
    for a,b in [('best','latest'),('best','zero'),('latest','zero')]:
        aa={r['case']['id']:r for r in by[(a,'heldout')]};bb={r['case']['id']:r for r in by[(b,'heldout')]}
        assert aa.keys()==bb.keys()
        assert all(aa[k]['scenario_hash']==bb[k]['scenario_hash'] for k in aa)
        wins=sum(aa[k]['success'] and not bb[k]['success'] for k in aa)
        losses=sum(bb[k]['success'] and not aa[k]['success'] for k in aa)
        summary['pairing'][f'{a}_vs_{b}']=dict(a_only=wins,b_only=losses,discordant=wins+losses,exact_two_sided_p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.)
    all_rows=rows+refinement+region
    for a,b in [('best','zero'),('latest','zero')]:
        aa={r['case']['id']:r for r in all_rows if r['controller']==a};bb={r['case']['id']:r for r in all_rows if r['controller']==b}
        assert aa.keys()==bb.keys() and len(aa)==656
        assert all(aa[k]['scenario_hash']==bb[k]['scenario_hash'] for k in aa)
    summary['paired_case_count']=656
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#dce4ec','axes.labelcolor':'#dce4ec','xtick.color':'#a3b1bf','ytick.color':'#a3b1bf','axes.edgecolor':'#34404b','axes.facecolor':'#11171e','figure.facecolor':'#11171e','grid.color':'#34404b','svg.fonttype':'none'})
    colors={'zero':'#a3b1bf','best':'#84d4c0','latest':'#edbe85'}
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,h in zip(axs.flat,HEIGHTS):
        for name in CONTROLLERS:
            values=[1. if next(r for r in by[(name,'nominal')] if r['case']['height']==h)['success'] else 0.]
            selected=sorted([r for r in summary['speed_grid'] if r['controller']==name and r['height']==h],key=lambda r:r['speed'])
            values += [r['successes']/r['count'] for r in selected]
            ax.plot([0]+[r['speed'] for r in selected],100*np.array(values),marker='o',label=name,color=colors[name],lw=2)
        ax.set(title=f'Wysokość bazy {h:g} m',xlabel='Moduł prędkości XY [m/s]',ylabel='Udane kierunki [%]',ylim=(-3,103));ax.grid(alpha=.4)
    axs[0,0].legend(frameon=False,labelcolor='#dce4ec');fig.suptitle('Bez przechyłu · 8 kierunków na każdą prędkość',fontsize=15)
    fig.savefig(OUT/'speed_envelope.svg');fig.savefig(OUT/'speed_envelope.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    from matplotlib.colors import ListedColormap
    for ax,name in zip(axs,['best','latest']):
        controlled=by[(name,'speed')]+[r for r in refinement if r['controller']==name]
        grid=np.array([[next(r['success'] for r in controlled if r['case']['height']==2.5 and r['case']['direction']==direction and r['case']['speed']==speed) for speed in [.5,1,1.25,1.5,2,2.5,3]] for direction in range(0,360,45)])
        ax.imshow(grid.astype(int),cmap=ListedColormap(['#49353a','#64b7a1']),vmin=0,vmax=1,aspect='auto')
        ax.set(xticks=range(7),xticklabels=['0,5','1','1,25','1,5','2','2,5','3'],yticks=range(8),yticklabels=['0° (+x)','45°','90° (+y)','135°','180° (−x)','225°','270° (−y)','315°'],xlabel='Moduł prędkości [m/s]',title=name)
        for i in range(8):
            for j in range(7):ax.text(j,i,'✓' if grid[i,j] else '×',ha='center',va='center',color='#f3f6fa',fontsize=14)
    fig.suptitle('2,5 m · bez przechyłu · sukces i porażka zależnie od kierunku',fontsize=14)
    fig.savefig(OUT/'direction_map.svg');fig.savefig(OUT/'direction_map.png',dpi=150);plt.close(fig)
    print(json.dumps(summary['per_controller'],indent=2))
    print('PAIRING',summary['pairing'])


if __name__=='__main__':main()
