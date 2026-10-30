"""Fixed-speed validation and empirical >80% operating boundaries for v8."""
import csv
import json
from pathlib import Path
import numpy as np


def wilson(successes, count):
    z=1.95996398454;p=successes/count;d=1+z*z/count
    c=(p+z*z/(2*count))/d;r=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return [float(c-r),float(c+r)]


def boundary_summary(cells, heights, speeds, threshold):
    lookup={(c['height'],c['speed']):c for c in cells if c['controller']=='policy'}
    result={}
    for height in heights:
        contiguous=None
        for speed in speeds:
            if lookup[height,speed]['success_rate'] <= threshold:break
            contiguous=speed
        result[f'{height:g}']=contiguous
    return result


def evaluate_velocity_grid(env, action, config, episodes, output_dir, global_step):
    from .landing_evaluation import evaluate_landing_suite,evaluation_heights
    heights=evaluation_heights(config)
    speeds=[float(v) for v in config['velocity_grid'].split(',')]
    threshold=float(config['success_boundary_threshold'])
    if (not speeds or speeds[0]!=0 or any(not np.isfinite(v) or v<0 for v in speeds)
            or any(b<=a for a,b in zip(speeds,speeds[1:])) or not 0<threshold<1):
        raise ValueError('Velocity grid must start at zero and increase; threshold must be in (0,1)')
    cells=[];rows=[];metrics={}
    directory=Path(output_dir)/'landing_evaluation';directory.mkdir(parents=True,exist_ok=True)
    for speed in speeds:
        tag=f'v_{speed:g}'.replace('.','p');condition='nominal' if speed==0 else 'perturbed'
        case=dict(config,velocity_grid='',max_angle_deg=0.,max_lateral_velocity=speed,
                  velocity_sampling='speed_ring',factorized_conditions=False,require_nominal_for_best=False,
                  _conditions=(condition,))
        nested=directory/'grid'/tag
        part=evaluate_landing_suite(env,action,case,episodes,nested,global_step)
        metrics.update({key.replace('eval/',f'eval/grid/{tag}/',1):value for key,value in part.items()})
        child=nested/'landing_evaluation'
        with (child/f'{global_step}_summary.csv').open() as f:
            for row in csv.DictReader(f):
                count=int(float(row['episode_count']));rate=float(row['success_rate']);wins=round(rate*count)
                cells.append(dict(height=float(row['height']),speed=speed,controller=row['controller'],
                    successes=wins,episodes=count,success_rate=rate,wilson95=wilson(wins,count),
                    passes=rate>threshold,confident_above_threshold=wilson(wins,count)[0]>threshold))
        with (child/f'{global_step}_episodes.csv').open() as f:
            rows.extend(dict(row,speed=speed) for row in csv.DictReader(f))
    policy=[c for c in cells if c['controller']=='policy'];nominal=[c for c in policy if c['speed']==0]
    velocity=[c for c in policy if c['speed']>0]
    selection=[c for c in velocity if c['speed']>=config.get('selection_speed_min',0.) and (config.get('selection_speed_max',-1.)<0 or c['speed']<=config['selection_speed_max'])]
    if not selection:raise ValueError('Empty checkpoint selection speed range')
    nominal_threshold=float(config['nominal_success_threshold'])
    if not 0<=nominal_threshold<=1:raise ValueError('Invalid nominal eligibility threshold')
    metrics.update({'eval/landing_success_rate':float(np.mean([c['success_rate'] for c in velocity])),
        'eval/episode_return':float(np.mean([float(r['episode_return']) for r in rows if r['controller']=='policy' and float(r['speed'])>0])),
        'eval/landing_episode_count':episodes*len(heights)*(len(speeds)-1),
        'eval/nominal_eligible':float(all((c['success_rate']>=1. if nominal_threshold==1. else c['success_rate']>nominal_threshold) for c in nominal)),
        'eval/nominal/policy/success_rate':float(np.mean([c['success_rate'] for c in nominal])),
        'eval/worst_height_success_rate':min(float(np.mean([c['success_rate'] for c in selection if c['height']==h])) for h in heights)})
    boundary=boundary_summary(cells,heights,speeds,threshold)
    for h,v in boundary.items():metrics[f'eval/boundary/h_{h.replace(".","p")}/max_speed_over_80']=v if v is not None else -1.
    history_path=directory/'boundary_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else dict(threshold=threshold,first_pass={},evaluations=[])
    for c in policy:
        key=f'h_{c["height"]:g}_v_{c["speed"]:g}'
        if c['passes']:history['first_pass'].setdefault(key,dict(step=global_step,**c))
    summary=dict(step=global_step,threshold=threshold,heights=heights,speeds=speeds,cells=cells,
        contiguous_speed_boundary=boundary,notes='Empirical rate strictly > threshold; 95% Wilson interval reported separately. First pass does not certify final checkpoint. Curriculum windows are local training outcomes, not this held-out validation.')
    history['evaluations']=[e for e in history['evaluations'] if e['step']!=global_step]+[summary]
    history_path.write_text(json.dumps(history,indent=2));(directory/f'{global_step}_boundaries.json').write_text(json.dumps(summary,indent=2))
    (directory/'config.json').write_text(json.dumps(dict(config,episodes_per_cell_per_controller=episodes),indent=2))
    with (directory/f'{global_step}_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    with (directory/f'{global_step}_episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return metrics
