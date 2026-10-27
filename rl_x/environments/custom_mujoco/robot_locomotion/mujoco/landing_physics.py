"""Explicit, shared integration settings for landing PPO and CPU/MJX search."""
import hashlib
import json
from pathlib import Path
import mujoco

PROJECT_ROOT=Path(__file__).resolve().parents[5]


def read_landing_physics(path):
    path=Path(path)
    if not path.is_absolute():path=PROJECT_ROOT/path
    blob=path.read_bytes();profile=json.loads(blob)
    if profile['integrator'] not in ('Euler','implicitfast','implicit','RK4'):
        raise ValueError('Unsupported landing integrator')
    dt=float(profile['timestep'])
    if dt<=0 or abs(.02/dt-round(.02/dt))>1e-9:
        raise ValueError('Landing physics timestep must divide the 20 ms control interval')
    return profile,hashlib.sha256(blob).hexdigest()


def apply_landing_physics(model,path,expected_timestep=None,expected_sha256=None):
    """Apply only integration settings; gains, home and contact remain unchanged."""
    profile,digest=read_landing_physics(path)
    if expected_sha256 and digest!=expected_sha256:
        raise ValueError('Landing physics profile SHA256 mismatch')
    if expected_timestep is not None and abs(float(expected_timestep)-profile['timestep'])>1e-12:
        raise ValueError('environment.timestep differs from the landing physics profile')
    model.opt.timestep=float(profile['timestep'])
    model.opt.integrator=getattr(mujoco.mjtIntegrator,'mjINT_'+profile['integrator'].upper())
    model.opt.iterations=int(profile['iterations']);model.opt.ls_iterations=int(profile['ls_iterations'])
    mask=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    model.opt.disableflags=int(model.opt.disableflags)&~mask
    return digest
