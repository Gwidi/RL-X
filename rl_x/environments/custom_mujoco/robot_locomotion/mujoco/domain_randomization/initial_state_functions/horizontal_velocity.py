"""Shared initial horizontal velocity distributions for training/evaluation."""
import numpy as np

MODES=('component_box','speed_disk')

def validate_velocity_sampling(mode):
    if mode not in MODES:
        raise ValueError('velocity_sampling must be component_box or speed_disk')
    return mode

def sample_horizontal_velocity(rng, limit, mode='component_box'):
    validate_velocity_sampling(mode)
    if mode=='component_box':
        # Preserve historical two scalar draws and ordering exactly.
        return rng.uniform(-limit,limit),rng.uniform(-limit,limit)
    # Uniform area in a disk: P(speed <= r) = (r/limit)^2.
    speed=limit*np.sqrt(rng.uniform(0.,1.))
    azimuth=rng.uniform(0.,2*np.pi)
    return speed*np.cos(azimuth),speed*np.sin(azimuth)
