"""Loaded support slip at actual contact points, sampled at control rate."""
import mujoco
import numpy as np


class LandingSlip:
    def __init__(self, env, cfg):
        self.env = env
        self.weight = float(cfg.get('weight', 0.))
        self.tau = float(cfg.get('force_filter_s', .04))
        self.threshold = float(cfg.get('load_threshold_n', 10.))
        self.full_load = float(cfg.get('full_load_n', 50.))
        self.delay = float(cfg.get('impact_grace_s', .1))
        self.ramp = float(cfg.get('ramp_s', .2))
        values = [self.weight, self.tau, self.threshold, self.full_load, self.delay, self.ramp]
        if not all(np.isfinite(v) and v >= 0 for v in values) or self.full_load <= self.threshold:
            raise ValueError('Invalid loaded-slip configuration')

    def step(self, elapsed):
        e = self.env
        m, d = e.internal_state['mj_model'], e.internal_state['data']
        # Distal calf support is allowed physically: include it to avoid moving
        # the load from the foot to the calf to evade this cost.
        cmap = e.internal_state['landing_safety'].contacts
        supports = cmap.foot_geom_ids | cmap.lower_calf_geom_ids
        ids = sorted(supports)
        state = e.internal_state.setdefault('landing_slip', {})
        filtered = state.setdefault('filtered_load', np.zeros(len(ids)))
        forces, weighted_speed = np.zeros(len(ids)), np.zeros(len(ids))
        index = {g: i for i, g in enumerate(ids)}
        force = np.zeros(6)
        jac_a, jac_b = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for ci, c in enumerate(d.contact):
            a, b = int(c.geom1), int(c.geom2)
            if a in supports and b in cmap.support_geom_ids:
                support = a
            elif b in supports and a in cmap.support_geom_ids:
                support = b
            else:
                continue
            mujoco.mj_contactForce(m, d, ci, force)
            load = abs(float(force[0]))
            if load <= 0:
                continue
            mujoco.mj_jac(m, d, jac_a, None, c.pos, int(m.geom_bodyid[a]))
            mujoco.mj_jac(m, d, jac_b, None, c.pos, int(m.geom_bodyid[b]))
            relative = (jac_a - jac_b) @ d.qvel
            normal = np.asarray(c.frame).reshape(3, 3)[0]
            tangent = relative - normal * np.dot(relative, normal)
            i = index[support]
            forces[i] += load
            weighted_speed[i] += load * float(tangent @ tangent)
        alpha = 1. if self.tau == 0 else -np.expm1(-e.dt / self.tau)
        filtered += alpha * (forces - filtered)
        speed2 = np.divide(weighted_speed, forces, out=np.zeros_like(forces), where=forces > 0)
        # No remembered-contact penalty in flight, even while the filter decays.
        load_weights = np.clip((filtered - self.threshold) / (self.full_load - self.threshold), 0, 1)
        load_weights *= forces > 0
        # Fixed denominator: adding support does not dilute another foot's slip.
        slip = float(np.sum(load_weights * speed2) / len(cmap.foot_geom_ids))
        ramp = float(elapsed >= self.delay) if self.ramp == 0 else float(np.clip((elapsed-self.delay)/self.ramp, 0, 1))
        cost = self.weight * e.dt * ramp * slip
        state['integral'] = state.get('integral', 0.) + e.dt * slip
        e.internal_state['info'].update({'reward/loaded_slip': -cost,
            'metrics/loaded_slip_speed2': slip, 'metrics/loaded_slip_integral': state['integral'],
            'metrics/loaded_slip_ramp': ramp})
        return cost
