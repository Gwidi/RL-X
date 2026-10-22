"""Shared physical landing limits and physics-rate measurements for RL/search."""

from collections import deque

import mujoco
import numpy as np

CALF_LEGS = ("RL", "RR", "FR", "FL")
MAX_LANDING_FOOT_FORCE = 3000.0
MAX_LANDING_CALF_FORCE = 1000.0
MAX_LANDING_BODY_ACCELERATION = 98.1
MIN_LANDING_BASE_HEIGHT = 0.05
FOOT_FORCE_WINDOW = CALF_FORCE_WINDOW = 0.010
BODY_ACCELERATION_WINDOW = 0.050


def landing_violations(foot_contact, disallowed_contact, min_height,
                       peak_foot_force, peak_calf_force, peak_body_acceleration,
                       xp=np):
    """Inclusive limits, usable with NumPy and JAX; missing support is final-only."""
    return {
        "missing_foot_contact": ~xp.asarray(foot_contact, dtype=bool),
        "disallowed_contact": xp.asarray(disallowed_contact, dtype=bool),
        "base_height": (min_height < MIN_LANDING_BASE_HEIGHT) | ~xp.isfinite(min_height),
        "foot_force": (peak_foot_force > MAX_LANDING_FOOT_FORCE) | ~xp.isfinite(peak_foot_force),
        "calf_force": (peak_calf_force > MAX_LANDING_CALF_FORCE) | ~xp.isfinite(peak_calf_force),
        "body_acceleration": ((peak_body_acceleration > MAX_LANDING_BODY_ACCELERATION)
                              | ~xp.isfinite(peak_body_acceleration)),
    }


class LandingContactMap:
    """Allow feet and the distal calf third on support; reject proximal calves.

    Historical calf/shin force metrics and penalties refer to the upper part.
    Lower calf contact stays physical, but is exempt from these penalties and
    never counts as foot support. Models without a split retain strict calves.
    """

    def __init__(self, model):
        self.ground_geom_id = model.geom("floor").id
        self.support_geom_ids = {self.ground_geom_id}
        box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_landing_box")
        if box >= 0:
            self.support_geom_ids.add(box)
        self.foot_geom_ids = {model.geom(f"{leg}_foot").id for leg in CALF_LEGS}
        self.calf_geom_ids = {model.geom(f"{leg}_calf").id for leg in CALF_LEGS}
        self.lower_calf_geom_ids = set()
        for leg in CALF_LEGS:
            lower = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_calf_lower")
            if lower >= 0:
                self.lower_calf_geom_ids.add(lower)
        self.allowed_support_geom_ids = self.foot_geom_ids | self.lower_calf_geom_ids
        calf_bodies = {model.geom_bodyid[g] for g in self.calf_geom_ids}
        self.shin_geom_ids = self.calf_geom_ids | {g for g in range(model.ngeom)
                              if model.geom_bodyid[g] in calf_bodies
                              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER}
        self.calf_geom_to_index = {model.geom(f"{leg}_calf").id: i
                                   for i, leg in enumerate(CALF_LEGS)}
        self.foot_geom_to_leg = {model.geom(f"{leg}_foot").id: leg for leg in CALF_LEGS}
        self.foot_legs = sorted(CALF_LEGS)

    def support_contact_forces(self, model, data, include_calf_forces=False):
        feet = dict.fromkeys(self.foot_legs, 0.0)
        calves = np.zeros(len(CALF_LEGS))
        shin = 0.0
        forbidden = False
        force = np.zeros(6)
        allowed = self.allowed_support_geom_ids
        for i, contact in enumerate(data.contact):
            pair = {int(contact.geom1), int(contact.geom2)}
            robot = {g for g in pair if model.geom_bodyid[g] != 0}
            if robot and not (pair & self.support_geom_ids and robot <= allowed):
                forbidden = True
            if not pair & self.support_geom_ids:
                continue
            mujoco.mj_contactForce(model, data, i, force)
            normal = abs(float(force[0]))
            for g in pair & self.foot_geom_ids:
                feet[self.foot_geom_to_leg[g]] += normal
            if pair & self.shin_geom_ids:
                shin += normal
            for g in pair & self.calf_geom_ids:
                calves[self.calf_geom_to_index[g]] += normal
        by_foot = np.array([feet[leg] for leg in self.foot_legs])
        result = (float(by_foot.sum()), float(by_foot.max()), by_foot, shin, forbidden)
        return result + (calves,) if include_calf_forces else result


class LandingSafetyMonitor:
    """Reset per episode; update immediately after every mj_step, before DR.

    Force windows include zeros before impact. Acceleration starts at the first
    ground contact, also zero padded; subsequent samples include gaps in contact.
    """

    def __init__(self, model):
        self.contacts = LandingContactMap(model)
        self.dt = float(model.opt.timestep)
        self.force_steps = max(1, int(np.ceil(FOOT_FORCE_WINDOW / self.dt)))
        self.acceleration_steps = max(1, int(np.ceil(BODY_ACCELERATION_WINDOW / self.dt)))
        self.force_history = deque(maxlen=self.force_steps)
        self.acceleration_history = deque(maxlen=self.acceleration_steps)
        self.peak_foot = np.zeros(4)
        self.peak_calf = np.zeros(4)
        self.peak_foot_raw = np.zeros(4)
        self.peak_calf_raw = np.zeros(4)
        self.foot_contact = self.disallowed_contact = self.touched = False
        self.samples = 0
        self.min_height = np.inf
        self.peak_acceleration = self.peak_acceleration_raw = self.acceleration = 0.0

    def update(self, model, data):
        forces_by_contact = self.contacts.support_contact_forces(
            model, data, include_calf_forces=True)
        total, _, feet, _, forbidden, calves = forces_by_contact
        self.samples += 1
        self.foot_contact |= total > 0
        self.disallowed_contact |= forbidden
        self.min_height = float(np.minimum(self.min_height, data.qpos[2]))
        forces = np.stack((feet, calves))
        self.force_history.append(forces)
        average = np.sum(self.force_history, axis=0) / self.force_steps
        self.peak_foot = np.maximum(self.peak_foot, average[0])
        self.peak_calf = np.maximum(self.peak_calf, average[1])
        self.peak_foot_raw = np.maximum(self.peak_foot_raw, feet)
        self.peak_calf_raw = np.maximum(self.peak_calf_raw, calves)
        self.touched |= any(bool({int(c.geom1), int(c.geom2)} & self.contacts.support_geom_ids)
                            for c in data.contact)
        if self.touched:
            acceleration = float(np.linalg.norm(data.qacc[:3]))
            self.acceleration_history.append(acceleration)
            self.acceleration = float(np.sum(self.acceleration_history) / self.acceleration_steps)
            self.peak_acceleration = float(np.maximum(self.peak_acceleration, self.acceleration))
            self.peak_acceleration_raw = float(np.maximum(self.peak_acceleration_raw, acceleration))

        return forces_by_contact

    def violations(self):
        return landing_violations(self.foot_contact, self.disallowed_contact, self.min_height if self.samples else 1.0,
                                  float(self.peak_foot.max()), float(self.peak_calf.max()),
                                  self.peak_acceleration)

    def metrics(self):
        return dict(peak_foot_force_10ms=float(self.peak_foot.max()),
                    peak_calf_force_10ms=float(self.peak_calf.max()),
                    safety_peak_body_acceleration=self.peak_acceleration,
                    safety_disallowed_contact=float(self.disallowed_contact),
                    safety_min_base_height=self.min_height)
