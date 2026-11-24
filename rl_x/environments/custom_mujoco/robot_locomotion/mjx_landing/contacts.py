"""Per-world landing contact quantities from the MJX-Warp contact buffer.

MJX-Warp keeps one contact buffer for all worlds, so these functions are batched
over worlds and must not be called inside jax.vmap. They reproduce
LandingContactMap: every collision pair of the robot contains the floor.
"""

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_safety import LandingContactMap


class ContactTables:
    """Geom classification shared with the CPU landing monitor."""

    def __init__(self, model, foot_geom_ids, body_geom_mask):
        contact_map = LandingContactMap(model)
        floor = contact_map.ground_geom_id
        if contact_map.support_geom_ids != {floor}:
            raise ValueError('MJX landing supports only the floor as a support surface')
        if not np.all((model.pair_geom1 == floor) | (model.pair_geom2 == floor)):
            raise ValueError('MJX landing requires robot-floor contact pairs only')
        if np.any(model.geom_contype) or np.any(model.geom_conaffinity):
            raise ValueError('MJX landing requires explicit contact pairs only')
        if np.any(model.pair_dim != 3) or model.opt.cone != 0:
            raise ValueError('MJX landing requires pyramidal condim 3 contacts')
        if np.any(model.pair_margin) or np.any(model.pair_gap):
            raise ValueError('MJX landing requires zero contact margin and gap')
        self.floor = int(floor)
        self.foot_slot = self._slots(model.ngeom, foot_geom_ids)
        self.calf_slot = self._slots(model.ngeom, sorted(contact_map.calf_geom_ids))
        self.support_slot = self._slots(model.ngeom, sorted(contact_map.allowed_support_geom_ids))
        self.allowed = jnp.asarray(contact_map.is_allowed_geom)
        self.body = jnp.asarray(body_geom_mask)
        self.geom_body = jnp.asarray(model.geom_bodyid)
        self.nr_feet, self.nr_calves = len(foot_geom_ids), len(contact_map.calf_geom_ids)
        self.nr_supports = len(contact_map.allowed_support_geom_ids)
        self.nr_foot_geoms = len(contact_map.foot_geom_ids)

    @staticmethod
    def _slots(ngeom, ids):
        slots = np.full(ngeom, -1)
        slots[list(ids)] = np.arange(len(ids))
        return jnp.asarray(slots)


def active_contacts(data, tables):
    """Flat buffer: (active mask, world id, robot geom, normal force)."""
    impl = data._impl
    count = impl.contact__worldid.shape[0]
    active = jnp.arange(count) < impl.nacon[0]
    active &= impl.contact__dist <= 0.
    geoms = impl.contact__geom
    geom = jnp.where(geoms[:, 0] == tables.floor, geoms[:, 1], geoms[:, 0])
    geom = jnp.where(active, geom, 0)
    world = jnp.where(active, impl.contact__worldid, 0)
    # Pyramidal condim 3: the normal force is the sum of the four edge forces.
    address = impl.contact__efc_address[:, :4]
    edge = impl.efc__force[world[:, None], jnp.maximum(address, 0)]
    force = jnp.abs(jnp.where(address >= 0, edge, 0.).sum(-1)) * active
    return active, world, geom, force


def contact_metrics(data, tables, nr_worlds):
    active, world, geom, force = active_contacts(data, tables)

    def per_slot(slots, size):
        slot = slots[geom]
        selected = active & (slot >= 0)
        index = world * size + jnp.maximum(slot, 0)
        return (jax.ops.segment_sum(jnp.where(selected, force, 0.), index, nr_worlds * size).reshape(nr_worlds, size),
                jax.ops.segment_max(selected.astype(jnp.int32), index, nr_worlds * size).reshape(nr_worlds, size) > 0)

    def any_contact(mask):
        return jax.ops.segment_max((active & mask).astype(jnp.int32), world, nr_worlds) > 0

    feet, feet_contact = per_slot(tables.foot_slot, tables.nr_feet)
    calves, _ = per_slot(tables.calf_slot, tables.nr_calves)
    return dict(feet_force=feet, calf_force=calves, feet_contact=feet_contact,
                floor=any_contact(jnp.ones_like(active)), body=any_contact(tables.body[geom]),
                forbidden=any_contact(~tables.allowed[geom]))


def contact_overflow(data):
    """True when a fixed contact or constraint buffer was too small."""
    impl = data._impl
    return jnp.any(impl.overflow != 0) | (impl.nacon[0] > impl.contact__worldid.shape[0])
