import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
import mujoco


class ADRAxis:
    """
    Jedna oś trudności (np. wysokość, kąt przechyłu, prędkość boczna).
    Ma własny przedział [low, high] i własne tempo wzrostu -
    niezależne od innych osi. To jest sedno ADR/SPDL, którego
    brakowało w wersji z globalnym `coeff`.
    """
    def __init__(self, name, init_low, init_high, floor, ceiling,
                 step, buffer_size=20, expand_thresh=0.75, shrink_thresh=0.3):
        self.name = name
        self.low = init_low
        self.high = init_high
        self.floor = floor        # absolutny dolny limit (np. 0)
        self.ceiling = ceiling    # absolutny górny limit (np. max wysokość spadku)
        self.step = step
        self.buf_low = deque(maxlen=buffer_size)
        self.buf_high = deque(maxlen=buffer_size)
        self.expand_thresh = expand_thresh
        self.shrink_thresh = shrink_thresh

    def sample(self, rng, pin_to="none"):
        """pin_to: 'low'/'high' (aktualna, WYUCZONA granica ADR),
        'floor'/'ceiling' (ABSOLUTNA granica z configu, niezależna od
        postępu treningu - do testów torture-test), albo 'none' (losowo)."""
        if pin_to == "low":
            return self.low
        if pin_to == "high":
            return self.high
        if pin_to == "floor":
            return self.floor
        if pin_to == "ceiling":
            return self.ceiling
        return rng.uniform(self.low, self.high)

    def report(self, pin_to, success):
        """Wywoływane na koniec epizodu, jeśli ta oś była testowana na granicy."""
        if pin_to == "low":
            self.buf_low.append(float(success))
            if len(self.buf_low) == self.buf_low.maxlen:
                rate = np.mean(self.buf_low)
                if rate > self.expand_thresh:
                    self.low = max(self.floor, self.low - self.step)
                elif rate < self.shrink_thresh:
                    self.low = min(self.high, self.low + self.step)
                self.buf_low.clear()
        elif pin_to == "high":
            self.buf_high.append(float(success))
            if len(self.buf_high) == self.buf_high.maxlen:
                rate = np.mean(self.buf_high)
                if rate > self.expand_thresh:
                    self.high = min(self.ceiling, self.high + self.step)
                elif rate < self.shrink_thresh:
                    self.high = max(self.low, self.high - self.step)
                self.buf_high.clear()


class EnergyDissipationCurriculumInitialState:
    """
    Zamiennik dla EnergyDissipationCurriculumInitialState.
    Różnice:
      - każda oś (wysokość, kąt, prędkość boczna) ma WŁASNE tempo,
        więc jeśli robot ogarnia wysokość szybciej niż rotację,
        te dwie osie rozjadą się naturalnie - nie trzeba tego ręcznie stroić.
      - w większości epizodów próbkujemy wszystkie osie niezależnie
        w ich bieżących przedziałach -> kombinacje (duża wysokość +
        duża rotacja + duża prędkość) powstają same, bez oddzielnych "trybów".
      - w ~p_boundary epizodów jedna losowa oś jest przypięta do swojej
        granicy (low albo high), żeby sprawdzić czy można ją poszerzyć.
      - yaw jest teraz też losowany (możesz to wyłączyć param. randomize_yaw).
    """

    def __init__(self, env, p_boundary=0.2, randomize_yaw=False):
        self.env = env
        self.p_boundary = p_boundary
        self.randomize_yaw = randomize_yaw

        # Test-only override: {"height": "ceiling", "angle": "ceiling", ...}
        # Wymusza konkretny pin per oś zamiast losowego p_boundary. Czytane
        # z configu, więc można sterować przez CLI bez zmiany kodu.
        self.force_pins = {}

        cfg = env.env_config.get("domain_randomization", {}).get("initial_state", {})
        adr_cfg = cfg.get("energy_dissipation_curriculum", {})
        self.is_energy_curriculum = cfg.get("type") == "energy_dissipation_curriculum"

        start_height = adr_cfg.get("start_height", 1.5)
        max_height = adr_cfg.get("max_height", 6.0)
        max_angle = np.deg2rad(adr_cfg.get("max_roll_angle_deg", 15))
        max_ang_ceiling = np.deg2rad(adr_cfg.get("max_angle_ceiling_deg", 180))
        max_vel = adr_cfg.get("max_lateral_velocity", 0.5)
        max_vel_ceiling = adr_cfg.get("max_lateral_velocity_ceiling", 3.0)

        # Ręczne wymuszenie pinów per oś, np. z CLI:
        #   ...force_pin_height=ceiling
        #   ...force_pin_angle=ceiling
        #   ...force_pin_lateral_vel=ceiling
        for axis_name in ("height", "angle", "lateral_vel"):
            forced = adr_cfg.get(f"force_pin_{axis_name}", None)
            if forced is not None:
                self.force_pins[axis_name] = forced

        self.axes = {
            "height": ADRAxis("height", start_height, start_height,
                               floor=0.3, ceiling=max_height, step=0.25),
            "angle": ADRAxis("angle", 0.0, max_angle,
                              floor=0.0, ceiling=max_ang_ceiling, step=np.deg2rad(5)),
            "lateral_vel": ADRAxis("lateral_vel", 0.0, max_vel,
                                    floor=0.0, ceiling=max_vel_ceiling, step=0.1),
        }
        self._last_pins = {}  # zapamiętane na potrzeby report() po epizodzie

    def setup(self):
        if not self.is_energy_curriculum:
            return None

        rng = self.env.np_rng

        if self.force_pins:
            # Tryb testowy: deterministyczne piny, żadnego losowania.
            # Osie bez wpisu w force_pins próbkują normalnie w [low, high].
            self._last_pins = {name: self.force_pins.get(name, "none")
                                for name in self.axes}
        else:
            pinned_axis = None
            if rng.random() < self.p_boundary:
                pinned_axis = rng.choice(list(self.axes.keys()))
            pin_side = rng.choice(["low", "high"]) if pinned_axis else None
            self._last_pins = {name: (pin_side if name == pinned_axis else "none")
                                for name in self.axes}

        height = self.axes["height"].sample(rng, self._last_pins["height"])
        angle_scale = self.axes["angle"].sample(rng, self._last_pins["angle"])
        vel_scale = self.axes["lateral_vel"].sample(rng, self._last_pins["lateral_vel"])

        roll_angle = rng.uniform(-angle_scale, angle_scale)
        pitch_angle = rng.uniform(-angle_scale, angle_scale)
        yaw_angle = rng.uniform(-np.pi, np.pi) if self.randomize_yaw else 0.0
        quaternion = Rotation.from_euler(
            "xyz", [roll_angle, pitch_angle, yaw_angle]
        ).as_quat(scalar_first=True)

        actuator_joint_positions = self.env.internal_state["actuator_joint_nominal_positions"].copy()
        actuator_joint_velocities = np.zeros(self.env.actuator_joint_max_velocities.size)

        linear_vel_x = rng.uniform(-vel_scale, vel_scale)
        linear_vel_y = rng.uniform(-vel_scale, vel_scale)
        linear_vel_z = 0.0
        angular_velocities = np.zeros(3)

        nominal_height = self.env.internal_state["robot_nominal_qpos_height_over_ground"]
        center_height = self.env.internal_state["center_height"]
        initial_z = nominal_height + center_height + height

        qpos = self.env.initial_qpos.copy()
        qpos[:3] = [0.0, 0.0, initial_z]
        qpos[3:7] = quaternion
        qpos[self.env.actuator_joint_mask_qpos] = actuator_joint_positions

        qvel = np.zeros(self.env.initial_mj_model.nv)
        qvel[:3] = [linear_vel_x, linear_vel_y, linear_vel_z]
        qvel[3:6] = angular_velocities
        qvel[self.env.actuator_joint_mask_qvel] = actuator_joint_velocities

        data = mujoco.MjData(self.env.internal_state["mj_model"])
        data.qpos = qpos
        data.qvel = qvel
        data.ctrl = np.zeros(self.env.nr_actuator_joints)
        mujoco.mj_forward(self.env.internal_state["mj_model"], data)

        feet_x_pos = data.geom_xpos[self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = np.max(
            self.env.terrain_function.ground_height_at(feet_x_pos, feet_y_pos)
            - data.geom_xpos[self.env.foot_geom_indices, 2]
        )
        if min_feet_z_pos_under_ground > 0:
            qpos[2] += min_feet_z_pos_under_ground

        info = self.env.internal_state["info"]
        info["curriculum/height"] = height
        info["curriculum/angle_scale"] = angle_scale
        info["curriculum/lateral_vel_scale"] = vel_scale
        axis_to_id = {"none": 0, "height": 1, "angle": 2, "lateral_vel": 3}
        pinned_name = next((name for name, pin in self._last_pins.items() if pin != "none"), "none")
        info["curriculum/pinned_axis_id"] = axis_to_id[pinned_name]
        for name, axis in self.axes.items():
            info[f"curriculum/{name}_low"] = axis.low
            info[f"curriculum/{name}_high"] = axis.high

        return qpos, qvel

    def report_episode_outcome(self, success: bool):
        """Wołaj to raz na koniec każdego epizodu, z Twoim istniejącym
        kryterium sukcesu (np. success_return threshold)."""
        for name, axis in self.axes.items():
            pin = self._last_pins.get(name, "none")
            if pin != "none":
                axis.report(pin, success)