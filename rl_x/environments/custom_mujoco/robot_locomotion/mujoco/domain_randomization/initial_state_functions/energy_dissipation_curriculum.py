import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

class EnergyDissipationCurriculumInitialState:
    def __init__(self, env):
        self.env = env

        domain_rand_config = env.env_config.get("domain_randomization", {})
        initial_state_config = domain_rand_config.get("initial_state", {})
        curriculum_config = initial_state_config.get("energy_dissipation_curriculum", {})

        self.start_height = curriculum_config.get("start_height", 1.5)
        self.target_height = curriculum_config.get("target_height", 0.5)

        self.max_roll_angle = np.deg2rad(curriculum_config.get("max_roll_angle_deg", 15))
        self.max_pitch_angle = np.deg2rad(curriculum_config.get("max_pitch_angle_deg", 15))
        self.max_lateral_velocity = curriculum_config.get("max_lateral_velocity", 0.5)

        self.is_energy_curriculum = initial_state_config.get("type") == "energy_dissipation_curriculum"

    def _get_curriculum_schedule(self, coeff):
        mode = self.env.np_rng.choice(["extreme_height", "extreme_rotation", "extreme_velocity", "mixed_combat"])
        
        if mode == "extreme_height":
            height = self.start_height + (4.0 * coeff)
            angle_multiplier = 0.0
            velocity_multiplier = 0.0
            
        elif mode == "extreme_rotation":
            height = self.start_height + (2.0 * coeff)
            angle_multiplier = coeff * 2.0 
            velocity_multiplier = 0.0
            
        elif mode == "extreme_velocity":
            height = self.start_height + (2.0 * coeff)
            angle_multiplier = 0.0
            velocity_multiplier = coeff * 3.0 
            
        else: # "mixed_combat"
            height = self.start_height + ((self.target_height - self.start_height) * coeff)
            angle_multiplier = coeff * 1.5
            velocity_multiplier = coeff * 1.5

        return height, angle_multiplier, velocity_multiplier

    def setup(self):
        if not self.is_energy_curriculum:
            return None

        curriculum_coeff = self.env.internal_state["env_curriculum_coeff"]
        height, angle_mult, vel_mult = self._get_curriculum_schedule(curriculum_coeff)

        # 1. NAPRAWA ORIENTACJI: Roll i Pitch lekko losowe, ale YAW ZAWSZE ZERO (patrzy prosto)
        roll_angle = self.env.np_rng.uniform(low=-self.max_roll_angle * angle_mult, high=self.max_roll_angle * angle_mult)
        pitch_angle = self.env.np_rng.uniform(low=-self.max_pitch_angle * angle_mult, high=self.max_pitch_angle * angle_mult)
        yaw_angle = 0.0  # <--- TUTAJ BYŁ BŁĄD. Teraz robot zawsze jest obrócony przodem!
        
        quaternion = Rotation.from_euler("xyz", [roll_angle, pitch_angle, yaw_angle]).as_quat(scalar_first=True)

        # 2. NAPRAWA NÓG: Zawsze rodzą się w idealnej, nominalnej pozycji (zero szarpania w locie)
        actuator_joint_positions = self.env.internal_state["actuator_joint_nominal_positions"].copy()
        
        # Prędkości stawów też na sztywno zero
        actuator_joint_velocities = np.zeros(self.env.actuator_joint_max_velocities.size)

        # 3. NAPRAWA RZUTU: Rzuca go idealnie w bok
        # Zwiększamy mnożnik na 3.0, żeby było widoczne pchnięcie
        lateral_strength = self.max_lateral_velocity * 3.0 
        linear_vel_x = self.env.np_rng.uniform(low=-lateral_strength * vel_mult, high=lateral_strength * vel_mult)
        linear_vel_y = self.env.np_rng.uniform(low=-lateral_strength * vel_mult, high=lateral_strength * vel_mult)
        linear_vel_z = 0.0 

        # 4. NAPRAWA SPINU: Zero obrotów na start
        angular_velocities = np.zeros(3) 

        # Pozycja Z (spadek z odpowiedniej wysokości)
        nominal_height = self.env.internal_state["robot_nominal_qpos_height_over_ground"]
        center_height = self.env.internal_state["center_height"]
        initial_z = nominal_height + center_height + height

        linear_positions = np.array([0.0, 0.0, initial_z])

        # Składanie stanu symulacji (QPOS i QVEL)
        qpos = self.env.initial_qpos.copy()
        qpos[:3] = linear_positions
        qpos[3:7] = quaternion
        qpos[self.env.actuator_joint_mask_qpos] = actuator_joint_positions

        qvel = np.zeros(self.env.initial_mj_model.nv)
        qvel[:3] = [linear_vel_x, linear_vel_y, linear_vel_z]
        qvel[3:6] = angular_velocities
        qvel[self.env.actuator_joint_mask_qvel] = actuator_joint_velocities

        # Zasilenie MuJoCo
        data = mujoco.MjData(self.env.internal_state["mj_model"])
        data.qpos = qpos
        data.qvel = qvel
        data.ctrl = np.zeros(self.env.nr_actuator_joints)
        mujoco.mj_forward(self.env.internal_state["mj_model"], data)

        # Poprawka na wejście pod podłogę
        feet_x_pos = data.geom_xpos[self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = np.max(
            self.env.terrain_function.ground_height_at(feet_x_pos, feet_y_pos) -
            data.geom_xpos[self.env.foot_geom_indices, 2]
        )
        if min_feet_z_pos_under_ground > 0:
            qpos[2] += min_feet_z_pos_under_ground

        # Logi
        self.env.internal_state["info"]["curriculum/initial_height"] = height
        self.env.internal_state["info"]["curriculum/angle_multiplier"] = angle_mult
        self.env.internal_state["info"]["curriculum/velocity_multiplier"] = vel_mult
        self.env.internal_state["info"]["curriculum/coeff"] = curriculum_coeff

        return qpos, qvel