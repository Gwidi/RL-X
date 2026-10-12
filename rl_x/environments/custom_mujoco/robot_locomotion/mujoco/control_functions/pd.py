import numpy as np


class PDControl:
    def __init__(self, env, control_frequency_hz=50):
        self.env = env
        self.control_frequency_hz = control_frequency_hz


    def process_action(self, action):
        scaled_action = action * self.env.internal_state["scaling_factor"]
        target_joint_positions = self.env.internal_state["actuator_joint_nominal_positions"] + scaled_action
        noisy_target_joint_positions = target_joint_positions + self.env.internal_state["position_offsets"]
        model = self.env.internal_state["mj_model"]
        joint_limits = model.jnt_range[self.env.actuator_joint_mask_joints]
        limited = model.jnt_limited[self.env.actuator_joint_mask_joints].astype(bool)
        noisy_target_joint_positions = np.where(
            limited,
            np.clip(noisy_target_joint_positions, joint_limits[:, 0], joint_limits[:, 1]),
            noisy_target_joint_positions,
        )
        
        # MuJoCo position actuators use actuator length (gear * joint angle)
        # as their control target. Policy actions are normalized residuals.
        return noisy_target_joint_positions * model.actuator_gear[:, 0]
