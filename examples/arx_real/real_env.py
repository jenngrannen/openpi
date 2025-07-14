# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
import numpy as np
import threading

from examples.arx_real import constants
from examples.arx_real import robot_utils

# OFFLINE flag for testing without hardware
OFFLINE = False

if not OFFLINE:
    import pyrealsense2 as rs
    from arx5_interface import Arx5CartesianController, Arx5JointController, Gain, LogLevel
else:
    from offline_mocks import MockController, MockJointState, MockEefState, MockCamera, MockRs
    rs = MockRs

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, *, reset_position: Optional[List[float]] = None):
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION

        self.puppet_bot_left = follower = Arx5JointController(
                "L5",
                "can3",
            )
        self.puppet_bot_left.enable_background_send_recv()
        self.puppet_bot_right = Arx5JointController(
                "L5",
                "can1",
            )
        self.puppet_bot_right.enable_background_send_recv()

        self.queue_event = threading.Event()

        self.recorder_left = robot_utils.Recorder("left", self.puppet_bot_left, self.queue_event)
        self.recorder_right = robot_utils.Recorder("right", self.puppet_bot_right, self.queue_event)
        self.image_recorder = robot_utils.ImageRecorder(self.queue_event)

    def get_qpos(self):
        left_qpos_raw = self.recorder_left.get_qpos()
        right_qpos_raw = self.recorder_right.get_qpos()
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
        ]  
        right_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
        ]  
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        joint_poses = self.get_qpos()
        left_joint_poses = joint_poses[:6] + [left_gripper_desired_joint]
        right_joint_poses = joint_poses[6:12] + [right_gripper_desired_joint]
        
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [left_joint_poses, right_joint_poses]
        )

    def _reset_joints(self):
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [self._reset_position, self._reset_position]
        )

    def _reset_gripper(self):
        """Set to position mode and do position resets: first close then open. Then change back to PWM mode

        NOTE: This diverges from the original Aloha code which first opens then closes the gripper. Pi internal aloha data
        was collected with the gripper starting in the open position. Leaving the grippers fully closed was also found to
        increase the frequency of motor faults.
        """
        joint_poses = self.get_qpos()

        desired_gripper_pose = constants.PUPPET_GRIPPER_JOINT_CLOSE
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [joint_poses[:6] + [desired_gripper_pose], joint_poses[6:12] + [desired_gripper_pose]]
        )

        desired_gripper_pose = constants.PUPPET_GRIPPER_JOINT_OPEN
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [joint_poses[:6] + [desired_gripper_pose], joint_poses[6:12] + [desired_gripper_pose]]
        )
     
    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        # obs["qvel"] = self.get_qvel()
        # obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        if not fake:
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [left_action, right_action]
        )
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

# No support for master-puppet yet
# def get_action(master_bot_left, master_bot_right):
#     action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
#     # Arm actions
#     action[:6] = master_bot_left.dxl.joint_states.position[:6]
#     action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
#     # Gripper actions
#     action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
#     action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

#     return action


def make_real_env(*, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(reset_position=reset_position)
