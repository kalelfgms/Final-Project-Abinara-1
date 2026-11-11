# functions for controlling the arm
from typing import List

import numpy as np

from simfiles import sim_state as S

def move_joint(joint: int, direction: float):
    """direction: left -1, right 1"""
    step = direction * 1
    vmax = 12  # max speed so one key press doesn't jump to the limit
    ctrl_min, ctrl_max = S.m.actuator_ctrlrange[joint]

    S.set_bot_state(S.BotState.TELEOP, "-")
    S.d.ctrl[joint] = np.clip(S.d.ctrl[joint] + step * vmax * S.m.opt.timestep,
                              ctrl_min, ctrl_max)


# POSE
class Pose:
    def __init__(self, name: str, key: int, orients):
        self.name = name
        self.key = key  # glfw key that plays this pose
        self.orients = list(orients)  # 5 values, one per actuator

    def get_pose(self):
        return self.orients

    def get_key(self):
        return self.key

    def get_name(self):
        return self.name


pose_list: List[Pose] = []


def play_pose(pose: Pose):
    S.set_bot_state(S.BotState.POSED, pose.get_name())
    set_pose(pose)


def set_pose(pose: Pose):
    for i, v in enumerate(pose.get_pose()):
        S.d.ctrl[i] = v