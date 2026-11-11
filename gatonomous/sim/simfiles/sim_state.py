# shared sim state (like simulation.hpp)
# other modules do `import sim_state as S` and use S.m, S.d, S.STATE, ... so they always see current values
from enum import Enum

import numpy as np

# mujoco and sim objects, set in simulation.main()
m = None    # mjModel
d = None    # mjData
scn = None  # MjvScene
cam = None  # MjvCamera
opt = None  # MjvOption
con = None  # MjrContext

# IK target
target_pos = np.array([0.2, 0.2, 0.2])


class BotState(Enum):
    TELEOP = "Teleop"
    POSED = "Posed"
    ANIMATED = "Animated"
    INVERSE_K = "InverseK"


STATE = BotState.TELEOP
STATENAME = "-"


def set_bot_state(st: BotState, name: str):
    global STATE, STATENAME
    STATE = st
    STATENAME = name