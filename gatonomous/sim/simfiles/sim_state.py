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

# target positions
CURRENTTARGET = np.array([0.2, 0.2, 0.2])

AUTO = False                  # ENTER toggles the automatic sequence
MANUAL_TASK = None            # generator for a manually triggered motion sequence
SLOT_INDEX = 0                # next slot to fill
SKIPPED = []                  # boxes the arm couldn't reach
BASE_XY = np.array([0.0, 0.0])   # world xy of the yaw axis; change if the base isn't at the origin
BOXES = [                     # sim stand-in for the camera; EDIT to positions your arm can reach
    dict(color="orange", pos=[-0.2, -0.12, 0.02]),
    dict(color="orange", pos=[-0.3, 0.05, 0.02]),
    dict(color="orange", pos=[-0.23, 0.17, 0.02]),
    dict(color="green",  pos=[-0.28, -0.06, 0.02]),
    dict(color="green",  pos=[-0.15, -0.23, 0.02]),
    dict(color="beige",  pos=[-0.15, 0.25, 0.02]),
]
# make randomization (?) hrs spawn xml boxesnya
SLOTS = [  # build order
    dict(color="orange", pos=[-0.0525, 0.3, 0.02]),
    dict(color="orange", pos=[-0.0,  0.3, 0.02]),
    dict(color="orange", pos=[0.0525,  0.3, 0.02]),
    dict(color="green",  pos=[-0.032, 0.3, 0.105]),
    dict(color="green",  pos=[0.028,  0.3, 0.105]),
    dict(color="beige",  pos=[0.0,   0.29, 0.15]),
]

class BotState(Enum):
    START = "Starting"
    SCANNING = "Scanning"
    GRABBING = "Grabbing" # Grab sequence
    PLACING = "Placing" # Place sequence
    FINISHED = "Finished"
    FAILED = "Failed"
    INVERSE_K = "IK"
    TELEOP = "Teleop"
    POSED = "Posed"


def set_bot_state(st: BotState, name: str):
    global STATE, STATENAME
    STATE = st
    STATENAME = name

set_bot_state(BotState.START, BotState.START.name)