import mujoco
import numpy as np

from simfiles import sim_state as S

OPEN_R, OPEN_L = 0.55, -0.55
# fallback values; calibrate() replaces the closed ones with the angle where the fingers just touch
CLOSE_R, CLOSE_L = -0.5, 0.5


def open():
    S.d.ctrl[4] = OPEN_R
    S.d.ctrl[5] = OPEN_L


def close():
    S.d.ctrl[4] = CLOSE_R
    S.d.ctrl[5] = CLOSE_L