import mujoco
import numpy as np

from simfiles import sim_state as S

OPEN_R, OPEN_L = 0.1, -0.4
# fallback values; calibrate() replaces the closed ones with the angle where the fingers just touch
CLOSE_R, CLOSE_L = -0.3, 0.3

def open():
    S.d.ctrl[4] = OPEN_R
    #S.d.ctrl[5] = OPEN_L

def close():
    S.d.ctrl[4] = CLOSE_R
    #S.d.ctrl[5] = CLOSE_L

def calibrate(margin=0.001, verbose=True):
    """Find how far the claw can close before Gear_R and Gear_L touch, and use that as 'closed'.
    Gear_R/Gear_L are excluded from collision in ARM.xml, so MuJoCo never stops them from crossing;
    here we measure the gap between their collision geoms (mj_geomDistance ignores the exclusion).
    Assumes the claw closes symmetrically: gear_R = t, gear_L = -t (matches open()/close() above).
    Call once after the model is loaded."""
    global CLOSE_R, CLOSE_L
    m = S.m
 
    def geoms(body_name):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] > 0]
 
    gR, gL = geoms("Gear_R"), geoms("Gear_L")
    qR = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gear_R")])
    qL = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gear_L")])
    if not gR or not gL:
        print("claw.calibrate: no collision geoms on the gears, keeping default closed angles")
        return
 
    tmp = mujoco.MjData(m)
    tmp.qpos[:] = S.d.qpos
    fromto = np.zeros(6)
 
    def gap(t):
        tmp.qpos[qR], tmp.qpos[qL] = t, -t
        mujoco.mj_kinematics(m, tmp)
        return min(mujoco.mj_geomDistance(m, tmp, a, b, 0.1, fromto) for a in gR for b in gL)
 
    ts = np.linspace(OPEN_R, CLOSE_R, 300)
    if gap(ts[0]) <= margin:
        print("claw.calibrate: the fingers already touch when open (hulls overlap), "
              "keeping default closed angles - set CLOSE_R / CLOSE_L by hand")
        return
    prev = ts[0]
    for t in ts[1:]:
        if gap(t) <= margin:
            CLOSE_R, CLOSE_L = float(prev), float(-prev)
            if verbose:
                print(f"claw.calibrate: closed at gear_R={CLOSE_R:.3f}, gear_L={CLOSE_L:.3f}")
            return
        prev = t
    if verbose:
        print("claw.calibrate: fingers never touch within the sweep, keeping default closed angles")