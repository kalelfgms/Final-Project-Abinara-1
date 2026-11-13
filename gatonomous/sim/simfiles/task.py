import mujoco
import numpy as np

from simfiles import sim_state as S
from simfiles import claw
from simfiles.ik import solve_ik
from simfiles.controls import play_pose, pose_list

# actuator order in ARM.xml: 0 joint1(yaw) 1 joint2 2 joint3 3 joint5(wrist pitch) 4 gear_R 5 gear_L
WRIST = 3
ARM_ACTS = [0, 1, 2, 3]

APPROACH = 0.06     # how far behind the target the claw stops before going in (m)
LIFT = 0.04         # how high to lift a box before carrying it (m)
SPEED = 0.08        # tip speed along a path (m/s of SIM time)
TOL = 0.008         # "arrived" tolerance (m)
MIN_TIP_Z = 0.03    # the claw tip is never commanded lower than this (m); raise it if the fingers still scrape

PITCH_CANDIDATES = [0.0, -0.15, -0.3, -0.45, -0.6]   # claw elevation (rad); first one the joint limits allow is used
_pitch = 0.0        # claw elevation currently commanded
_cmd = None         # the Cartesian point we are currently *commanding* (not where the tip is)


# ---------------------------------------------------------------- timing (all in SIM time)
def wait(seconds):
    t0 = S.d.time
    while S.d.time - t0 < seconds:
        yield


def _tip():
    sid = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_SITE, "endeff")
    return S.d.site_xpos[sid].copy()


def _arm_speed():
    dofs = [int(S.m.jnt_dofadr[int(S.m.actuator_trnid[i, 0])]) for i in ARM_ACTS]
    return float(np.linalg.norm(S.d.qvel[dofs]))   # ignores the boxes' free joints


def settle(timeout=3.0):
    """Wait until the real tip has caught up with the commanded point and the arm stopped."""
    t0 = S.d.time
    while S.d.time - t0 < timeout:
        if np.linalg.norm(_tip() - _cmd) < TOL and _arm_speed() < 0.05:
            return
        yield


# ---------------------------------------------------------------- geometry
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _polar(p):
    v = np.asarray(p[:2], float) - S.BASE_XY
    return np.array([np.hypot(*v), np.arctan2(v[1], v[0]), p[2]])


def _cart(r, th, z):
    return np.array([S.BASE_XY[0] + r * np.cos(th), S.BASE_XY[1] + r * np.sin(th), z])


def _floor(p):
    """Keep a target at least MIN_TIP_Z above the ground (the claw body hangs below the tip site)."""
    p = np.array(p, dtype=float)
    p[2] = max(p[2], MIN_TIP_Z)
    return p


def _behind(p, dist=APPROACH):
    """Point `dist` short of p on the line from the base axis through p, same height as p."""
    p = np.array(p, dtype=float)
    v = p[:2] - S.BASE_XY
    n = np.linalg.norm(v)
    if n > 1e-8:
        p[:2] -= v / n * dist
    return p


# ---------------------------------------------------------------- IK helpers
def _ik_step(p, pitch=None):
    """One small IK step seeded from the previous command (keeps the arm on one configuration)."""
    seed = S.d.ctrl.copy()
    ok = solve_ik(S.m, S.d, "endeff", p, pitch=_pitch if pitch is None else pitch, seed_ctrl=seed,
                  multi_start=False, max_iter=60, verbose=False)
    if not ok:
        S.d.ctrl[:] = seed
    return ok


def move_to(target, speed=SPEED, pitch=None):
    """Move the tip to `target` along a smooth path, one small IK step per frame.
    The path is interpolated in (radius, yaw, height) around the base axis:
      - radial moves (approach / retreat) are perfectly straight lines, with yaw frozen
      - travel between two different directions is an arc, which stays inside the workspace
    Returns True if it got there."""
    global _cmd, _pitch
    pitch0 = _pitch
    pitch1 = pitch0 if pitch is None else pitch      # claw elevation is blended along the path
    target = _floor(target)
    a, b = _polar(_cmd), _polar(target)
    dth = _wrap(b[1] - a[1])
    arc = abs(dth) * 0.5 * (a[0] + b[0])
    length = float(np.sqrt((b[0] - a[0]) ** 2 + arc ** 2 + (b[2] - a[2]) ** 2))
    T = max(length / speed, 0.1)

    t0 = S.d.time
    while True:
        s = min((S.d.time - t0) / T, 1.0)
        p = _cart(a[0] + s * (b[0] - a[0]), a[1] + s * dth, a[2] + s * (b[2] - a[2]))
        pitch_s = pitch0 + s * (pitch1 - pitch0)
        if not _ik_step(p, pitch_s):
            print(f"Unreachable on path near {np.round(p, 3)}")
            return False
        _cmd, _pitch = p, pitch_s
        if s >= 1.0:
            break
        yield
    yield from settle()
    return True


def pick_pitch(points):
    """First claw elevation in PITCH_CANDIDATES for which every point is reachable within the joint
    limits (checked with the full multi-start IK). Returns None if none works. Leaves d.ctrl untouched."""
    old = S.d.ctrl.copy()
    chosen = None
    for pt in PITCH_CANDIDATES:
        if all(solve_ik(S.m, S.d, "endeff", p, pitch=pt, seed_ctrl=old, verbose=False) for p in points):
            chosen = pt
            break
    S.d.ctrl[:] = old
    return chosen


def _ramp(goal, idxs, duration=0.7):
    """Smoothly move the listed actuators from their current ctrl to goal[idxs]."""
    start = S.d.ctrl.copy()
    t0 = S.d.time
    while True:
        s = min((S.d.time - t0) / duration, 1.0)
        s = s * s * (3 - 2 * s)          # smoothstep
        for i in idxs:
            S.d.ctrl[i] = start[i] + (goal[i] - start[i]) * s
        if s >= 1.0:
            return
        yield


def deploy_to(p, pitch=0.0):
    """Leave the scan pose: solve IK for p once, then unfold the wrist FIRST (so it is no longer
    curled in when the arm starts swinging down), then move shoulder/elbow/yaw."""
    global _cmd, _pitch
    old = S.d.ctrl.copy()
    if not solve_ik(S.m, S.d, "endeff", p, pitch=pitch):     # full multi-start solve
        S.d.ctrl[:] = old
        print("Unreachable")
        return False
    goal = S.d.ctrl.copy()
    S.d.ctrl[:] = old
    yield from _ramp(goal, [WRIST])
    yield from _ramp(goal, [0, 1, 2])
    _cmd, _pitch = np.asarray(p, dtype=float), pitch
    yield from settle()
    return True


# ---------------------------------------------------------------- old instant helpers (kept in case controls.py uses them)
def behind_target(targetpos):
    return _ik_step(_behind(targetpos))


def to_target(targetpos):
    return _ik_step(np.asarray(targetpos, float))


# ---------------------------------------------------------------- pick & place
def grab(box):
    box = _floor(box)
    pitch = pick_pitch([_behind(box), box])
    if pitch is None:
        print("Box unreachable within the joint limits")
        return False
    ok = yield from deploy_to(_behind(box), pitch)        # unfold wrist, go to the point behind the box
    if not ok:
        return False
    claw.open()
    yield from wait(0.5)
    ok = yield from move_to(box)                   # straight in
    if not ok:
        return False
    claw.close()
    yield from wait(0.6)
    return True


def place(slot):
    slot = _floor(slot)
    behind = _behind(slot)
    z_safe = max(_cmd[2], slot[2]) + LIFT
    path = [
        _cmd + np.array([0.0, 0.0, LIFT]),                       # lift the box off the floor
        np.array([behind[0], behind[1], z_safe]),                # carry it round (arc) above the slot approach
        behind,                                                  # come down behind the slot
        slot,                                                    # straight in
    ]
    pitch = pick_pitch(path)
    if pitch is None:
        print("Slot unreachable within the joint limits")
        return False
    for p in path:
        ok = yield from move_to(p, pitch=pitch)
        if not ok:
            return False
    claw.open()
    yield from wait(0.5)
    ok = yield from move_to(behind)                # straight back out
    claw.close()
    yield from wait(0.5)
    return ok


# ---------------------------------------------------------------- main sequence
scanFinished = False
grabFinished = False
placeFinished = False


def run_cycle():
    global scanFinished, grabFinished, placeFinished

    while S.SLOT_INDEX < len(S.SLOTS) and S.SLOT_INDEX < len(S.BOXES):
        scanFinished = grabFinished = placeFinished = False

        # 0. scan pose (wrist curled in, out of the webcam's view)
        claw.close()
        if len(pose_list) > 0:
            play_pose(pose_list[0])
            print("SCAN")
            yield from wait(1.5)

        currentBox = S.BOXES[S.SLOT_INDEX]["pos"]
        currentSlot = S.SLOTS[S.SLOT_INDEX]["pos"]

        S.set_bot_state(S.BotState.SCANNING, "Using Index Position")
        scanFinished = True
        print("SCAN DONE")
        yield

        S.set_bot_state(S.BotState.GRABBING, "Grabbing")
        print("GRAB")
        grabFinished = yield from grab(currentBox)
        if not grabFinished:
            print("GRAB FAILED, skipping box", S.SLOT_INDEX)
            S.SKIPPED.append(S.SLOT_INDEX)
            S.SLOT_INDEX += 1
            continue
        print("GRAB DONE")

        S.set_bot_state(S.BotState.PLACING, "Placing")
        print("PLACE")
        placeFinished = yield from place(currentSlot)
        print("PLACE DONE" if placeFinished else "PLACE FAILED")

        S.SLOT_INDEX += 1
        yield