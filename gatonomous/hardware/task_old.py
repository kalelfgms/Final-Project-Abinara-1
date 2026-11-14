import mujoco
import numpy as np

from simfiles import sim_state as S
from simfiles import claw
from simfiles.ik import solve_ik
from simfiles.controls import play_pose, pose_list

def wait(seconds):
    t0 = S.d.time
    while S.d.time - t0 < seconds:
        yield

def wait_frames(seconds): #seconds input  #60 = 1s
    """Yields control back to the simulation loop for a set number of frames."""
    for _ in range(seconds*60): # cuz sim is 60 fps 
        yield

def behind_target(targetpos):
    backup = S.d.ctrl.copy()
    actual_target = np.array(targetpos, dtype=float)
    ik_target = actual_target.copy()
    
    direction = actual_target[:2] - S.BASE_XY
    distance = np.linalg.norm(direction)

    if distance > 1e-8:
        ik_target[:2] -= direction / distance * 0.06

    if solve_ik(S.m, S.d, "endeff", ik_target, pitch=0.0, locked={5: 0.0}):
        S.set_bot_state(S.BotState.INVERSE_K, "Reaching Target...")
        return True
    else:
        S.d.ctrl[:] = backup
        print("Unreachable")
        return False

def to_target(targetpos):
    backup = S.d.ctrl.copy()
    actual_target = np.array(targetpos, dtype=float)

    if solve_ik(S.m, S.d, "endeff", actual_target, pitch=0.0, locked={5: 0.0}):
        S.set_bot_state(S.BotState.INVERSE_K, "Reaching Target...")
        return True
    else:
        S.d.ctrl[:] = backup
        print("Unreachable")
        return False

def grab(slot):
    if behind_target(slot):
        yield from wait_frames(2)  # Wait for arm to reach behind position
    claw.open()
    yield from wait_frames(1)      # Wait for claw action
    if to_target(slot):
        yield from wait_frames(2)  # Wait for arm to reach target position
    claw.close()
    yield from wait_frames(1)

def place(slot):
    if to_target(slot):
        yield from wait_frames(3)
    claw.open()
    yield from wait_frames(1)
    if behind_target(slot):
        yield from wait_frames(3)
    claw.close()
    yield from wait_frames(1)

# Global state flags
scanFinished = False
grabFinished = False
placeFinished = False

def run_cycle():
    """Generator function that visits each slot frame-by-frame when driven by next(task)."""
    global scanFinished, grabFinished, placeFinished

    while S.SLOT_INDEX < len(S.SLOTS) and S.SLOT_INDEX < len(S.BOXES):
        scanFinished = False
        grabFinished = False
        placeFinished = False

        if 'play_pose' in globals() and 'pose_list' in globals() and len(pose_list) > 0:
            play_pose(pose_list[0])
            print("SCAN")
            yield from wait_frames(5)

        currentBox = S.BOXES[S.SLOT_INDEX]["pos"]
        currentSlot = S.SLOTS[S.SLOT_INDEX]["pos"]

        # 1. Scanning Phase
        S.set_bot_state(S.BotState.SCANNING, "Using Index Position")
        scanFinished = True
        print("SCAN DONE")
        yield

        # 2. Grabbing Phase
        if scanFinished:
            S.set_bot_state(S.BotState.GRABBING, "Grabbing")
            print("GRAB")
            yield from grab(currentBox)
            grabFinished = True
            print("GRAB DONE")

        # 3. Placing Phase
        if grabFinished:
            S.set_bot_state(S.BotState.PLACING, "Placing")
            print("PLACE")
            yield from place(currentSlot)
            placeFinished = True
            print("PLACE DONE")

        print("CURRENT TARGET")
        print("CURRENT SLOT")
        S.SLOT_INDEX += 1
        yield