# sim callbacks
import glfw
import mujoco

import numpy as np
from simfiles import sim_state as S
from simfiles.controls import move_joint, play_pose, pose_list
from simfiles.ik import solve_ik
from simfiles import claw as claw

# key currently held
current_key = 0

# key -> (joint, direction)
KEY_BINDINGS = {
    glfw.KEY_N: (0, -1), glfw.KEY_M: (0, 1),  # torso yaw
    glfw.KEY_Z: (1, -1), glfw.KEY_X: (1, 1),  # shoulder pitch
    glfw.KEY_A: (2, -1), glfw.KEY_S: (2, 1),  # upper arm pitch
    glfw.KEY_Q: (3, -1), glfw.KEY_W: (3, 1),  # forearm pitch
    glfw.KEY_J: (4, -1), glfw.KEY_K: (4, 1),  # hand yaw
}

def keyboard_callback(window, key, scancode, act, mods):
    global current_key
    if act == glfw.RELEASE:
        current_key = 0
    if act != glfw.PRESS:
        return
    current_key = key

    for p in pose_list:
        if key == p.get_key():
            play_pose(p)

    if key == glfw.KEY_ENTER:     # start / stop the automatic sequence
        S.AUTO = not S.AUTO
        print("auto:", S.AUTO)
        return
    S.AUTO = False   

    if key == glfw.KEY_1: # open claw
        claw.open()
    elif key == glfw.KEY_2: # close claw
        claw.close()
    elif key == glfw.KEY_3: # IK behind target then move forward
        backup = S.d.ctrl.copy()
        actual_target = S.d.mocap_pos[0].copy()
        direction = actual_target[:2] - S.BASE_XY
        distance = np.linalg.norm(direction)

        ik_target = actual_target.copy()
        if distance > 1e-8:
            ik_target[:2] -= direction / distance * 0.06

        if solve_ik(S.m, S.d, "endeff", ik_target, pitch=0.0, locked={5: 0.0}):
            S.set_bot_state(S.BotState.INVERSE_K, "Reaching Target...")
        else:
            S.d.ctrl[:] = backup
            print("Unreachable")
    elif key == glfw.KEY_4:  # trigger IK to the target position
        backup = S.d.ctrl.copy()
        target = S.d.mocap_pos[0].copy()   # the green sphere in your XML
        if solve_ik(S.m, S.d, "endeff", target, pitch=0.0, locked={5: 0.0}):
            S.set_bot_state(S.BotState.INVERSE_K, "Reaching Target...")
        else:
            S.d.ctrl[:] = backup
            print("Unreachable")

def key_held_update():
    binding = KEY_BINDINGS.get(current_key)
    if binding:
        move_joint(*binding)


def is_key_held():
    return current_key != 0


# mouse / camera
button_left = False
button_middle = False
button_right = False
lastx = 0.0
lasty = 0.0


def glfw_error_callback(code, desc):
    print(f"glfw error {code}: {desc}")


def mouse_button_callback(window, button, act, mods):
    global button_left, button_middle, button_right, lastx, lasty
    pressed = act == glfw.PRESS
    if act in (glfw.PRESS, glfw.RELEASE):
        if button == glfw.MOUSE_BUTTON_LEFT:
            button_left = pressed
        if button == glfw.MOUSE_BUTTON_MIDDLE:
            button_middle = pressed
        if button == glfw.MOUSE_BUTTON_RIGHT:
            button_right = pressed

    lastx, lasty = glfw.get_cursor_pos(window)


def cursor_pos_callback(window, xpos, ypos):
    global lastx, lasty
    if S.m is None:
        return
    if not (button_left or button_right or button_middle):
        lastx, lasty = xpos, ypos
        return

    _, fbh = glfw.get_framebuffer_size(window)
    fbh = max(fbh, 1)

    dx = (xpos - lastx) / fbh
    dy = (ypos - lasty) / fbh
    lastx, lasty = xpos, ypos

    if button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
    elif button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(S.m, action, dx, dy, S.scn, S.cam)


def scroll_callback(window, xoffset, yoffset):
    if S.m is None:
        return
    mujoco.mjv_moveCamera(S.m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.1 * yoffset, S.scn, S.cam)