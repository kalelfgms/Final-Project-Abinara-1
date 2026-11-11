import sys
from enum import Enum

import glfw
import mujoco
import numpy as np

MODEL_PATH = "/home/kalelgms/Desktop/Projects/final_project/gatonomous/models/ARM.xml"

# ---------------------------------------------------------------- globals
# mujoco / sim objects (set in main, used by the callbacks)
m = None
d = None
scn = None
cam = None
opt = None
con = None

target_pos = np.array([0.3, 0.5, 0.3])  # IK target


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


# ---------------------------------------------------------------- poses
class Pose:
    def __init__(self, name: str, key: int, orients):
        self.name = name
        self.key = key  # glfw key that plays this pose
        self.orients = list(orients)  # 5 values, one per actuator


pose_list = [Pose("Start", 0, [0, 0, 0, 0, 0])]  # index 0 = start pos
# e.g. pose_list.append(Pose("Wave", glfw.KEY_1, [0.5, 0.2, -0.3, 0.1, 0]))


def play_pose(pose: Pose):
    set_bot_state(BotState.POSED, pose.name)
    set_pose(pose)


def set_pose(pose: Pose):
    for i in range(5):
        d.ctrl[i] = pose.orients[i]


# ---------------------------------------------------------------- controls
def move_joint(joint: int, direction: float):
    """direction: left -1, right 1"""
    step = direction * 1
    vmax = 6  # max speed so one key press doesn't jump to the limit
    ctrl_min, ctrl_max = m.actuator_ctrlrange[joint]

    set_bot_state(BotState.TELEOP, "-")
    d.ctrl[joint] = np.clip(d.ctrl[joint] + step * vmax * m.opt.timestep,
                            ctrl_min, ctrl_max)


# ---------------------------------------------------------------- IK
def solve_ik(site_name, target, max_iter=300, tolerance=1e-3, lam=0.05):
    """Damped least squares IK. Writes the result into d.ctrl.
    Returns True if the final error is within 2 * tolerance."""
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return False

    nv = m.nv
    temp = mujoco.MjData(m)
    temp.qpos[:] = d.qpos  # start from the current pose

    jacp = np.zeros((3, nv))

    for _ in range(max_iter):
        mujoco.mj_kinematics(m, temp)
        mujoco.mj_comPos(m, temp)
        err = target - temp.site_xpos[site_id]
        if np.linalg.norm(err) < tolerance:
            break

        mujoco.mj_jacSite(m, temp, jacp, None, site_id)

        # dq = J^T (J J^T + lambda^2 I)^-1 err
        A = jacp @ jacp.T + lam ** 2 * np.eye(3)
        dq = jacp.T @ np.linalg.solve(A, err)

        for j in range(nv):  # hinge joints: qpos index == dof index
            q = temp.qpos[j] + dq[j]
            if m.jnt_limited[j]:
                q = np.clip(q, m.jnt_range[j, 0], m.jnt_range[j, 1])
            temp.qpos[j] = q

    mujoco.mj_kinematics(m, temp)
    final_error = np.linalg.norm(target - temp.site_xpos[site_id])

    # joint angles -> actuator targets (clamped to each actuator's ctrlrange)
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        q = temp.qpos[m.jnt_qposadr[jid]]
        ctrl_min, ctrl_max = m.actuator_ctrlrange[i]
        d.ctrl[i] = np.clip(q, ctrl_min, ctrl_max)

    return final_error < tolerance * 2.0


# ---------------------------------------------------------------- callbacks
current_key = 0  # key currently held

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
        if key == p.key:
            play_pose(p)

    if key == glfw.KEY_T:  # trigger IK to the target position
        backup = d.ctrl.copy()
        if solve_ik("endeff", target_pos):
            set_bot_state(BotState.INVERSE_K, "Reaching Target...")
        else:
            d.ctrl[:] = backup  # don't keep a failed solution
            print("Unreachable")


def key_held_update():
    binding = KEY_BINDINGS.get(current_key)
    if binding:
        move_joint(*binding)


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
    if m is None:
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
    mujoco.mjv_moveCamera(m, action, dx, dy, scn, cam)


def scroll_callback(window, xoffset, yoffset):
    if m is None:
        return
    mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.1 * yoffset, scn, cam)


# ---------------------------------------------------------------- main
def main():
    global m, d, scn, cam, opt, con

    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH

    try:
        m = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Couldn't load model: {e}", file=sys.stderr)
        return 1
    d = mujoco.MjData(m)

    glfw.set_error_callback(glfw_error_callback)
    if not glfw.init():
        print("glfwInit() gagal", file=sys.stderr)
        return 1

    window = glfw.create_window(900, 1200, "arm sim", None, None)
    if not window:
        print("glfwCreateWindow() gagal", file=sys.stderr)
        glfw.terminate()
        return 1
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, keyboard_callback)

    # scene setup
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(m, maxgeom=2000)
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)

    # cam setup
    cam = mujoco.MjvCamera()
    cam.lookat[2] = 0  # cam origin up a bit
    cam.orthographic = 1
    cam.distance = 1.8
    cam.azimuth = 45
    cam.elevation = -25

    # starting pose
    play_pose(pose_list[0])

    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "endeff")

    # sim loop
    while not glfw.window_should_close(window):
        if current_key != 0:  # detect key holding
            key_held_update()

        mujoco.mj_step(m, d)

        # IK: only leave InverseK once the tip has reached the target and settled
        if STATE == BotState.INVERSE_K:
            dist = np.linalg.norm(d.site_xpos[tip_id] - target_pos)
            if dist < 0.01 and np.linalg.norm(d.qvel) < 0.05:
                set_bot_state(BotState.TELEOP, "Reached Target!!")

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)

        mujoco.mjv_updateScene(m, d, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

    con.free()
    glfw.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())