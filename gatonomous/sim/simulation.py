#simulationzz, convert dari cpp
import sys

import glfw
import mujoco
import numpy as np

from simfiles import sim_state as S
from simfiles import callbacks as cb
from simfiles import claw
from simfiles.controls import play_pose, pose_list
from simfiles.poses import initialize_poses
from simfiles.task import run_cycle

#MODEL_PATH = "/home/kalelgms/Desktop/Projects/final_project/gatonomous/models/ARM.xml"
MODEL_PATH = "/home/kalelgms/Desktop/Projects/final_project/gatonomous/models/ACCURATE_ARM.xml"
#ACCURATE_ARM.xml udh diedit sesuai ama torque n stuff servo asli

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH

    try:
        S.m = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Couldn't load model: {e}", file=sys.stderr)
        return 1
    S.d = mujoco.MjData(S.m)
    initialize_poses()
    play_pose(pose_list[0])

    for i in range(S.m.nu):
        lo, hi = S.m.actuator_ctrlrange[i]
        S.d.ctrl[i] = np.clip(S.d.ctrl[i], lo, hi)
        S.d.qpos[S.m.jnt_qposadr[S.m.actuator_trnid[i, 0]]] = S.d.ctrl[i]
    S.d.qpos[S.m.joint("gear_L").qposadr[0]] = -S.d.qpos[S.m.joint("gear_R").qposadr[0]]
    mujoco.mj_forward(S.m, S.d)
    #claw.calibrate()

    # error
    glfw.set_error_callback(cb.glfw_error_callback)
    if not glfw.init():
        print("glfwInit() gagal", file=sys.stderr)
        return 1

    # window
    window = glfw.create_window(1500, 1500, "ARM SIM", None, None)
    if not window:
        print("glfwCreateWindow() gagal", file=sys.stderr)
        glfw.terminate()
        return 1
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # callbacks
    glfw.set_mouse_button_callback(window, cb.mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cb.cursor_pos_callback)
    glfw.set_scroll_callback(window, cb.scroll_callback)
    glfw.set_key_callback(window, cb.keyboard_callback)

    # scene setup
    S.opt = mujoco.MjvOption()
    S.scn = mujoco.MjvScene(S.m, maxgeom=2000)
    S.con = mujoco.MjrContext(S.m, mujoco.mjtFontScale.mjFONTSCALE_100)

    # cam setup
    S.cam = mujoco.MjvCamera()
    S.cam.lookat[2] = 0.15  # cam origin up a bit
    S.cam.orthographic = 1
    S.cam.distance = 1.5
    S.cam.azimuth = 45
    S.cam.elevation = -25

    tip_id = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_SITE, "endeff")
    task = None
    SUBSTEPS = 10

    # sim loop, scan, detect, move, till all pyramid items are placed
    while not glfw.window_should_close(window):
        if cb.is_key_held():  # detect key holding
            cb.key_held_update()

        S.d.qfrc_applied[:] = 0.0 # S.d.qfrc_bias
        for _ in range(SUBSTEPS):
            mujoco.mj_step(S.m, S.d)

        ##
        if S.AUTO:
            S.MANUAL_TASK = None
            if task is None:
                task = run_cycle()
            try:
                next(task)
            except StopIteration:
                task = None
                S.AUTO = False
        else:
            task = None
            if S.MANUAL_TASK is not None:
                try:
                    next(S.MANUAL_TASK)
                except StopIteration:
                    S.MANUAL_TASK = None
        ##

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)

        mujoco.mjv_updateScene(S.m, S.d, S.opt, None, S.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, S.scn)
        mujoco.mjr_render(viewport, S.scn, S.con)

        glfw.swap_buffers(window)
        glfw.poll_events()

    # cleanup
    S.con.free()
    glfw.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())