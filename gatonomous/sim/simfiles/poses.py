# create poses (like poses.hpp)
import glfw

from simfiles.controls import Pose, pose_list

def initialize_poses():
    pose_list.clear()
    pose_list.append(Pose("Scan", glfw.KEY_SPACE, [0, 0.75, -0.35, 2.5])) # scan pose
    
    # pose_list.append(Pose("Start", glfw.KEY_F, [0, -1, 2, 1.5, 0]))  # idx 1
    # pose_list.append(Pose("ZigZag", glfw.KEY_C, [0, -1, 2, -2, 0]))
    # pose_list.append(Pose("Zero", glfw.KEY_V, [0, 0.05, -0.05, 0.05, 0]))
    # pose_list.append(Pose("Max Length", glfw.KEY_B, [0, -1.5, -1, 1, 0]))

    # pose_list.append(Pose("Tri1", glfw.KEY_1, [0, -0.7, 1.85, 0.4, 0]))
    # pose_list.append(Pose("Tri2", glfw.KEY_2, [0.3, -0.05, 2.1, 0, -0.5]))
    # pose_list.append(Pose("Tri3", glfw.KEY_3, [-0.3, -0.05, 2.1, 0, -0.5]))

    