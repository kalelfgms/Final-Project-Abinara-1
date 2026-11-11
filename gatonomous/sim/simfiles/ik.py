import mujoco
import numpy as np


def solve_ik(m, d, site_name, target, pitch=None, locked=None,
             approach_axis=(0.0, -1.0, 0.0), pitch_weight=0.2,
             max_iter=300, tolerance=1e-3, ang_tol=1e-2, lam=0.05, verbose=True):
    """Damped least squares IK on the actuated joints that lead to `site_name`.
    Writes the result into d.ctrl and returns True on success.

    target         world position (3,) the site should reach
    pitch          desired elevation of the claw's approach direction in radians
                   (0 = horizontal, negative = pointing down). None = ignore orientation.
    locked         {actuator_index: angle_rad}; these joints are held at that angle
                   and excluded from the solve, e.g. {4: 0.0} to fix the wrist.
    approach_axis  the site-frame axis the claw points along (default: -y of Link_5,
                   the direction the end effector site is offset in the XML).
    """
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        print(f"IK: site '{site_name}' not found")
        return False

    locked = dict(locked or {})

    def jnt(i):  return int(m.actuator_trnid[i, 0])
    def qadr(i): return int(m.jnt_qposadr[jnt(i)])
    def dadr(i): return int(m.jnt_dofadr[jnt(i)])

    # actuated joints on the chain leading to the site (this leaves the gripper gears alone)
    chain = set()
    body = int(m.site_bodyid[site_id])
    while body != 0:
        chain.add(body)
        body = int(m.body_parentid[body])
    act = [i for i in range(m.nu) if int(m.jnt_bodyid[jnt(i)]) in chain]
    free = [i for i in act if i not in locked]
    dofs = [dadr(i) for i in free]

    temp = mujoco.MjData(m)
    temp.qpos[:] = d.qpos  # start from the current pose
    for i, v in locked.items():
        temp.qpos[qadr(i)] = v

    axis = np.asarray(approach_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    def tip_and_dir():
        mujoco.mj_kinematics(m, temp)
        mujoco.mj_comPos(m, temp)
        a = temp.site_xmat[site_id].reshape(3, 3) @ axis  # claw direction in world
        return temp.site_xpos[site_id].copy(), a

    for _ in range(max_iter):
        tip, a = tip_and_dir()
        err_p = target - tip
        e_a = (np.sin(pitch) - a[2]) if pitch is not None else 0.0
        if np.linalg.norm(err_p) < tolerance and abs(e_a) < ang_tol / 10:
            break

        mujoco.mj_jacSite(m, temp, jacp, jacr, site_id)
        J = jacp[:, dofs]
        e = err_p
        if pitch is not None:
            # d(a_z)/dq = (omega_k x a)_z for each joint k
            da_z = np.cross(jacr[:, dofs].T, a)[:, 2]
            J = np.vstack([J, pitch_weight * da_z])
            e = np.append(err_p, pitch_weight * e_a)

        # dq = J^T (J J^T + lambda^2 I)^-1 e
        A = J @ J.T + lam ** 2 * np.eye(len(e))
        dq = J.T @ np.linalg.solve(A, e)

        for k, i in enumerate(free):
            q = temp.qpos[qadr(i)] + dq[k]
            if m.jnt_limited[jnt(i)]:
                q = np.clip(q, m.jnt_range[jnt(i), 0], m.jnt_range[jnt(i), 1])
            temp.qpos[qadr(i)] = q

    tip, a = tip_and_dir()
    pos_err = np.linalg.norm(target - tip)
    ang_err = abs(np.sin(pitch) - a[2]) if pitch is not None else 0.0
    if verbose:
        print(f"IK: target {np.round(target, 3)}  reached {np.round(tip, 3)}  "
              f"pos error {pos_err:.4f} m  claw elevation {np.degrees(np.arcsin(a[2])):.1f} deg")

    # joint angles -> actuator targets (clamped to each actuator's ctrlrange)
    for i in free:
        lo, hi = m.actuator_ctrlrange[i]
        d.ctrl[i] = np.clip(temp.qpos[qadr(i)], lo, hi)
    for i, v in locked.items():
        lo, hi = m.actuator_ctrlrange[i]
        d.ctrl[i] = np.clip(v, lo, hi)

    return pos_err < tolerance * 2.0 and ang_err < ang_tol * 2.0