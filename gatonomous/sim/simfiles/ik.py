import mujoco
import numpy as np

# Starting guesses for the three pitch joints [shoulder, elbow, wrist] (joint2, joint3, joint5 of ARM.xml).
# They are only seeds: the solver refines them. Tuned for this arm's joint signs.
PLANAR_SEEDS = [(0.0, 0.0, 0.0), (-0.4, 0.25, -0.13), (-0.8, 0.8, 0.0), (0.4, -0.4, 0.0),
                (-0.3, -0.3, 0.3), (0.6, 0.6, -0.6), (-1.0, 1.5, 0.5)]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def solve_ik(m, d, site_name, target, pitch=None, locked=None,
             approach_axis=(0.0, -1.0, 0.0), pitch_weight=0.2,
             max_iter=300, tolerance=1e-3, ang_tol=1e-2, lam=0.05, verbose=True,
             seed_ctrl=None, multi_start=True):
    """Damped least squares IK on the actuated joints that lead to `site_name`.
    Writes the result into d.ctrl and returns True on success.

    target         world position (3,) the site should reach
    pitch          desired elevation of the claw's approach direction in radians
                   (0 = horizontal, negative = pointing down). None = ignore orientation.
    locked         {actuator_index: angle_rad}; these joints are held at that angle and excluded
                   from the solve. Entries that aren't on the arm chain (e.g. the claw) are ignored.
    approach_axis  the site-frame axis the claw points along (default: -y of Link_5).

    NEW
    seed_ctrl      array of actuator values (len nu) to start from instead of the *measured* pose
                   d.qpos. Pass the previous d.ctrl when following a path, so every step continues
                   from the last solution (no jumping between IK branches, no dependence on how far
                   the real arm has lagged behind its commands).
    multi_start    False = only try the seed (small path steps: stays on the same arm configuration).
                   True  = also retry from the standard arm shapes (big jumps / first deploy).
    """
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        print(f"IK: site '{site_name}' not found")
        return False

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

    locked = {i: v for i, v in dict(locked or {}).items() if i in act}   # never touch non-chain actuators
    free = [i for i in act if i not in locked]
    dofs = [dadr(i) for i in free]
    jnt_free = [jnt(i) for i in free]

    target = np.asarray(target, dtype=float)
    temp = mujoco.MjData(m)
    temp.qpos[:] = d.qpos  # start from the current pose...
    if seed_ctrl is not None:  # ...or from the previous command
        for i in act:
            temp.qpos[qadr(i)] = seed_ctrl[i]
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

    def set_q(vec):
        for k, i in enumerate(free):
            temp.qpos[qadr(i)] = vec[k]

    def clip_q(k, q):
        j = jnt_free[k]
        if m.jnt_limited[j]:
            return float(np.clip(q, m.jnt_range[j, 0], m.jnt_range[j, 1]))
        return float(q)

    def descend(q_start, iters):
        """Damped least squares from q_start -> (joint values, position error, angle error)."""
        set_q(q_start)
        for _ in range(iters):
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
                temp.qpos[qadr(i)] = clip_q(k, temp.qpos[qadr(i)] + dq[k])

        tip, a = tip_and_dir()
        pos_err = float(np.linalg.norm(target - tip))
        ang_err = float(abs(np.sin(pitch) - a[2])) if pitch is not None else 0.0
        return np.array([temp.qpos[qadr(i)] for i in free]), pos_err, ang_err

    # which free joint is the yaw (vertical axis) joint?
    tip_and_dir()
    yaw_k = None
    for k, j in enumerate(jnt_free):
        if abs(temp.xaxis[j][2]) > 0.9:
            yaw_k = k
            break

    def aim_at_target(seed):
        """Same arm shape as `seed`, but with the yaw turned so the arm points at the target."""
        s = np.array(seed, dtype=float)
        if yaw_k is None:
            return s
        s[yaw_k] = 0.0
        set_q(s)
        tip, _ = tip_and_dir()
        j = jnt_free[yaw_k]
        ax, ay = temp.xanchor[j][:2]
        sign = 1.0 if temp.xaxis[j][2] > 0 else -1.0
        az_tip = np.arctan2(tip[1] - ay, tip[0] - ax)
        az_tgt = np.arctan2(target[1] - ay, target[0] - ax)
        s[yaw_k] = clip_q(yaw_k, sign * _wrap(az_tgt - az_tip))
        return s

    # starting points: the seed arm shape first (smooth motion), then standard shapes
    cur = np.array([temp.qpos[qadr(i)] for i in free])
    candidates = [(cur, max_iter)]
    planar = [k for k in range(len(free)) if k != yaw_k]
    if multi_start and len(planar) == 3:
        for s3 in PLANAR_SEEDS:
            v = cur.copy()
            for k, val in zip(planar, s3):
                v[k] = val
            candidates.append((v, 100))

    best = None
    for start, iters in candidates:
        start = aim_at_target(start)
        q, pos_err, ang_err = descend(start, iters)
        flipped = yaw_k is not None and abs(_wrap(q[yaw_k] - start[yaw_k])) > 1.0   # arm swung round backwards
        ok = pos_err < tolerance * 2.0 and ang_err < ang_tol * 2.0 and not flipped
        if ok or best is None or pos_err + ang_err < best[1] + best[2]:
            best = (q, pos_err, ang_err)
        if ok:
            break

    q_best, pos_err, ang_err = best
    set_q(q_best)
    tip, a = tip_and_dir()
    if verbose:
        print(f"IK: target {np.round(target, 3)}  reached {np.round(tip, 3)}  "
              f"pos error {pos_err:.4f} m  claw elevation {np.degrees(np.arcsin(np.clip(a[2], -1, 1))):.1f} deg")

    # joint angles -> actuator targets (clamped to each actuator's ctrlrange)
    for k, i in enumerate(free):
        lo, hi = m.actuator_ctrlrange[i]
        d.ctrl[i] = np.clip(q_best[k], lo, hi)
    for i, v in locked.items():
        lo, hi = m.actuator_ctrlrange[i]
        d.ctrl[i] = np.clip(v, lo, hi)

    return pos_err < tolerance * 2.0 and ang_err < ang_tol * 2.0