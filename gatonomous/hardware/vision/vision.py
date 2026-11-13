# access webcam
# get video feed
# blob detection, detect green, orange, beige
# put all blobs in list
# or, get the screen coordinates of those blobs and put them on a list
# find way to turn screen space coords into world coordinates

# choose coordinates that are closest to bot/origin
# return those coordinates to driver

# vision.py -- stand-in for the camera, so the scan/grab/place sequence can run in the sim now.
# The real detector keeps the same signature: detect(color, skip) -> xyz (world frame) or None.

import numpy as np

from simfiles import sim_state as S


def detect(color, skip=()):
    """xyz of the next box of this color, or None.

    SIM STUB: reads S.BOXES (hand-written positions). Replace the body with the real thing later:
    camera frame -> HSV mask -> centroid -> pixel-to-world mapping. Boxes in `skip` (positions the
    arm already failed to reach) are ignored.
    """
    best = None
    for b in S.BOXES:
        if b["color"] != color:
            continue
        p = np.array(b["pos"], dtype=float)
        if any(np.linalg.norm(p - s) < 0.02 for s in skip):
            continue
        dist = np.linalg.norm(p[:2] - S.BASE_XY)
        if best is None or dist < best[0]:
            best = (dist, p)
    return None if best is None else best[1]


def consume(box):
    """Sim stub only: remove a picked box. A real camera sees it vanish from the floor by itself."""
    box = np.asarray(box, dtype=float)
    for b in S.BOXES:
        if np.linalg.norm(np.array(b["pos"], dtype=float) - box) < 1e-6:
            S.BOXES.remove(b)
            return