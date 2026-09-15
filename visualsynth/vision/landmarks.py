"""Hand landmark constants and geometry helpers.

MediaPipe 1.0 dropped `mediapipe.solutions`, so the connection list that used to
come from `mp.solutions.hands.HAND_CONNECTIONS` is spelled out here, along with
the joint triples each finger's extension test needs (FR-3.1, FR-3.3).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

LANDMARK_COUNT = 21

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class Finger(IntEnum):
    """Finger order matches the degree order: thumb = first degree of the hand."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


FINGER_COUNT = len(Finger)

#: (proximal, joint, distal) - the interior angle is measured at `joint`.
FINGER_JOINTS: dict[Finger, tuple[int, int, int]] = {
    Finger.THUMB: (THUMB_MCP, THUMB_IP, THUMB_TIP),
    Finger.INDEX: (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    Finger.MIDDLE: (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    Finger.RING: (RING_MCP, RING_PIP, RING_TIP),
    Finger.PINKY: (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

FINGER_TIPS: dict[Finger, int] = {
    Finger.THUMB: THUMB_TIP,
    Finger.INDEX: INDEX_TIP,
    Finger.MIDDLE: MIDDLE_TIP,
    Finger.RING: RING_TIP,
    Finger.PINKY: PINKY_TIP,
}

#: Bones, for drawing the skeleton overlay (UX-2.2).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
)

#: Bones per finger, used to thicken a sounding finger's chain (UX-2.3).
FINGER_CHAINS: dict[Finger, tuple[int, ...]] = {
    Finger.THUMB: (WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    Finger.INDEX: (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    Finger.MIDDLE: (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    Finger.RING: (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    Finger.PINKY: (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


def interior_angle_deg(points: np.ndarray, a: int, b: int, c: int) -> float:
    """Angle at `b` in the triangle a-b-c, in degrees (180 = straight)."""
    v1 = points[a] - points[b]
    v2 = points[c] - points[b]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cosine = float(np.dot(v1, v2)) / (n1 * n2)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def palm_size(points: np.ndarray) -> float:
    """Wrist-to-middle-knuckle distance: the scale every ratio is measured in."""
    return float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]))


def to_isotropic(normalized: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale normalised landmarks to pixel units so angles are not skewed by the
    frame's aspect ratio. MediaPipe's z is on roughly the same scale as x."""
    scaled = np.asarray(normalized, dtype=np.float64).copy()
    scaled[:, 0] *= width
    scaled[:, 1] *= height
    if scaled.shape[1] > 2:
        scaled[:, 2] *= width
    return scaled
