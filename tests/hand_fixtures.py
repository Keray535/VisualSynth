"""Synthetic hand landmarks, so the vision logic is testable without a camera.

A hand is built pointing up in normalised image coordinates; each finger is
placed at a requested interior angle, which is exactly what the extension test
measures.
"""

from __future__ import annotations

import math

import numpy as np

from visualsynth.vision.finger_state import HandObservation
from visualsynth.vision.landmarks import (
    Finger,
    LANDMARK_COUNT,
    to_isotropic,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

_WRIST = (0.50, 0.92)
_MCP_Y = 0.62
_MCP_X = {
    Finger.INDEX: 0.42,
    Finger.MIDDLE: 0.50,
    Finger.RING: 0.57,
    Finger.PINKY: 0.63,
}
_PROXIMAL_LEN = 0.11
_DISTAL_LEN = 0.13

STRAIGHT_DEG = 178.0
CURLED_DEG = 60.0


def _finger_points(finger: Finger, angle_deg: float) -> list[tuple[float, float]]:
    """MCP, PIP, DIP, TIP with the given interior angle at the PIP joint."""
    mcp = np.array([_MCP_X[finger], _MCP_Y])
    pip = mcp + np.array([0.0, -_PROXIMAL_LEN])  # straight up from the knuckle

    # direction MCP -> PIP, then bend away from straight by (180 - angle)
    forward = math.radians(-90.0)  # up, in image coordinates (y grows downward)
    bend = math.radians(180.0 - angle_deg)
    theta = forward + bend
    tip = pip + _DISTAL_LEN * np.array([math.cos(theta), math.sin(theta)])
    dip = pip + 0.55 * (tip - pip)
    return [tuple(mcp), tuple(pip), tuple(dip), tuple(tip)]


def _thumb_points(angle_deg: float, abduction: float) -> list[tuple[float, float]]:
    """CMC, MCP, IP, TIP.

    `abduction` interpolates the thumb tip between tucked against the index
    knuckle (0) and swung wide of the palm (1), which is what the abduction
    ratio in the extension test measures.
    """
    wrist = np.array(_WRIST)
    tucked = np.array([_MCP_X[Finger.INDEX] + 0.01, _MCP_Y + 0.03])
    swung = np.array([_WRIST[0] - 0.26, _WRIST[1] - 0.10])
    target = tucked + (swung - tucked) * float(np.clip(abduction, 0.0, 1.0))

    span = target - wrist
    length = float(np.linalg.norm(span)) / 3.2
    direction = span / max(float(np.linalg.norm(span)), 1e-9)

    cmc = wrist + direction * length * 0.9
    mcp = cmc + direction * length
    ip = mcp + direction * length

    bend = math.radians(180.0 - angle_deg)
    cos_b, sin_b = math.cos(bend), math.sin(bend)
    rotated = np.array(
        [
            direction[0] * cos_b - direction[1] * sin_b,
            direction[0] * sin_b + direction[1] * cos_b,
        ]
    )
    tip = ip + rotated * length
    return [tuple(cmc), tuple(mcp), tuple(ip), tuple(tip)]


def make_landmarks(
    angles: dict[Finger, float] | None = None,
    thumb_abduction: float = 1.0,
) -> np.ndarray:
    """(21, 3) normalised landmarks. `angles` gives the interior angle per finger."""
    angles = angles or {}
    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
    points[0, :2] = _WRIST

    thumb_angle = angles.get(Finger.THUMB, STRAIGHT_DEG)
    for i, (x, y) in enumerate(_thumb_points(thumb_angle, thumb_abduction), start=1):
        points[i, :2] = (x, y)

    starts = {Finger.INDEX: 5, Finger.MIDDLE: 9, Finger.RING: 13, Finger.PINKY: 17}
    for finger, start in starts.items():
        angle = angles.get(finger, STRAIGHT_DEG)
        for offset, (x, y) in enumerate(_finger_points(finger, angle)):
            points[start + offset, :2] = (x, y)
    return points


def make_hand(
    handedness: str = "Right",
    extended: dict[Finger, bool] | None = None,
    angles: dict[Finger, float] | None = None,
    thumb_abduction: float | None = None,
    score: float = 0.95,
) -> HandObservation:
    """A hand observation with the requested fingers extended or curled."""
    extended = extended or {}
    resolved = dict(angles or {})
    for finger in Finger:
        if finger in resolved:
            continue
        resolved[finger] = STRAIGHT_DEG if extended.get(finger, False) else CURLED_DEG

    if thumb_abduction is None:
        thumb_abduction = 1.0 if resolved[Finger.THUMB] > 120.0 else 0.15

    normalized = make_landmarks(resolved, thumb_abduction)
    return HandObservation(
        handedness=handedness,
        score=score,
        landmarks_norm=normalized,
        landmarks_px=to_isotropic(normalized, FRAME_WIDTH, FRAME_HEIGHT),
    )


def open_hand(handedness: str = "Right", **kwargs) -> HandObservation:
    return make_hand(handedness, extended=dict.fromkeys(Finger, True), **kwargs)


def fist(handedness: str = "Right", **kwargs) -> HandObservation:
    return make_hand(handedness, extended=dict.fromkeys(Finger, False), **kwargs)
