"""Finger extension -> note events.

Extended finger = sustained note, curled = silent (the locked trigger rule).
Two guards keep that stable: hysteresis on the joint angle (FR-3.2) and an
asymmetric debounce - fast note-on, slow note-off - so a one-frame tracking
dropout never cuts a held note (FR-3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .landmarks import (
    FINGER_JOINTS,
    INDEX_MCP,
    THUMB_TIP,
    Finger,
    interior_angle_deg,
    palm_size,
)

#: Degrees 0-4 are the right hand, 5-9 the left hand.
RIGHT_HAND_OFFSET = 0
LEFT_HAND_OFFSET = 5


@dataclass(frozen=True)
class FingerThresholds:
    """Every threshold is tunable without touching code (FR-3.6)."""

    extend_deg: float = 160.0
    curl_deg: float = 140.0
    thumb_extend_deg: float = 150.0
    thumb_curl_deg: float = 130.0
    #: dist(thumb tip, index knuckle) / palm size - the thumb also has to move away
    thumb_abduction_on: float = 0.60
    thumb_abduction_off: float = 0.50
    #: Debounce is measured in milliseconds, not frames: the detection rate
    #: swings with scene complexity, so a frame count would mean 60 ms one
    #: moment and 200 ms the next (FR-3.4).
    on_debounce_ms: float = 0.0
    off_debounce_ms: float = 60.0
    hand_lost_ms: float = 150.0

    def __post_init__(self) -> None:
        if self.curl_deg > self.extend_deg:
            raise ValueError("curl_deg must be <= extend_deg")
        if self.thumb_curl_deg > self.thumb_extend_deg:
            raise ValueError("thumb_curl_deg must be <= thumb_extend_deg")
        if self.thumb_abduction_off > self.thumb_abduction_on:
            raise ValueError("thumb_abduction_off must be <= thumb_abduction_on")
        if min(self.on_debounce_ms, self.off_debounce_ms) < 0.0:
            raise ValueError("debounce times must be >= 0")


def hand_offset(handedness: str, swap_hands: bool = False) -> int:
    """First degree of a hand's block: right hand 0, left hand 5 (FR-2.2)."""
    if swap_hands:
        handedness = "Left" if handedness == "Right" else "Right"
    return LEFT_HAND_OFFSET if handedness == "Left" else RIGHT_HAND_OFFSET


@dataclass(frozen=True)
class NoteEvent:
    """An edge, not a level: degree 0-9 turning on or off (FR-3.5)."""

    degree: int
    on: bool


@dataclass(frozen=True)
class HandObservation:
    """One tracked hand for a single frame."""

    handedness: str  # "Left" or "Right", as reported by MediaPipe
    score: float
    landmarks_norm: np.ndarray  # (21, 3) in 0..1 image coordinates, for drawing
    landmarks_px: np.ndarray  # (21, 3) aspect-corrected, for geometry


def finger_angle(points: np.ndarray, finger: Finger) -> float:
    a, b, c = FINGER_JOINTS[finger]
    return interior_angle_deg(points, a, b, c)


def thumb_abduction(points: np.ndarray) -> float:
    """How far the thumb tip sits from the index knuckle, in palm widths."""
    size = palm_size(points)
    if size == 0.0:
        return 0.0
    return float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_MCP])) / size


@dataclass
class HandFingers:
    """Debounced extension state for the five fingers of one hand."""

    thresholds: FingerThresholds = field(default_factory=FingerThresholds)
    extended: list[bool] = field(default_factory=lambda: [False] * len(Finger))
    #: when each finger first started disagreeing with its committed state
    _pending_since: list[float | None] = field(
        default_factory=lambda: [None] * len(Finger)
    )
    _last_seen_ms: float | None = None

    def raw_extended(self, points: np.ndarray, finger: Finger) -> bool:
        """Hysteresis test: which state this frame argues for (FR-3.2)."""
        t = self.thresholds
        angle = finger_angle(points, finger)
        currently = self.extended[finger]

        if finger is Finger.THUMB:
            abduction = thumb_abduction(points)
            if currently:
                straight = angle > t.thumb_curl_deg
                away = abduction > t.thumb_abduction_off
            else:
                straight = angle > t.thumb_extend_deg
                away = abduction > t.thumb_abduction_on
            return straight and away

        if currently:
            return angle > t.curl_deg
        return angle > t.extend_deg

    def update(self, observation: HandObservation, now_ms: float) -> list[tuple[Finger, bool]]:
        """Feed one frame. Returns the fingers whose committed state changed."""
        # the finger could have moved any time since the previous observation, so
        # the debounce window starts there rather than at this frame
        previous_ms = self._last_seen_ms
        self._last_seen_ms = now_ms
        points = observation.landmarks_px
        changes: list[tuple[Finger, bool]] = []
        t = self.thresholds

        for finger in Finger:
            candidate = self.raw_extended(points, finger)
            if candidate == self.extended[finger]:
                self._pending_since[finger] = None
                continue
            since = self._pending_since[finger]
            if since is None:
                since = self._pending_since[finger] = (
                    previous_ms if previous_ms is not None else now_ms
                )
            needed = t.on_debounce_ms if candidate else t.off_debounce_ms
            if now_ms - since >= needed:
                self.extended[finger] = candidate
                self._pending_since[finger] = None
                changes.append((finger, candidate))
        return changes

    def is_lost(self, now_ms: float) -> bool:
        if self._last_seen_ms is None:
            return True
        return now_ms - self._last_seen_ms > self.thresholds.hand_lost_ms

    def release_all(self) -> list[tuple[Finger, bool]]:
        """Drop every held finger - used when the hand disappears (FR-2.4)."""
        changes = [(f, False) for f in Finger if self.extended[f]]
        self.extended = [False] * len(Finger)
        self._pending_since = [None] * len(Finger)
        self._last_seen_ms = None
        return changes

    def mark_absent(self) -> None:
        self._last_seen_ms = None


class GestureMapper:
    """Turns per-frame hand observations into degree on/off events.

    Right hand -> degrees 0-4, left hand -> degrees 5-9 (thumb first).
    """

    def __init__(
        self,
        thresholds: FingerThresholds | None = None,
        swap_hands: bool = False,
    ) -> None:
        self.thresholds = thresholds or FingerThresholds()
        self.swap_hands = swap_hands
        self.hands: dict[str, HandFingers] = {
            "Right": HandFingers(self.thresholds),
            "Left": HandFingers(self.thresholds),
        }

    def set_thresholds(self, thresholds: FingerThresholds) -> None:
        self.thresholds = thresholds
        for hand in self.hands.values():
            hand.thresholds = thresholds

    def offset_for(self, handedness: str) -> int:
        return hand_offset(handedness, self.swap_hands)

    def update(
        self, observations: list[HandObservation], now_ms: float
    ) -> list[NoteEvent]:
        best = self._best_per_hand(observations)
        events: list[NoteEvent] = []

        for handedness, state in self.hands.items():
            observation = best.get(handedness)
            offset = self.offset_for(handedness)
            if observation is not None:
                changes = state.update(observation, now_ms)
            elif state.is_lost(now_ms):
                changes = state.release_all()
            else:
                changes = []  # still inside the grace period
            events.extend(NoteEvent(offset + int(f), on) for f, on in changes)
        return events

    def release_all(self) -> list[NoteEvent]:
        """Panic path: every held degree turns off."""
        events: list[NoteEvent] = []
        for handedness, state in self.hands.items():
            offset = self.offset_for(handedness)
            events.extend(
                NoteEvent(offset + int(f), on) for f, on in state.release_all()
            )
        return events

    def active_degrees(self) -> set[int]:
        degrees: set[int] = set()
        for handedness, state in self.hands.items():
            offset = self.offset_for(handedness)
            degrees.update(offset + int(f) for f in Finger if state.extended[f])
        return degrees

    @staticmethod
    def _best_per_hand(
        observations: list[HandObservation],
    ) -> dict[str, HandObservation]:
        """Keep the highest-scoring observation per handedness (FR-2.3)."""
        best: dict[str, HandObservation] = {}
        for observation in observations:
            current = best.get(observation.handedness)
            if current is None or observation.score > current.score:
                best[observation.handedness] = observation
        return best
