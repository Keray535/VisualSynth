"""Finger extension state machine and degree mapping (FR-2, FR-3).

Everything runs on synthetic landmarks - no camera, no MediaPipe.
"""

from __future__ import annotations

import numpy as np
import pytest

from hand_fixtures import fist, make_hand, open_hand
from visualsynth.vision.finger_state import (
    FingerThresholds,
    GestureMapper,
    HandFingers,
    NoteEvent,
    finger_angle,
    thumb_abduction,
)
from visualsynth.vision.landmarks import (
    Finger,
    HAND_CONNECTIONS,
    LANDMARK_COUNT,
    interior_angle_deg,
    palm_size,
    to_isotropic,
)

FRAME = 1000.0 / 30.0  # one frame at 30 FPS, in ms


# ---------- geometry ----------


def test_interior_angle_basics():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert interior_angle_deg(points, 0, 1, 2) == pytest.approx(90.0)

    straight = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert interior_angle_deg(straight, 0, 1, 2) == pytest.approx(180.0)


def test_degenerate_angle_is_zero_not_nan():
    points = np.zeros((3, 3))
    assert interior_angle_deg(points, 0, 1, 2) == 0.0


def test_aspect_correction_scales_axes_independently():
    normalized = np.array([[0.5, 0.5, 0.1]])
    scaled = to_isotropic(normalized, 640, 480)
    assert scaled[0].tolist() == pytest.approx([320.0, 240.0, 64.0])


def test_hand_connections_are_valid_landmark_pairs():
    assert len(HAND_CONNECTIONS) == 21
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a < LANDMARK_COUNT and 0 <= b < LANDMARK_COUNT


def test_open_hand_reads_as_straight_fingers():
    points = open_hand().landmarks_px
    for finger in Finger:
        assert finger_angle(points, finger) > 170.0, finger.name
    assert palm_size(points) > 0.0


def test_fist_reads_as_curled_fingers():
    points = fist().landmarks_px
    for finger in Finger:
        assert finger_angle(points, finger) < 100.0, finger.name


def test_thumb_abduction_drops_when_the_thumb_tucks_in():
    out = thumb_abduction(make_hand(thumb_abduction=1.0).landmarks_px)
    tucked = thumb_abduction(make_hand(thumb_abduction=0.15).landmarks_px)
    assert tucked < out


# ---------- per-hand state machine ----------


def test_notes_start_at_once_and_release_after_the_debounce_window():
    """FR-3.4: fast note-on, slow note-off, both measured in milliseconds."""
    state = HandFingers(FingerThresholds(on_debounce_ms=0.0, off_debounce_ms=60.0))
    changes = state.update(open_hand(), 0.0)
    assert sorted(f for f, on in changes if on) == list(Finger)
    assert state.extended == [True] * 5

    # one curled frame, 33 ms later, is inside the release window
    assert state.update(fist(), FRAME) == []
    assert state.extended == [True] * 5

    changes = state.update(fist(), 2 * FRAME)  # 66 ms of curl
    assert all(on is False for _, on in changes)
    assert state.extended == [False] * 5


def test_release_window_is_frame_rate_independent():
    """The same 60 ms holds whether detection runs at 10 or 30 FPS."""
    thresholds = FingerThresholds(off_debounce_ms=60.0)
    slow = HandFingers(thresholds)
    slow.update(open_hand(), 0.0)
    assert slow.update(fist(), 100.0) != []  # one frame at 10 FPS is already 100 ms

    fast = HandFingers(thresholds)
    fast.update(open_hand(), 0.0)
    assert fast.update(fist(), 20.0) == []  # 20 ms later: still held
    assert fast.update(fist(), 70.0) != []


def test_single_dropout_frame_does_not_cut_a_held_note():
    state = HandFingers(FingerThresholds(off_debounce_ms=60.0))
    state.update(open_hand(), 0.0)
    state.update(fist(), FRAME)  # glitch frame
    state.update(open_hand(), 2 * FRAME)  # recovered: the window restarts
    state.update(fist(), 3 * FRAME)  # first genuinely curled frame
    assert state.extended == [True] * 5


def test_hysteresis_dead_band_holds_the_current_state():
    """FR-3.2: extend above 160, release below 140, hold in between."""
    thresholds = FingerThresholds()
    state = HandFingers(thresholds)
    index = Finger.INDEX

    mid = make_hand(angles={index: 150.0})
    assert state.raw_extended(mid.landmarks_px, index) is False  # needs > 160 to turn on

    state.extended[index] = True
    assert state.raw_extended(mid.landmarks_px, index) is True  # needs < 140 to turn off

    low = make_hand(angles={index: 135.0})
    assert state.raw_extended(low.landmarks_px, index) is False


def test_thumb_needs_both_a_straight_angle_and_abduction():
    """FR-3.3."""
    state = HandFingers()
    straight_but_tucked = make_hand(
        angles={Finger.THUMB: 175.0}, thumb_abduction=0.05
    )
    assert state.raw_extended(straight_but_tucked.landmarks_px, Finger.THUMB) is False

    straight_and_out = make_hand(angles={Finger.THUMB: 175.0}, thumb_abduction=1.0)
    assert state.raw_extended(straight_and_out.landmarks_px, Finger.THUMB) is True


def test_release_all_clears_held_fingers():
    state = HandFingers()
    state.update(open_hand(), 0.0)
    changes = state.release_all()
    assert len(changes) == 5
    assert state.extended == [False] * 5
    assert state.release_all() == []


def test_is_lost_respects_the_grace_period():
    state = HandFingers(FingerThresholds(hand_lost_ms=150.0))
    assert state.is_lost(0.0) is True  # never seen
    state.update(open_hand(), 1000.0)
    assert state.is_lost(1100.0) is False
    assert state.is_lost(1200.0) is True


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        FingerThresholds(extend_deg=100.0, curl_deg=120.0)
    with pytest.raises(ValueError):
        FingerThresholds(thumb_abduction_on=0.2, thumb_abduction_off=0.5)
    with pytest.raises(ValueError):
        FingerThresholds(off_debounce_ms=-1.0)


# ---------- degree mapping ----------


def degrees_on(events: list[NoteEvent]) -> list[int]:
    return sorted(e.degree for e in events if e.on)


def test_right_hand_maps_to_degrees_one_to_five():
    mapper = GestureMapper()
    events = mapper.update([open_hand("Right")], 0.0)
    assert degrees_on(events) == [0, 1, 2, 3, 4]
    assert mapper.active_degrees() == {0, 1, 2, 3, 4}


def test_left_hand_maps_to_degrees_six_to_ten():
    mapper = GestureMapper()
    events = mapper.update([open_hand("Left")], 0.0)
    assert degrees_on(events) == [5, 6, 7, 8, 9]


def test_both_hands_cover_all_ten_degrees():
    mapper = GestureMapper()
    events = mapper.update([open_hand("Right"), open_hand("Left")], 0.0)
    assert degrees_on(events) == list(range(10))


def test_thumb_is_the_first_degree_of_each_hand():
    mapper = GestureMapper()
    right_thumb = make_hand("Right", extended={Finger.THUMB: True})
    left_pinky = make_hand("Left", extended={Finger.PINKY: True})
    events = mapper.update([right_thumb, left_pinky], 0.0)
    assert degrees_on(events) == [0, 9]


def test_swap_hands_inverts_the_blocks():
    mapper = GestureMapper(swap_hands=True)
    events = mapper.update([open_hand("Right")], 0.0)
    assert degrees_on(events) == [5, 6, 7, 8, 9]


def test_higher_scoring_hand_wins_a_duplicate_handedness():
    """FR-2.3."""
    mapper = GestureMapper()
    weak = make_hand("Right", extended=dict.fromkeys(Finger, False), score=0.4)
    strong = open_hand("Right", score=0.9)
    events = mapper.update([weak, strong], 0.0)
    assert degrees_on(events) == [0, 1, 2, 3, 4]


def test_lost_hand_releases_its_notes_after_the_grace_period():
    """FR-2.4: no stuck notes."""
    mapper = GestureMapper(FingerThresholds(hand_lost_ms=150.0))
    mapper.update([open_hand("Right")], 1000.0)

    assert mapper.update([], 1100.0) == []  # inside the grace period
    events = mapper.update([], 1200.0)
    assert sorted(e.degree for e in events if not e.on) == [0, 1, 2, 3, 4]
    assert mapper.active_degrees() == set()


def test_one_hand_leaving_does_not_disturb_the_other():
    mapper = GestureMapper(FingerThresholds(hand_lost_ms=100.0))
    mapper.update([open_hand("Right"), open_hand("Left")], 0.0)
    events = mapper.update([open_hand("Left")], 200.0)
    assert sorted(e.degree for e in events if not e.on) == [0, 1, 2, 3, 4]
    assert mapper.active_degrees() == {5, 6, 7, 8, 9}


def test_release_all_reports_every_held_degree():
    mapper = GestureMapper()
    mapper.update([open_hand("Right"), open_hand("Left")], 0.0)
    events = mapper.release_all()
    assert sorted(e.degree for e in events) == list(range(10))
    assert all(e.on is False for e in events)
    assert mapper.active_degrees() == set()


def test_no_events_when_nothing_changes():
    mapper = GestureMapper()
    mapper.update([open_hand("Right")], 0.0)
    assert mapper.update([open_hand("Right")], FRAME) == []


def test_individual_fingers_toggle_independently():
    mapper = GestureMapper(FingerThresholds(off_debounce_ms=0.0))
    mapper.update([fist("Right")], 0.0)

    one = make_hand("Right", extended={Finger.MIDDLE: True})
    assert degrees_on(mapper.update([one], FRAME)) == [2]

    two = make_hand("Right", extended={Finger.MIDDLE: True, Finger.PINKY: True})
    assert degrees_on(mapper.update([two], 2 * FRAME)) == [4]
    assert mapper.active_degrees() == {2, 4}
