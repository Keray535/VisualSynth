"""Vision worker logic and audio device selection, without camera or stream."""

from __future__ import annotations

import time

import numpy as np
import pytest

from visualsynth.audio.engine import (
    OutputDevice,
    list_output_devices,
    preferred_output_device,
    resolve_output_config,
)
from visualsynth.config import AppConfig
from visualsynth.vision.finger_state import FingerThresholds, NoteEvent
from visualsynth.vision.tracker_pool import TrackerResult
from visualsynth.vision.worker import RateMeter, VisionWorker

from hand_fixtures import fist, open_hand


# ---------- rate meter ----------


def test_rate_meter_starts_at_zero():
    assert RateMeter().rate == 0.0


def test_rate_meter_counts_over_the_window():
    meter = RateMeter(window_s=10.0)
    for _ in range(5):
        meter.tick()
        time.sleep(0.01)
    assert meter.rate > 0.0


def test_rate_meter_does_not_spike_on_a_burst():
    """Results arrive in bursts; the reported rate must stay sane."""
    meter = RateMeter(window_s=1.0)
    meter.tick()
    time.sleep(0.2)
    for _ in range(3):  # three results delivered back to back
        meter.tick()
    assert meter.rate < 100.0


def test_rate_meter_forgets_old_events():
    meter = RateMeter(window_s=0.05)
    meter.tick()
    time.sleep(0.12)
    meter.tick()
    assert meter.rate <= 20.0


# ---------- worker logic ----------


@pytest.fixture
def worker() -> VisionWorker:
    events: list[NoteEvent] = []
    config = AppConfig(
        thresholds=FingerThresholds(hand_lost_ms=150.0, off_debounce_ms=0.0),
        idle_inference_fps=10.0,
    )
    w = VisionWorker(config, events.extend)
    w.collected = events  # type: ignore[attr-defined]
    return w


def result_for(observations, timestamp_ms: int) -> TrackerResult:
    return TrackerResult(
        timestamp_ms=timestamp_ms,
        frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        observations=observations,
        inference_ms=12.0,
    )


def test_apply_routes_note_events_to_the_callback(worker):
    worker._apply(result_for([open_hand("Right")], 1000))
    assert sorted(e.degree for e in worker.collected if e.on) == [0, 1, 2, 3, 4]


def test_apply_reports_the_active_degrees_to_the_ui(worker):
    seen = []
    worker.frameReady.connect(seen.append)
    worker._apply(result_for([open_hand("Left")], 1000))
    assert seen and sorted(seen[-1].active_degrees) == [5, 6, 7, 8, 9]
    assert seen[-1].inference_ms == 12.0


def test_curling_the_fingers_releases_the_notes(worker):
    worker._apply(result_for([open_hand("Right")], 1000))
    worker.collected.clear()
    worker._apply(result_for([fist("Right")], 1100))
    assert sorted(e.degree for e in worker.collected if not e.on) == [0, 1, 2, 3, 4]


def test_stalled_detection_releases_held_notes(worker):
    """NFR-3.1: a wedged tracker or stalled camera must not hang a note."""
    worker._apply(result_for([open_hand("Right")], 1000))
    worker.collected.clear()

    worker._age_hands(now_ms=1100, last_applied_ms=1000)  # inside the grace period
    assert worker.collected == []

    worker._age_hands(now_ms=1400, last_applied_ms=1000)
    assert sorted(e.degree for e in worker.collected if not e.on) == [0, 1, 2, 3, 4]


def test_age_hands_does_nothing_before_the_first_result(worker):
    worker._age_hands(now_ms=5000, last_applied_ms=-1)
    assert worker.collected == []


def test_submission_is_throttled_only_while_no_hand_is_visible(worker):
    assert worker._should_submit(0) is True  # nothing submitted yet
    worker._last_submit_ms = 0.0
    worker._hands_present = False
    assert worker._should_submit(50) is False  # 10 fps idle rate -> 100 ms apart
    assert worker._should_submit(150) is True

    worker._hands_present = True
    assert worker._should_submit(50) is True  # full rate while playing


def test_timestamps_strictly_increase(worker):
    stamps = [worker._timestamp_ms() for _ in range(5)]
    assert stamps == sorted(set(stamps))
    assert len(stamps) == len(set(stamps))


def test_release_notes_turns_everything_off(worker):
    worker._apply(result_for([open_hand("Right"), open_hand("Left")], 1000))
    worker.collected.clear()
    worker._release_notes()
    assert sorted(e.degree for e in worker.collected) == list(range(10))
    assert all(not e.on for e in worker.collected)


def test_swap_hands_moves_the_blocks(worker):
    worker.set_swap_hands(True)
    worker._apply(result_for([open_hand("Right")], 1000))
    assert sorted(e.degree for e in worker.collected if e.on) == [5, 6, 7, 8, 9]


# ---------- audio device selection ----------


def test_output_devices_are_listed_with_their_host_api():
    devices = list_output_devices()
    assert devices, "no audio output devices on this machine"
    assert all(isinstance(d, OutputDevice) for d in devices)
    assert all(d.max_channels >= 1 for d in devices)
    assert all(d.host_api in d.label for d in devices)


def test_preferred_device_avoids_the_slow_host_api_when_possible():
    """NFR-1.2: MME reports ~100 ms of output latency, WASAPI ~20 ms."""
    devices = list_output_devices()
    chosen = preferred_output_device()
    assert chosen is not None
    if any(d.host_api == "Windows WASAPI" for d in devices):
        assert chosen.host_api == "Windows WASAPI"


def test_resolve_output_config_follows_the_device_rate():
    index, rate = resolve_output_config()
    assert rate in (44_100, 48_000, 88_200, 96_000) or rate > 8_000
    if index is not None:
        device = next(d for d in list_output_devices() if d.index == index)
        assert rate == int(round(device.default_samplerate))


def test_resolve_output_config_honours_an_explicit_device():
    devices = list_output_devices()
    target = devices[0]
    index, rate = resolve_output_config(target.index)
    assert index == target.index
    assert rate == int(round(target.default_samplerate))
