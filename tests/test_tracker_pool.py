"""Parallel tracker pool, with a stubbed tracker - no model, no camera."""

from __future__ import annotations

import time

import numpy as np
import pytest

from visualsynth.vision import tracker_pool
from visualsynth.vision.finger_state import HandObservation
from visualsynth.vision.tracker_pool import TrackerPool

WORK_S = 0.05


class FakeTracker:
    """Stands in for MediaPipe: sleeps like inference does, then reports a hand."""

    instances = 0
    fail_next = False

    def __init__(self, **_kwargs) -> None:
        type(self).instances += 1
        self.closed = False
        self.last_inference_ms = 0.0

    def process(self, frame, timestamp_ms: int) -> list[HandObservation]:
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("inference blew up")
        time.sleep(WORK_S)
        self.last_inference_ms = WORK_S * 1000
        return [
            HandObservation(
                handedness="Right",
                score=0.9,
                landmarks_norm=np.zeros((21, 3)),
                landmarks_px=np.zeros((21, 3)),
            )
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def stub_tracker(monkeypatch):
    FakeTracker.instances = 0
    FakeTracker.fail_next = False
    monkeypatch.setattr(tracker_pool, "HandTracker", FakeTracker)


def frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def drain(pool: TrackerPool, expected: int, timeout: float = 5.0) -> list:
    results: list = []
    deadline = time.perf_counter() + timeout
    while len(results) < expected and time.perf_counter() < deadline:
        results.extend(pool.poll())
        time.sleep(0.005)
    return results


def test_pool_starts_one_tracker_per_worker():
    with TrackerPool(workers=3) as pool:
        assert pool.workers == 3
        deadline = time.perf_counter() + 2.0
        while FakeTracker.instances < 3 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert FakeTracker.instances == 3


def test_results_come_back_with_their_frame_and_timestamp():
    with TrackerPool(workers=1) as pool:
        assert pool.submit(frame(), 1000) is True
        result = drain(pool, 1)[0]
        assert result.timestamp_ms == 1000
        assert len(result.observations) == 1
        assert result.inference_ms == pytest.approx(WORK_S * 1000, rel=0.5)


def test_submit_is_refused_when_every_worker_is_busy():
    with TrackerPool(workers=1) as pool:
        assert pool.submit(frame(), 1) is True
        # the single worker is now sleeping through its "inference"
        assert pool.submit(frame(), 2) is False
        assert pool.busy is True
        drain(pool, 1)


def test_workers_run_in_parallel():
    """Three frames on three workers take about one inference, not three."""
    with TrackerPool(workers=3) as pool:
        drain(pool, 0, timeout=0.3)  # let the trackers come up
        started = time.perf_counter()
        for ts in (1, 2, 3):
            assert pool.submit(frame(), ts) is True
        results = drain(pool, 3)
        elapsed = time.perf_counter() - started
        assert len(results) == 3
        assert elapsed < WORK_S * 2.5


def test_poll_returns_results_oldest_first():
    with TrackerPool(workers=3) as pool:
        drain(pool, 0, timeout=0.3)
        for ts in (30, 10, 20):
            pool.submit(frame(), ts)
        results = drain(pool, 3)
        assert [r.timestamp_ms for r in results] == sorted(r.timestamp_ms for r in results)


def test_poll_is_empty_when_nothing_is_queued():
    with TrackerPool(workers=1) as pool:
        assert pool.poll() == []


def test_a_failing_frame_does_not_stop_the_worker():
    with TrackerPool(workers=1) as pool:
        drain(pool, 0, timeout=0.3)
        FakeTracker.fail_next = True
        pool.submit(frame(), 1)
        first = drain(pool, 1)[0]
        assert first.observations == []

        pool.submit(frame(), 2)
        second = drain(pool, 1)[0]
        assert len(second.observations) == 1


def test_close_is_idempotent_and_releases_the_trackers():
    pool = TrackerPool(workers=2)
    pool.start()
    pool.close()
    pool.close()
    assert pool.poll() == []


def test_start_failure_is_reported():
    class Broken(FakeTracker):
        def __init__(self, **_kwargs):
            raise RuntimeError("no model here")

    import visualsynth.vision.tracker_pool as module

    module.HandTracker = Broken
    pool = TrackerPool(workers=1)
    with pytest.raises(RuntimeError, match="no model here"):
        pool.start()
    pool.close()
