"""Parallel hand tracking.

MediaPipe's CPU inference here is single-threaded and costs roughly 42 ms per
hand, so one tracker caps out near 10 detections/s with two hands in frame.
The MediaPipe call goes through ctypes, which releases the GIL, so several
tracker instances in Python threads really do run in parallel: measured 8.6/s
with one worker, 17.2/s with two, 22.2/s with three.

Each worker owns its own `HandTracker` and takes frames round-robin. Results can
finish out of order, so callers drop any result older than the newest one they
have already applied.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from .finger_state import HandObservation
from .hand_tracker import HandTracker

log = logging.getLogger(__name__)

DEFAULT_WORKERS = 3
_SHUTDOWN = object()


@dataclass
class TrackerResult:
    timestamp_ms: int
    frame_bgr: np.ndarray
    observations: list[HandObservation] = field(default_factory=list)
    inference_ms: float = 0.0


class TrackerPool:
    """A small pool of hand trackers fed frame by frame."""

    def __init__(self, workers: int = DEFAULT_WORKERS, **tracker_kwargs) -> None:
        self.workers = max(1, int(workers))
        self._tracker_kwargs = tracker_kwargs
        self._jobs: queue.Queue = queue.Queue(maxsize=self.workers)
        self._results: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._stopping = threading.Event()
        self._pending = 0
        self._lock = threading.Lock()

    # ---- lifecycle ----

    def start(self, timeout: float = 30.0) -> None:
        """Spin up the workers and wait for the first tracker to load its model."""
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._run, args=(index,), name=f"tracker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        if not self._ready.wait(timeout):
            raise TimeoutError("hand trackers did not start in time")
        if self._start_error is not None:
            raise self._start_error

    def close(self) -> None:
        self._stopping.set()
        for _ in self._threads:
            try:
                self._jobs.put_nowait(_SHUTDOWN)
            except queue.Full:
                pass
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()

    def __enter__(self) -> "TrackerPool":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- producer side ----

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending >= self.workers

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int) -> bool:
        """Queue a frame. Returns False when every worker is already busy."""
        try:
            self._jobs.put_nowait((frame_bgr, timestamp_ms))
        except queue.Full:
            return False
        with self._lock:
            self._pending += 1
        return True

    def poll(self) -> list[TrackerResult]:
        """Every result finished so far, oldest first. Never blocks."""
        results: list[TrackerResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        results.sort(key=lambda r: r.timestamp_ms)
        return results

    # ---- worker side ----

    def _run(self, index: int) -> None:
        try:
            tracker = HandTracker(**self._tracker_kwargs)
        except Exception as exc:  # noqa: BLE001 - reported through start()
            log.exception("tracker %s failed to load", index)
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()

        try:
            while not self._stopping.is_set():
                job = self._jobs.get()
                if job is _SHUTDOWN:
                    break
                frame, timestamp_ms = job
                try:
                    observations = tracker.process(frame, timestamp_ms)
                    inference_ms = tracker.last_inference_ms
                except Exception:  # noqa: BLE001 - one bad frame must not kill audio
                    log.exception("inference failed on tracker %s", index)
                    observations, inference_ms = [], 0.0
                finally:
                    with self._lock:
                        self._pending -= 1
                self._results.put(
                    TrackerResult(
                        timestamp_ms=timestamp_ms,
                        frame_bgr=frame,
                        observations=observations,
                        inference_ms=inference_ms,
                    )
                )
        finally:
            tracker.close()
