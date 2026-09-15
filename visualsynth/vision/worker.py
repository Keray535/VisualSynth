"""Vision thread: capture -> track -> finger state -> note events.

Runs off the UI thread (NFR-2.3). Note events are handed straight to the synth's
queue from this thread - that is the shortest path to sound - while frames and
status go to the UI as Qt signals.

Detection happens in a `TrackerPool`, because a single MediaPipe tracker only
manages ~10 detections/s with two hands in frame on this hardware.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from ..config import AppConfig
from .camera import Camera, CameraError
from .finger_state import FingerThresholds, GestureMapper, HandObservation, NoteEvent
from .hand_tracker import ModelMissingError
from .tracker_pool import TrackerPool, TrackerResult

log = logging.getLogger(__name__)

RETRY_DELAY_S = 1.0


class RateMeter:
    """Events per second over a sliding window.

    Results arrive from the pool in bursts, so a rate derived from the gap
    between two consecutive events reads wildly high; counting over a window
    does not.
    """

    def __init__(self, window_s: float = 1.5) -> None:
        self._window_s = window_s
        self._times: deque[float] = deque()

    def tick(self) -> None:
        now = time.perf_counter()
        self._times.append(now)
        while self._times and now - self._times[0] > self._window_s:
            self._times.popleft()

    @property
    def rate(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0.0 else 0.0


@dataclass
class FrameUpdate:
    """Everything the UI needs to paint one frame."""

    frame_bgr: np.ndarray
    observations: list[HandObservation] = field(default_factory=list)
    fps: float = 0.0
    inference_ms: float = 0.0
    inference_fps: float = 0.0
    active_degrees: frozenset[int] = frozenset()


class VisionWorker(QThread):
    frameReady = Signal(object)  # FrameUpdate
    failed = Signal(str)  # human-readable cause, shown by the UI (UX-6.1)
    recovered = Signal()

    def __init__(
        self,
        config: AppConfig,
        on_events: Callable[[list[NoteEvent]], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._on_events = on_events
        self._mutex = QMutex()
        self._stopping = False
        self._camera_index = config.camera_index
        self._mirror = config.mirror
        self._pending_camera: int | None = None
        self._workers = config.tracker_workers
        self.mapper = GestureMapper(config.thresholds, swap_hands=config.swap_hands)

        self._idle_interval_ms = 1000.0 / max(1.0, config.idle_inference_fps)
        self._min_frame_interval = 1.0 / max(1.0, config.preview_fps)
        self._capture_rate = RateMeter()
        self._detect_rate = RateMeter()
        self._inference_ms = 0.0
        self._last_timestamp_ms = -1
        self._last_submit_ms = -1.0
        self._hands_present = False

    # ---- control API (UI thread) ----

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stopping = True
        self.wait(8000)

    def request_camera(self, index: int) -> None:
        with QMutexLocker(self._mutex):
            self._pending_camera = index

    @property
    def camera_index(self) -> int:
        with QMutexLocker(self._mutex):
            return self._camera_index

    def set_swap_hands(self, swap: bool) -> None:
        self.mapper.swap_hands = swap

    def set_thresholds(self, thresholds: FingerThresholds) -> None:
        self.mapper.set_thresholds(thresholds)

    def set_idle_inference_fps(self, fps: float) -> None:
        self._idle_interval_ms = 1000.0 / max(1.0, fps)

    # ---- worker thread ----

    def run(self) -> None:
        try:
            pool = TrackerPool(workers=self._workers)
            pool.start()
        except ModelMissingError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never kill the app from here
            log.exception("hand tracking failed to start")
            self.failed.emit(f"Hand tracking could not start: {exc}")
            return

        camera: Camera | None = None
        announced_failure = False
        last_applied_ms = -1

        try:
            while not self._should_stop():
                loop_started = time.perf_counter()

                pending = self._take_pending_camera()
                if pending is not None and pending != self._camera_index:
                    self._camera_index = pending
                    camera = self._close(camera)

                if camera is None:
                    try:
                        camera = Camera(index=self._camera_index, mirror=self._mirror)
                        camera.open()
                        if announced_failure:
                            announced_failure = False
                            self.recovered.emit()
                    except CameraError as exc:
                        self._release_notes()
                        if not announced_failure:
                            announced_failure = True
                            self.failed.emit(str(exc))
                        self._sleep(RETRY_DELAY_S)
                        continue

                try:
                    frame = camera.read()
                except CameraError as exc:
                    camera = self._close(camera)
                    self._release_notes()
                    self.failed.emit(str(exc))
                    announced_failure = True
                    continue

                now_ms = self._timestamp_ms()
                if frame is not None:
                    self._capture_rate.tick()
                if frame is not None and self._should_submit(now_ms):
                    if pool.submit(frame, now_ms):
                        self._last_submit_ms = now_ms

                for result in pool.poll():
                    if result.timestamp_ms <= last_applied_ms:
                        continue  # a slower worker finished out of order
                    last_applied_ms = result.timestamp_ms
                    self._apply(result)

                self._age_hands(now_ms, last_applied_ms)
                self._pace(loop_started)
        finally:
            self._release_notes()
            self._close(camera)
            pool.close()

    # ---- per-result handling ----

    def _apply(self, result: TrackerResult) -> None:
        self._hands_present = bool(result.observations)
        self._inference_ms = result.inference_ms
        self._detect_rate.tick()
        self._dispatch(self.mapper.update(result.observations, result.timestamp_ms))
        self.frameReady.emit(
            FrameUpdate(
                frame_bgr=result.frame_bgr,
                observations=result.observations,
                fps=self._capture_rate.rate,
                inference_ms=self._inference_ms,
                inference_fps=self._detect_rate.rate,
                active_degrees=frozenset(self.mapper.active_degrees()),
            )
        )

    def _age_hands(self, now_ms: int, last_applied_ms: int) -> None:
        """Release held notes when results stop arriving (FR-2.4).

        A stalled camera or a wedged tracker must not leave a note hanging.
        """
        if last_applied_ms < 0:
            return
        if now_ms - last_applied_ms > self.mapper.thresholds.hand_lost_ms:
            self._dispatch(self.mapper.update([], now_ms))

    # ---- helpers ----

    def _should_stop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stopping

    def _take_pending_camera(self) -> int | None:
        with QMutexLocker(self._mutex):
            pending, self._pending_camera = self._pending_camera, None
            return pending

    def _sleep(self, seconds: float) -> None:
        self.msleep(int(seconds * 1000))

    def _timestamp_ms(self) -> int:
        """MediaPipe VIDEO mode needs strictly increasing timestamps."""
        now = int(time.monotonic() * 1000)
        if now <= self._last_timestamp_ms:
            now = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now
        return now

    def _pace(self, loop_started: float) -> None:
        """Hold the loop to the preview rate.

        A slow detection lets the driver queue frames, which then come back
        instantly and burst the loop well past the camera's own FPS.
        """
        remaining = self._min_frame_interval - (time.perf_counter() - loop_started)
        if remaining > 0.001:
            self.msleep(int(remaining * 1000))

    def _should_submit(self, now_ms: int) -> bool:
        """Full rate while a hand is in frame, throttled while idle.

        Palm detection on an empty scene still costs tens of milliseconds per
        frame and nothing is waiting on it, so idle frames go in slowly (NFR-2.2).
        """
        if self._hands_present or self._last_submit_ms < 0:
            return True
        return now_ms - self._last_submit_ms >= self._idle_interval_ms

    def _dispatch(self, events: list[NoteEvent]) -> None:
        if events:
            self._on_events(events)

    def _release_notes(self) -> None:
        self._dispatch(self.mapper.release_all())

    @staticmethod
    def _close(camera: Camera | None) -> None:
        if camera is not None:
            camera.release()
        return None
