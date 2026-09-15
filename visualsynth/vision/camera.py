"""Webcam capture (FR-1).

Thin wrapper over cv2.VideoCapture: device probing, mirrored frames and errors
that the UI can show and retry (FR-1.4).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


class CameraError(RuntimeError):
    """Camera missing, busy or refusing to deliver frames."""


@dataclass(frozen=True)
class CameraDevice:
    index: int
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"Camera {self.index} ({self.width}x{self.height})"


def _backend() -> int:
    # DirectShow opens far faster than MSMF on Windows; elsewhere let OpenCV pick.
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def list_cameras(
    max_index: int = 4, skip: Iterable[int] = ()
) -> list[CameraDevice]:
    """Probe device indices. OpenCV has no enumeration API, so this opens each
    index briefly - call it off the UI thread.

    `skip` leaves out indices that are already open elsewhere: on Windows a
    second open of a live camera fails and can disturb the running capture.
    """
    skipped = set(skip)
    found: list[CameraDevice] = []
    for index in range(max_index + 1):
        if index in skipped:
            continue
        capture = cv2.VideoCapture(index, _backend())
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            found.append(
                CameraDevice(index=index, width=frame.shape[1], height=frame.shape[0])
            )
        finally:
            capture.release()
    return found


class Camera:
    """One open capture device. Frames come back mirrored (FR-1.3)."""

    def __init__(
        self,
        index: int = 0,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        mirror: bool = True,
    ) -> None:
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.mirror = mirror
        self._capture: cv2.VideoCapture | None = None

    # ---- lifecycle ----

    def open(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(self.index, _backend())
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Camera {self.index} could not be opened. "
                "It may be missing, in use by another app, or blocked by Windows "
                "privacy settings (Settings > Privacy & security > Camera)."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # a small driver buffer keeps latency down (NFR-1.1)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, _ = capture.read()
        if not ok:
            capture.release()
            raise CameraError(f"Camera {self.index} opened but delivered no frames.")

        self._capture = capture
        log.info(
            "camera %s opened: %sx%s @ %s fps",
            self.index,
            self.width,
            self.height,
            self.fps,
        )

    def release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def reopen(self, index: int | None = None) -> None:
        """Switch device without tearing the app down (FR-1.2)."""
        self.release()
        if index is not None:
            self.index = index
        self.open()

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    # ---- properties ----

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def width(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._capture else 0

    @property
    def height(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._capture else 0

    @property
    def fps(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_FPS)) if self._capture else 0.0

    # ---- capture ----

    def read(self) -> np.ndarray | None:
        """Next BGR frame, mirrored. None means a dropped frame, not a failure."""
        if self._capture is None:
            raise CameraError("Camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame
