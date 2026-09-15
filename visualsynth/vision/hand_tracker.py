"""MediaPipe hand landmark detection (FR-2).

MediaPipe 1.0 only ships the Tasks API - `mediapipe.solutions` is gone - so this
uses `vision.HandLandmarker` in VIDEO running mode, which expects monotonically
increasing timestamps.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .finger_state import HandObservation
from .landmarks import to_isotropic

log = logging.getLogger(__name__)

MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


class ModelMissingError(FileNotFoundError):
    """The .task model has not been downloaded yet (FR-6.4)."""


class HandTracker:
    """Detects up to two hands per frame and reports them as observations."""

    def __init__(
        self,
        model_path: Path | None = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        if not self.model_path.exists():
            raise ModelMissingError(
                f"Hand model not found at {self.model_path}.\n"
                "Download it with:  python scripts/fetch_models.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._rgb: np.ndarray | None = None
        self.last_inference_ms = 0.0
        log.info("hand tracker ready (%s)", self.model_path.name)

    # ---- lifecycle ----

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- detection ----

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        """Detect hands in a BGR frame. `timestamp_ms` must strictly increase."""
        if self._landmarker is None:
            raise RuntimeError("HandTracker is closed")

        height, width = frame_bgr.shape[:2]
        if self._rgb is None or self._rgb.shape != frame_bgr.shape:
            self._rgb = np.empty_like(frame_bgr)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb)
        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        return self._to_observations(result, width, height)

    @staticmethod
    def _to_observations(result, width: int, height: int) -> list[HandObservation]:
        observations: list[HandObservation] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            if not handedness:
                continue
            category = handedness[0]
            normalized = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64
            )
            observations.append(
                HandObservation(
                    handedness=category.category_name,
                    score=float(category.score),
                    landmarks_norm=normalized,
                    landmarks_px=to_isotropic(normalized, width, height),
                )
            )
        return observations
