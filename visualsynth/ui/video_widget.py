"""Mirrored camera preview with the hand skeleton and fingertip badges (UX-2, UX-6)."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..music.scales import DEGREE_COUNT
from ..vision.finger_state import HandObservation, hand_offset
from ..vision.landmarks import FINGER_TIPS, HAND_CONNECTIONS, Finger
from ..vision.worker import FrameUpdate
from . import theme
from .note_strip import CIRCLED

NO_HAND_HINT_S = 2.0
LOW_CONFIDENCE = 0.6


class VideoWidget(QWidget):
    """Draws the latest frame, letterboxed, with overlays on top."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: QImage | None = None
        self._observations: list[HandObservation] = []
        self._active: frozenset[int] = frozenset()
        self._names: list[str] = ["-"] * DEGREE_COUNT
        self._swap_hands = False
        self._error: str | None = None
        self._hint: str | None = None
        self._last_hand_time = time.monotonic()

    # ---- state ----

    @property
    def error(self) -> str | None:
        return self._error

    def set_frame(self, update: FrameUpdate) -> None:
        self._image = _to_qimage(update.frame_bgr)
        self._observations = update.observations
        self._active = update.active_degrees
        if update.observations:
            self._last_hand_time = time.monotonic()
        self.update()

    def set_note_names(self, names: list[str]) -> None:
        self._names = list(names)
        self.update()

    def set_swap_hands(self, swap: bool) -> None:
        self._swap_hands = swap
        self.update()

    def set_error(self, message: str | None) -> None:
        self._error = message
        if message is not None:
            self._image = None
            self._observations = []
        self.update()

    def set_hint(self, message: str | None) -> None:
        self._hint = message
        self.update()

    # ---- painting ----

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.BACKGROUND)

        if self._error is not None:
            self._paint_card(painter, "Camera unavailable", self._error, theme.ERROR)
            painter.end()
            return

        if self._image is None:
            self._paint_card(painter, "Starting camera...", "", theme.TEXT_DIM)
            painter.end()
            return

        target = self._image_rect()
        painter.drawImage(target, self._image)

        for observation in self._observations:
            self._paint_hand(painter, observation, target)

        self._paint_hints(painter, target)
        painter.end()

    # ---- helpers ----

    def _image_rect(self) -> QRectF:
        """Letterbox the frame inside the widget, preserving aspect (UX-2.1)."""
        assert self._image is not None
        iw, ih = self._image.width(), self._image.height()
        if iw == 0 or ih == 0:
            return QRectF(self.rect())
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_widget(self, landmark: np.ndarray, rect: QRectF) -> QPointF:
        return QPointF(
            rect.x() + float(landmark[0]) * rect.width(),
            rect.y() + float(landmark[1]) * rect.height(),
        )

    def _paint_hand(
        self, painter: QPainter, observation: HandObservation, rect: QRectF
    ) -> None:
        offset = hand_offset(observation.handedness, self._swap_hands)
        accent = theme.LEFT_ACCENT if offset else theme.RIGHT_ACCENT
        points = [self._to_widget(lm, rect) for lm in observation.landmarks_norm]

        active_tips = {
            FINGER_TIPS[f] for f in Finger if (offset + int(f)) in self._active
        }

        bone = QColor(accent)
        bone.setAlpha(150)
        for a, b in HAND_CONNECTIONS:
            lit = a in active_tips or b in active_tips
            pen = QPen(accent if lit else bone)
            pen.setWidthF(4.0 if lit else 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(points[a], points[b])

        joint = QColor(accent)
        joint.setAlpha(190)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(joint)
        for index, point in enumerate(points):
            if index in active_tips:
                continue
            painter.drawEllipse(point, 2.6, 2.6)

        for finger in Finger:
            degree = offset + int(finger)
            self._paint_badge(
                painter,
                points[FINGER_TIPS[finger]],
                degree,
                accent,
                degree in self._active,
            )

    def _paint_badge(
        self,
        painter: QPainter,
        tip: QPointF,
        degree: int,
        accent: QColor,
        active: bool,
    ) -> None:
        """Degree number plus note name at the fingertip.

        Fill, outline weight and text all change with state, so the cue never
        depends on colour alone (UX-2.4).
        """
        radius = 15.0 if active else 11.0
        fill = QColor(accent) if active else QColor(theme.BACKGROUND)
        fill.setAlpha(235 if active else 150)
        pen = QPen(accent)
        pen.setWidthF(3.0 if active else 1.4)
        painter.setPen(pen)
        painter.setBrush(fill)

        if degree < 5:  # right hand: circle
            painter.drawEllipse(tip, radius, radius)
        else:  # left hand: rounded square
            painter.drawRoundedRect(
                QRectF(tip.x() - radius, tip.y() - radius, radius * 2, radius * 2),
                5.0,
                5.0,
            )

        font = QFont(self.font())
        font.setPointSizeF(11.0 if active else 9.0)
        font.setBold(active)
        painter.setFont(font)
        painter.setPen(theme.BACKGROUND if active else accent)
        painter.drawText(
            QRectF(tip.x() - radius, tip.y() - radius, radius * 2, radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            CIRCLED[degree],
        )

        if active:
            name = self._names[degree] if degree < len(self._names) else ""
            font.setPointSizeF(10.0)
            painter.setFont(font)
            painter.setPen(theme.TEXT)
            painter.drawText(
                QRectF(tip.x() - 40, tip.y() - radius - 20, 80, 16),
                Qt.AlignmentFlag.AlignCenter,
                name,
            )

    def _paint_hints(self, painter: QPainter, rect: QRectF) -> None:
        message = self._hint
        if message is None:
            if not self._observations and time.monotonic() - self._last_hand_time > NO_HAND_HINT_S:
                message = "Show your hand to the camera"
            elif self._observations and max(o.score for o in self._observations) < LOW_CONFIDENCE:
                message = "Low tracking confidence - improve lighting"
        if not message:
            return

        font = QFont(self.font())
        font.setPointSizeF(13.0)
        painter.setFont(font)
        box = QRectF(rect.x() + 24, rect.bottom() - 62, rect.width() - 48, 38)
        background = QColor(theme.SURFACE)
        background.setAlpha(215)
        painter.setPen(QPen(theme.BORDER))
        painter.setBrush(background)
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(theme.TEXT_DIM)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, message)

    def _paint_card(
        self, painter: QPainter, title: str, body: str, accent: QColor
    ) -> None:
        width = min(520.0, self.width() - 48.0)
        height = 150.0
        box = QRectF(
            (self.width() - width) / 2, (self.height() - height) / 2, width, height
        )
        painter.setPen(QPen(accent, 1.5))
        painter.setBrush(theme.SURFACE)
        painter.drawRoundedRect(box, 10, 10)

        font = QFont(self.font())
        font.setPointSizeF(14.0)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(accent)
        painter.drawText(
            QRectF(box.x() + 20, box.y() + 16, box.width() - 40, 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        font.setPointSizeF(11.5)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(theme.TEXT_DIM)
        painter.drawText(
            QRectF(box.x() + 20, box.y() + 48, box.width() - 40, box.height() - 64),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            | int(Qt.TextFlag.TextWordWrap),
            body,
        )


def _to_qimage(frame_bgr: np.ndarray) -> QImage:
    """BGR ndarray -> RGB QImage that owns its buffer."""
    height, width = frame_bgr.shape[:2]
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()
