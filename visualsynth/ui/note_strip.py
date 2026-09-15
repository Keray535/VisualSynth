"""The ten-slot degree indicator along the bottom of the window (UX-3)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..music.scales import DEGREE_COUNT
from . import theme

#: Screen order. The preview is mirrored, so the left hand sits on the left of
#: the window; within each hand the thumb is the finger nearest the centre, so
#: the left block counts down towards the middle (UX-3.1).
LEFT_BLOCK = (9, 8, 7, 6, 5)
RIGHT_BLOCK = (0, 1, 2, 3, 4)
SLOT_ORDER = LEFT_BLOCK + RIGHT_BLOCK

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


class NoteStrip(QWidget):
    """Shows which degree each finger plays and lights the sounding ones."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._names: list[str] = ["-"] * DEGREE_COUNT
        self._active: frozenset[int] = frozenset()
        self.setMinimumHeight(74)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # ---- state ----

    def set_note_names(self, names: list[str]) -> None:
        """Relabel from the current scale and root, even with no hand in frame."""
        self._names = list(names)
        self.update()

    def set_active(self, degrees: frozenset[int]) -> None:
        if degrees != self._active:
            self._active = degrees
            self.update()

    # ---- painting ----

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.SURFACE)

        margin = 12.0
        gap = 8.0
        group_gap = 28.0
        usable = self.width() - 2 * margin - group_gap - 8 * gap
        slot_w = max(48.0, usable / DEGREE_COUNT)
        slot_h = self.height() - 2 * margin
        y = margin

        x = margin
        for position, degree in enumerate(SLOT_ORDER):
            if position == len(LEFT_BLOCK):
                x += group_gap - gap
            self._paint_slot(painter, QRectF(x, y, slot_w, slot_h), degree)
            x += slot_w + gap

        self._paint_hand_labels(painter, margin)
        painter.end()

    def _paint_slot(self, painter: QPainter, rect: QRectF, degree: int) -> None:
        accent = theme.LEFT_ACCENT if degree >= 5 else theme.RIGHT_ACCENT
        active = degree in self._active

        background = QColor(accent)
        background.setAlpha(46 if active else 14)
        painter.setBrush(background)
        pen = QPen(accent if active else theme.BORDER)
        pen.setWidthF(2.4 if active else 1.0)
        painter.setPen(pen)
        radius = 10.0 if degree < 5 else 6.0  # circle vs square-ish (UX-2.5)
        painter.drawRoundedRect(rect, radius, radius)

        # degree number
        font = QFont(self.font())
        font.setPointSizeF(13.0)
        font.setBold(active)
        painter.setFont(font)
        painter.setPen(accent if active else theme.TEXT_DIM)
        top = QRectF(rect.x(), rect.y() + 6, rect.width(), rect.height() * 0.45)
        painter.drawText(top, Qt.AlignmentFlag.AlignCenter, CIRCLED[degree])

        # note name
        font.setPointSizeF(12.0)
        painter.setFont(font)
        painter.setPen(theme.TEXT if active else theme.TEXT_DIM)
        bottom = QRectF(
            rect.x(), rect.y() + rect.height() * 0.45, rect.width(), rect.height() * 0.5
        )
        name = self._names[degree] if degree < len(self._names) else "-"
        painter.drawText(bottom, Qt.AlignmentFlag.AlignCenter, name)

    def _paint_hand_labels(self, painter: QPainter, margin: float) -> None:
        font = QFont(self.font())
        font.setPointSizeF(9.0)
        font.setBold(True)
        painter.setFont(font)

        painter.setPen(theme.LEFT_ACCENT)
        painter.drawText(
            QRectF(margin, 0, self.width() / 2 - margin, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "LEFT HAND  6-10",
        )
        painter.setPen(theme.RIGHT_ACCENT)
        painter.drawText(
            QRectF(self.width() / 2, 0, self.width() / 2 - margin, 14),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "1-5  RIGHT HAND",
        )
