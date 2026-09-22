"""Wavetable panel: frame stack, morph position, phase and import (UX-8).

Fixed into the main window under the preview and never detachable - `Ctrl+T`
only hides and shows it. The instrument keeps playing either way, so a held
chord is heard morphing as the position moves (UX-8.1).

Everything drawn here arrives as plain float64 arrays from `WavetableController`;
the panel never touches a voice or an oscillator (NFR-4.1).
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..audio.wavetable import WavetableSettings
from ..audio.wavfile import WavError
from ..wavetable_library import WavetableController
from . import theme

log = logging.getLogger(__name__)

#: Slider resolution. Sliders are integers, every control here is a 0..1 float.
SLIDER_STEPS = 1000

#: Points drawn per cycle - far more than the eye needs, far less than 2048.
CYCLE_POINTS = 512
FRAME_POINTS = 256

#: A 64-frame table drawn frame by frame is mush, so the backdrop is thinned.
MAX_DRAWN_FRAMES = 16


class WavetableCanvas(QWidget):
    """The frame stack, with the sounding cycle highlighted in place (UX-8.2)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # low, because the panel shares the window height with the preview; the
        # splitter decides how much of it the stack actually gets
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frames: list[np.ndarray] = []
        self._cycle: np.ndarray = np.zeros(2)
        self._position = 0.0
        self._phase = 0.0
        self._rand_phase = 0.0

    # ---- state ----

    def set_table(self, frames: list[np.ndarray], cycle: np.ndarray) -> None:
        self._frames = list(frames)
        self._cycle = np.asarray(cycle, dtype=np.float64)
        self.update()

    def set_controls(self, position: float, phase: float, rand_phase: float) -> None:
        self._position = float(position)
        self._phase = float(phase)
        self._rand_phase = float(rand_phase)
        self.update()

    # ---- painting ----

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.SURFACE)

        inset = 18.0
        area = QRectF(
            inset, inset, max(1.0, self.width() - 2 * inset), max(1.0, self.height() - 2 * inset)
        )
        depth = min(30.0, area.height() * 0.22)
        plane = QRectF(area.x(), area.y() + depth, area.width() - depth, area.height() - depth)

        self._paint_backdrop(painter, plane, depth)
        self._paint_current(painter, plane, depth)
        painter.end()

    def _paint_backdrop(self, painter: QPainter, plane: QRectF, depth: float) -> None:
        """Every frame, dim, receding up and to the right in table order."""
        count = len(self._frames)
        if count == 0:
            return
        drawn = _thin(count, MAX_DRAWN_FRAMES)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # back to front, so nearer frames overlap the ones behind them
        for index in reversed(drawn):
            depth_t = index / (count - 1) if count > 1 else 0.0
            colour = QColor(theme.TEXT_DIM)
            colour.setAlpha(40 + int(70 * (1.0 - depth_t)))
            pen = QPen(colour)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.drawPolyline(_polyline(self._frames[index], plane, depth_t, depth))

    def _paint_current(self, painter: QPainter, plane: QRectF, depth: float) -> None:
        """The blended cycle, at the depth the position actually sits at."""
        depth_t = min(1.0, max(0.0, self._position))
        offset_x = depth_t * depth
        offset_y = -depth_t * depth
        rect = QRectF(
            plane.x() + offset_x, plane.y() + offset_y, plane.width(), plane.height()
        )

        # zero line, so the shape is readable against the stack
        zero = rect.y() + rect.height() / 2.0
        pen = QPen(theme.BORDER)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(rect.x(), zero), QPointF(rect.right(), zero))

        self._paint_rand_band(painter, rect)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(theme.RIGHT_ACCENT)
        pen.setWidthF(2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPolyline(_polyline(self._cycle, plane, depth_t, depth))

        self._paint_start_marker(painter, rect)

    def _paint_rand_band(self, painter: QPainter, rect: QRectF) -> None:
        """How far a note-on may push the start phase (UX-8.5).

        A ruler strip along the top rather than a wash over the whole plane: the
        waveform is the thing being read here, and a half-height tint buries it.
        """
        if self._rand_phase <= 0.0:
            return
        width = rect.width() * self._rand_phase
        band = QColor(theme.LEFT_ACCENT)
        band.setAlpha(95)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(band)
        painter.drawRect(QRectF(rect.x(), rect.y(), width, 5.0))

        edge = QColor(theme.LEFT_ACCENT)
        edge.setAlpha(120)
        pen = QPen(edge)
        pen.setWidthF(1.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(
            QPointF(rect.x() + width, rect.y()), QPointF(rect.x() + width, rect.bottom())
        )

    def _paint_start_marker(self, painter: QPainter, rect: QRectF) -> None:
        """The cycle is drawn rotated, so a note always starts at the left edge.

        The marker names the offset in degrees as well as showing it, so the cue
        never rests on position alone (UX-2.4).
        """
        pen = QPen(theme.LEFT_ACCENT)
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.drawLine(QPointF(rect.x(), rect.y()), QPointF(rect.x(), rect.bottom()))

        font = QFont(self.font())
        font.setPointSizeF(9.0)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(theme.LEFT_ACCENT)
        label = f"start {self._phase * 360.0:.0f}°"
        if self._rand_phase > 0.0:
            label += f" +{self._rand_phase * 360.0:.0f}° rand"
        painter.drawText(
            QRectF(rect.x() + 6, rect.y() + 8, 200.0, 14.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )


def _thin(count: int, limit: int) -> list[int]:
    """At most `limit` evenly spaced frame indices, always including the ends."""
    if count <= limit:
        return list(range(count))
    return sorted({int(round(i)) for i in np.linspace(0, count - 1, limit)})


def _polyline(cycle: np.ndarray, plane: QRectF, depth_t: float, depth: float) -> QPolygonF:
    """One cycle as a polyline in the plane offset by its depth in the stack."""
    n = int(cycle.size)
    if n < 2:
        return QPolygonF()
    offset_x = depth_t * depth
    offset_y = -depth_t * depth
    left = plane.x() + offset_x
    top = plane.y() + offset_y
    centre = top + plane.height() / 2.0
    half = plane.height() / 2.0
    step = plane.width() / (n - 1)
    return QPolygonF(
        [
            QPointF(left + i * step, centre - float(cycle[i]) * half)
            for i in range(n)
        ]
    )


class WavetablePanel(QFrame):
    """Frame display plus the position, phase, rand and import controls.

    A child of the main window, built once and only ever hidden or shown, so
    there is no window state to track and no way to tear it off.
    """

    def __init__(self, controller: WavetableController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("wavetablePanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.controller = controller

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        # everything that is not a slider on one line, so the canvas keeps the
        # height the panel is given
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self._title("WAVETABLE"))
        self.table_combo = QComboBox()
        self.table_combo.currentTextChanged.connect(self._on_table_changed)
        header.addWidget(self.table_combo, stretch=1)
        self.load_button = QPushButton("Load .wav...")
        self.load_button.clicked.connect(self._on_load)
        header.addWidget(self.load_button)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._on_reset)
        header.addWidget(self.reset_button)
        layout.addLayout(header)

        self.canvas = WavetableCanvas()
        layout.addWidget(self.canvas, stretch=1)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        layout.addLayout(grid)

        self.position = self._slider()
        self.position.valueChanged.connect(self._on_position_changed)
        self.position_value = self._readout()
        self._add_row(grid, 0, "POSITION", self.position, self.position_value)

        self.phase = self._slider()
        self.phase.valueChanged.connect(self._on_phase_changed)
        self.phase_value = self._readout()
        self._add_row(grid, 1, "PHASE", self.phase, self.phase_value)

        self.rand = self._slider()
        self.rand.valueChanged.connect(self._on_rand_changed)
        self.rand_value = self._readout()
        self._add_row(grid, 2, "RAND", self.rand, self.rand_value)

        # hidden while empty, so a failed import grows the panel instead of
        # leaving a blank line above the sliders (UX-8.7)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.ERROR.name()};")
        self.error.setVisible(False)
        layout.addWidget(self.error)

        self._populate_tables()
        self._pull_from_controller()

    # ---- construction ----

    def _title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _slider() -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        return slider

    @staticmethod
    def _readout() -> QLabel:
        label = QLabel()
        label.setObjectName("statusValue")
        label.setMinimumWidth(140)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    def _add_row(
        self, grid: QGridLayout, row: int, title: str, slider: QSlider, value: QLabel
    ) -> None:
        grid.addWidget(self._title(title), row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(value, row, 2)
        grid.setColumnStretch(1, 1)

    def focus_controls(self) -> None:
        """Ctrl+T shows the panel; the position slider is what it lands on."""
        self.position.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _set_error(self, message: str | None) -> None:
        self.error.setText(message or "")
        self.error.setVisible(bool(message))

    def _populate_tables(self) -> None:
        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        self.table_combo.addItems(self.controller.library.names())
        index = self.table_combo.findText(self.controller.table_name)
        if index >= 0:
            self.table_combo.setCurrentIndex(index)
        self.table_combo.blockSignals(False)

    # ---- slots ----

    def _on_table_changed(self, name: str) -> None:
        if not name:
            return
        self.controller.set_table(name)
        self._set_error(None)
        self._refresh()

    def _on_position_changed(self, value: int) -> None:
        self.controller.set_position(value / SLIDER_STEPS)
        self._refresh()

    def _on_phase_changed(self, value: int) -> None:
        self.controller.set_phase(value / SLIDER_STEPS)
        self._refresh()

    def _on_rand_changed(self, value: int) -> None:
        self.controller.set_rand_phase(value / SLIDER_STEPS)
        self._refresh()

    def _on_load(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load wavetable", "", "WAV audio (*.wav);;All files (*)"
        )
        if not path:
            return
        try:
            self.controller.import_wav(path)
        except (WavError, ValueError, OSError) as exc:
            # the current table keeps sounding; only the message changes (UX-8.7)
            log.warning("wavetable import failed: %s", exc)
            self._set_error(str(exc))
            return
        self._set_error(None)
        self._populate_tables()
        self._pull_from_controller()

    def _on_reset(self) -> None:
        defaults = WavetableSettings()
        self.controller.set_table(defaults.table_name)
        self.controller.set_position(defaults.position)
        self.controller.set_phase(defaults.phase)
        self.controller.set_rand_phase(defaults.rand_phase)
        self._set_error(None)
        self._populate_tables()
        self._pull_from_controller()

    # ---- refresh ----

    def _pull_from_controller(self) -> None:
        """Push controller state into the widgets without re-entering the slots."""
        for slider, value in (
            (self.position, self.controller.position),
            (self.phase, self.controller.phase),
            (self.rand, self.controller.rand_phase),
        ):
            slider.blockSignals(True)
            slider.setValue(int(round(value * SLIDER_STEPS)))
            slider.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        self.canvas.set_table(
            self.controller.frame_cycles(FRAME_POINTS),
            self.controller.display_cycle(CYCLE_POINTS),
        )
        self.canvas.set_controls(
            self.controller.position, self.controller.phase, self.controller.rand_phase
        )
        self.position_value.setText(self.controller.position_label())
        self.phase_value.setText(f"{self.controller.phase * 360.0:.0f}°")
        self.rand_value.setText(f"{self.controller.rand_phase * 100:.0f}%")
