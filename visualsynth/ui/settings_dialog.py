"""Detection tuning dialog (Ctrl+, - UX-4.5).

Exposes the thresholds behind FR-3 so they can be tuned for a camera, a room and
a pair of hands without editing code (FR-3.6).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..config import AppConfig
from ..vision.finger_state import FingerThresholds


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detection settings")
        self.setMinimumWidth(420)
        self._config = config
        t = config.thresholds

        layout = QVBoxLayout(self)
        intro = QLabel(
            "A finger counts as extended above the open angle and only releases "
            "below the closed angle; the gap between them stops chattering."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        layout.addLayout(form)

        self.extend = self._degrees(t.extend_deg)
        self.curl = self._degrees(t.curl_deg)
        self.thumb_extend = self._degrees(t.thumb_extend_deg)
        self.thumb_curl = self._degrees(t.thumb_curl_deg)
        self.thumb_on = self._ratio(t.thumb_abduction_on)
        self.thumb_off = self._ratio(t.thumb_abduction_off)
        self.on_delay = self._milliseconds(t.on_debounce_ms)
        self.off_delay = self._milliseconds(t.off_debounce_ms)
        self.hand_lost = QDoubleSpinBox()
        self.hand_lost.setRange(0.0, 2000.0)
        self.hand_lost.setSuffix(" ms")
        self.hand_lost.setDecimals(0)
        self.hand_lost.setValue(t.hand_lost_ms)
        self.idle_fps = QDoubleSpinBox()
        self.idle_fps.setRange(1.0, 60.0)
        self.idle_fps.setSuffix(" fps")
        self.idle_fps.setDecimals(0)
        self.idle_fps.setValue(config.idle_inference_fps)

        form.addRow("Finger open above", self.extend)
        form.addRow("Finger closed below", self.curl)
        form.addRow("Thumb open above", self.thumb_extend)
        form.addRow("Thumb closed below", self.thumb_curl)
        form.addRow("Thumb spread on", self.thumb_on)
        form.addRow("Thumb spread off", self.thumb_off)
        form.addRow("Delay before a note starts", self.on_delay)
        form.addRow("Delay before a note stops", self.off_delay)
        form.addRow("Release notes after hand lost", self.hand_lost)
        form.addRow("Idle detection rate", self.idle_fps)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ef6461;")
        layout.addWidget(self.error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(buttons)

    # ---- widgets ----

    @staticmethod
    def _degrees(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(60.0, 180.0)
        box.setSuffix(" deg")
        box.setDecimals(0)
        box.setValue(value)
        return box

    @staticmethod
    def _ratio(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 3.0)
        box.setSingleStep(0.05)
        box.setDecimals(2)
        box.setValue(value)
        return box

    @staticmethod
    def _milliseconds(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 500.0)
        box.setSingleStep(10.0)
        box.setDecimals(0)
        box.setSuffix(" ms")
        box.setValue(value)
        return box

    # ---- result ----

    def thresholds(self) -> FingerThresholds:
        return FingerThresholds(
            extend_deg=self.extend.value(),
            curl_deg=self.curl.value(),
            thumb_extend_deg=self.thumb_extend.value(),
            thumb_curl_deg=self.thumb_curl.value(),
            thumb_abduction_on=self.thumb_on.value(),
            thumb_abduction_off=self.thumb_off.value(),
            on_debounce_ms=self.on_delay.value(),
            off_debounce_ms=self.off_delay.value(),
            hand_lost_ms=self.hand_lost.value(),
        )

    def idle_inference_fps(self) -> float:
        return self.idle_fps.value()

    def _accept(self) -> None:
        try:
            self.thresholds()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()

    def _restore_defaults(self) -> None:
        defaults = FingerThresholds()
        self.extend.setValue(defaults.extend_deg)
        self.curl.setValue(defaults.curl_deg)
        self.thumb_extend.setValue(defaults.thumb_extend_deg)
        self.thumb_curl.setValue(defaults.thumb_curl_deg)
        self.thumb_on.setValue(defaults.thumb_abduction_on)
        self.thumb_off.setValue(defaults.thumb_abduction_off)
        self.on_delay.setValue(defaults.on_debounce_ms)
        self.off_delay.setValue(defaults.off_debounce_ms)
        self.hand_lost.setValue(defaults.hand_lost_ms)
        self.idle_fps.setValue(AppConfig().idle_inference_fps)
        self.error.clear()
