"""Main window: preview, controls, status and the note strip (UX-1, UX-4, UX-5)."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..audio.engine import AudioEngine, list_output_devices
from ..config import AppConfig
from ..music.tuning import NOTE_NAMES
from ..performance import Performance
from ..vision.camera import CameraDevice, list_cameras
from ..vision.worker import FrameUpdate, VisionWorker
from . import theme
from .note_strip import NoteStrip
from .settings_dialog import SettingsDialog
from .video_widget import VideoWidget

log = logging.getLogger(__name__)

PANEL_WIDTH = 300
FIRST_RUN_HINT = "Fingers play scale degrees - right hand 1-5, left hand 6-10"
FIRST_RUN_HINT_MS = 9000


class CameraProbe(QThread):
    """Probing camera indices opens each device, so it happens off the UI thread.

    The index the vision thread is already using is skipped, not probed.
    """

    found = Signal(object)  # list[CameraDevice]

    def __init__(self, active_index: int, parent=None) -> None:
        super().__init__(parent)
        self._active_index = active_index

    def run(self) -> None:
        try:
            self.found.emit(list_cameras(skip={self._active_index}))
        except Exception:  # noqa: BLE001 - a failed probe just leaves the list as-is
            log.exception("camera probe failed")


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        performance: Performance,
        engine: AudioEngine,
        worker: VisionWorker,
    ) -> None:
        super().__init__()
        self.config = config
        self.performance = performance
        self.engine = engine
        self.worker = worker
        self._muted = False
        self._gain_before_mute = config.master_gain

        self.setWindowTitle("VisualSynth")
        self.resize(1100, 720)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(theme.STYLESHEET)

        self.video = VideoWidget()
        self.video.set_swap_hands(config.swap_hands)
        self.strip = NoteStrip()

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        top.addWidget(self.video, stretch=1)
        top.addWidget(self._build_panel())
        outer.addLayout(top, stretch=1)
        outer.addWidget(self.strip)
        self.setCentralWidget(central)

        self._build_shortcuts()
        self._connect_worker()
        self._refresh_note_names()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(500)

        self._probe = CameraProbe(config.camera_index, self)
        self._probe.found.connect(self._on_cameras_found)
        self._probe.start()

        if not config.first_run_hint_shown:
            self.video.set_hint(FIRST_RUN_HINT)
            QTimer.singleShot(FIRST_RUN_HINT_MS, lambda: self.video.set_hint(None))
            config.first_run_hint_shown = True

    # ---- construction ----

    def _build_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(PANEL_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        layout.addWidget(self._title("SCALE"))
        self.scale_combo = QComboBox()
        self._populate_scales()
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        layout.addWidget(self.scale_combo)

        layout.addWidget(self._title("ROOT"))
        self.root_combo = QComboBox()
        self.root_combo.addItems(
            [f"{name}{self.config.base_octave}" for name in NOTE_NAMES]
        )
        self.root_combo.setCurrentIndex(self.config.root_pitch_class)
        self.root_combo.currentIndexChanged.connect(self._on_root_changed)
        layout.addWidget(self.root_combo)

        layout.addWidget(self._separator())

        layout.addWidget(self._title("CAMERA"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem(f"Camera {self.config.camera_index}", self.config.camera_index)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        layout.addWidget(self.camera_combo)

        self.swap_check = QCheckBox("Swap left / right hands")
        self.swap_check.setChecked(self.config.swap_hands)
        self.swap_check.toggled.connect(self._on_swap_toggled)
        layout.addWidget(self.swap_check)

        layout.addWidget(self._title("AUDIO OUTPUT"))
        self.audio_combo = QComboBox()
        self._populate_audio_devices()
        self.audio_combo.currentIndexChanged.connect(self._on_audio_changed)
        layout.addWidget(self.audio_combo)

        layout.addWidget(self._title("VOLUME"))
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(self.config.master_gain * 100))
        self.volume.valueChanged.connect(self._on_volume_changed)
        layout.addWidget(self.volume)

        layout.addWidget(self._separator())

        layout.addWidget(self._title("STATUS"))
        self.status_label = QLabel()
        self.status_label.setObjectName("statusValue")
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        self.settings_button = QPushButton("Detection settings...")
        self.settings_button.clicked.connect(self._open_settings)
        layout.addWidget(self.settings_button)

        self.panic_button = QPushButton("PANIC  (Esc)")
        self.panic_button.setObjectName("panicButton")
        self.panic_button.clicked.connect(self.panic)
        layout.addWidget(self.panic_button)
        return panel

    def _title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _populate_scales(self) -> None:
        """Grouped by family, with the family headers shown as disabled rows."""
        model_row = 0
        for family, scales in self.performance.registry.by_family().items():
            self.scale_combo.addItem(f"-- {family} --")
            item = self.scale_combo.model().item(model_row)
            item.setEnabled(False)
            model_row += 1
            for scale in scales:
                self.scale_combo.addItem(scale.name)
                model_row += 1
        self.scale_combo.setCurrentText(self.performance.scale_name)

    def _populate_audio_devices(self) -> None:
        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        for device in list_output_devices():
            self.audio_combo.addItem(device.label, device.index)
        index = self.audio_combo.findData(self.engine.device)
        if index >= 0:
            self.audio_combo.setCurrentIndex(index)
        self.audio_combo.blockSignals(False)

    def _build_shortcuts(self) -> None:
        panic = QAction("Panic", self)
        panic.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        panic.triggered.connect(self.panic)
        self.addAction(panic)

        mute = QAction("Mute", self)
        mute.setShortcut(QKeySequence(Qt.Key.Key_Space))
        mute.triggered.connect(self.toggle_mute)
        self.addAction(mute)

        settings = QAction("Settings", self)
        settings.setShortcut(QKeySequence("Ctrl+,"))
        settings.triggered.connect(self._open_settings)
        self.addAction(settings)

    def _connect_worker(self) -> None:
        self.worker.frameReady.connect(self._on_frame)
        self.worker.failed.connect(self._on_failure)
        self.worker.recovered.connect(lambda: self.video.set_error(None))

    # ---- slots ----

    def _on_frame(self, update: FrameUpdate) -> None:
        if self.video.error:
            self.video.set_error(None)
        self.video.set_frame(update)
        self.strip.set_active(update.active_degrees)
        self._last_update = update

    def _on_failure(self, message: str) -> None:
        self.video.set_error(message)
        self.strip.set_active(frozenset())

    def _on_scale_changed(self, name: str) -> None:
        if name.startswith("--"):
            return
        self.performance.set_scale(name)
        self.config.scale_name = name
        self._refresh_note_names()

    def _on_root_changed(self, index: int) -> None:
        self.performance.set_root(index)
        self.config.root_pitch_class = index
        self._refresh_note_names()

    def _on_camera_changed(self, index: int) -> None:
        device = self.camera_combo.itemData(index)
        if device is None:
            return
        self.config.camera_index = int(device)
        self.worker.request_camera(int(device))

    def _on_cameras_found(self, devices: list[CameraDevice]) -> None:
        """Merge probed devices with the one already running (never re-probed)."""
        active = self.config.camera_index
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for index in sorted({active} | {d.index for d in devices}):
            device = next((d for d in devices if d.index == index), None)
            self.camera_combo.addItem(
                device.label if device else f"Camera {index}", index
            )
        self.camera_combo.setCurrentIndex(max(0, self.camera_combo.findData(active)))
        self.camera_combo.blockSignals(False)

    def _on_swap_toggled(self, swap: bool) -> None:
        self.performance.panic()
        self.worker.set_swap_hands(swap)
        self.video.set_swap_hands(swap)
        self.config.swap_hands = swap

    def _on_audio_changed(self, index: int) -> None:
        device = self.audio_combo.itemData(index)
        if device is None or device == self.engine.device:
            return
        try:
            self.engine.restart(device=int(device))
            self.config.audio_device = int(device)
        except Exception as exc:  # noqa: BLE001 - fall back and tell the user
            log.exception("could not switch audio device")
            self.video.set_hint(f"Audio device unavailable: {exc}")

    def _on_volume_changed(self, value: int) -> None:
        gain = value / 100.0
        self._muted = False
        self.config.master_gain = gain
        self.engine.synth.set_master_gain(gain)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        thresholds = dialog.thresholds()
        self.config.thresholds = thresholds
        self.config.idle_inference_fps = dialog.idle_inference_fps()
        self.worker.set_thresholds(thresholds)
        self.worker.set_idle_inference_fps(self.config.idle_inference_fps)

    # ---- actions ----

    def panic(self) -> None:
        self.performance.panic()
        self.strip.set_active(frozenset())

    def toggle_mute(self) -> None:
        if self._muted:
            self.engine.synth.set_master_gain(self._gain_before_mute)
            self.volume.setValue(int(self._gain_before_mute * 100))
            self._muted = False
        else:
            self._gain_before_mute = self.config.master_gain
            self.engine.synth.set_master_gain(0.0)
            self._muted = True

    # ---- periodic ----

    def _refresh_note_names(self) -> None:
        names = self.performance.degree_names()
        self.strip.set_note_names(names)
        self.video.set_note_names(names)

    def _refresh_status(self) -> None:
        update = getattr(self, "_last_update", None)
        fps = update.fps if update else 0.0
        inference_fps = update.inference_fps if update else 0.0
        inference_ms = update.inference_ms if update else 0.0
        hands = len(update.observations) if update else 0

        def colour(value: str, ok: bool) -> str:
            return f'<span style="color:{(theme.TEXT if ok else theme.WARNING).name()}">{value}</span>'

        rows = [
            f"preview {colour(f'{fps:4.1f} fps', fps >= 20)}",
            f"detect&nbsp; {colour(f'{inference_fps:4.1f} fps', inference_fps >= 15 or hands == 0)}"
            f" ({inference_ms:.0f} ms)",
            f"hands&nbsp;&nbsp; {hands}",
            f"audio&nbsp;&nbsp; {self.engine.output_latency_ms:.0f} ms buffer "
            f"{self.engine.settings.block_size}",
            f"xruns&nbsp;&nbsp; {colour(str(self.engine.xruns), self.engine.xruns == 0)}",
            f"voices&nbsp; {self.engine.synth.active_voice_count}",
        ]
        self.status_label.setText("<br>".join(rows))

    # ---- lifecycle ----

    def closeEvent(self, event) -> None:
        self._status_timer.stop()
        self.worker.stop()
        self.performance.panic()
        self.engine.stop()
        if self._probe.isRunning():
            self._probe.wait(2000)
        try:
            self.config.save()
        except OSError:
            log.exception("could not save settings")
        super().closeEvent(event)
