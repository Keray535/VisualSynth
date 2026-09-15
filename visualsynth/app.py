"""Application wiring: audio engine, performance, vision thread, window."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from .audio.engine import AudioEngine, resolve_output_config
from .config import AppConfig
from .music.scales import DEFAULT_SCALE_NAME, default_registry
from .performance import Performance
from .ui.main_window import MainWindow
from .vision.worker import VisionWorker

log = logging.getLogger(__name__)


def _start_audio(config: AppConfig) -> AudioEngine:
    """Open the output stream, falling back to the default device (UX-6.5)."""
    device, device_rate = resolve_output_config(config.audio_device)
    sample_rate = config.sample_rate or device_rate
    engine = AudioEngine(settings=config.synth_settings(sample_rate), device=device)
    try:
        engine.start()
    except Exception:  # noqa: BLE001 - keep the app usable without audio
        log.exception("audio device %s failed; falling back to the default", device)
        engine = AudioEngine(settings=config.synth_settings(device_rate), device=None)
        try:
            engine.start()
        except Exception:  # noqa: BLE001
            log.exception("no audio output available")
    return engine


def _resolve_scale(config: AppConfig) -> str:
    registry = default_registry()
    try:
        registry.get(config.scale_name)
    except KeyError:
        log.warning("unknown scale %r in settings; using %s", config.scale_name, DEFAULT_SCALE_NAME)
        config.scale_name = DEFAULT_SCALE_NAME
    return config.scale_name


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config = AppConfig.load()

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("VisualSynth")

    engine = _start_audio(config)
    performance = Performance(
        engine.synth,
        _resolve_scale(config),
        root_pitch_class=config.root_pitch_class,
        base_octave=config.base_octave,
    )
    worker = VisionWorker(config, performance.handle_events)

    window = MainWindow(config, performance, engine, worker)
    window.show()
    worker.start()

    try:
        return app.exec()
    finally:
        worker.stop()
        engine.stop()
