"""Application settings and their JSON persistence (FR-6.2)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .audio.envelope import AdsrSettings
from .audio.synth import SynthSettings
from .music.scales import DEFAULT_SCALE_NAME
from .vision.finger_state import FingerThresholds

log = logging.getLogger(__name__)

APP_NAME = "visualsynth"


def settings_path() -> Path:
    """%APPDATA%/visualsynth/settings.json on Windows, ~/.config elsewhere."""
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / APP_NAME / "settings.json"


@dataclass
class AppConfig:
    # music
    scale_name: str = DEFAULT_SCALE_NAME
    root_pitch_class: int = 0  # 0 = C
    base_octave: int = 4  # static: octave is not gesture-controlled in this version

    # vision
    camera_index: int = 0
    mirror: bool = True
    swap_hands: bool = False
    thresholds: FingerThresholds = field(default_factory=FingerThresholds)
    #: Inference rate while no hand is in frame. Palm detection costs ~45 ms on
    #: this class of CPU, so idling at full camera rate burns a core for nothing.
    idle_inference_fps: float = 12.0
    #: Upper bound on preview/emit rate; the camera can burst faster than this
    #: after a slow inference frame, which would flood the UI with repaints.
    preview_fps: float = 30.0
    #: Parallel hand trackers. MediaPipe's CPU inference is single-threaded and
    #: costs ~42 ms per hand here, so one worker caps near 10 detections/s;
    #: three reach ~22/s at the cost of three cores.
    tracker_workers: int = 3

    # audio
    audio_device: int | None = None
    sample_rate: int | None = None  # None = follow the device
    block_size: int = 512
    master_gain: float = 0.5
    adsr: AdsrSettings = field(default_factory=AdsrSettings)

    # ui
    first_run_hint_shown: bool = False

    # ---- derived ----

    def synth_settings(self, sample_rate: int) -> SynthSettings:
        return SynthSettings(
            sample_rate=sample_rate,
            block_size=self.block_size,
            master_gain=self.master_gain,
            adsr=self.adsr,
        )

    # ---- persistence ----

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Tolerant of missing and unknown keys, so an old settings file still
        loads after the config grows."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            nested = _nested_type(f.name)
            if nested is not None and isinstance(value, dict):
                allowed = {g.name for g in fields(nested)}
                try:
                    kwargs[f.name] = nested(**{k: v for k, v in value.items() if k in allowed})
                except (TypeError, ValueError) as exc:
                    log.warning("ignoring bad %s in settings: %s", f.name, exc)
                continue
            kwargs[f.name] = value
        return cls(**kwargs)

    def save(self, path: Path | None = None) -> Path:
        target = path or settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        source = path or settings_path()
        if not source.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("could not read %s (%s); using defaults", source, exc)
            return cls()


def _nested_type(name: str) -> type | None:
    mapping: dict[str, type] = {"thresholds": FingerThresholds, "adsr": AdsrSettings}
    nested = mapping.get(name)
    return nested if nested is not None and is_dataclass(nested) else None
