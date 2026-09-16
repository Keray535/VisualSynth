"""Wavetable library on disk and the control seam the UI talks to (FR-7.5, FR-7.6).

Sits beside `performance.py` for the same reason: the UI drives the synth through
a small object that speaks in user-facing terms (a table name, a position, a
phase), and never reaches into the DSP.

Imported `.wav` files are copied into `%APPDATA%/visualsynth/wavetables/`, so a
table stays available after a restart even if the user moves the original file.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from .audio.synth import Synth
from .audio.wavetable import (
    BASIC_TABLE_NAME,
    Wavetable,
    WavetableSettings,
    WavetableView,
    basic_wavetable,
    load_wavetable,
)
from .audio.wavfile import WavError
from .config import settings_path

log = logging.getLogger(__name__)


def user_wavetable_dir() -> Path:
    """%APPDATA%/visualsynth/wavetables, alongside settings.json."""
    return settings_path().parent / "wavetables"


class WavetableLibrary:
    """The built-in table plus every `.wav` in the user directory."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or user_wavetable_dir()
        self._tables: dict[str, Wavetable] = {}
        self._failed: dict[str, str] = {}

    # ---- discovery ----

    def scan(self) -> None:
        """Reload the library. Built-in first, then user files by name.

        A file that will not decode is skipped and remembered, not raised: one
        bad `.wav` must not stop the app from starting (NFR-3.3).
        """
        self._tables = {BASIC_TABLE_NAME: basic_wavetable()}
        self._failed = {}
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.wav")):
            name = path.stem
            if name == BASIC_TABLE_NAME:
                continue  # never let a user file shadow the built-in table
            try:
                self._tables[name] = load_wavetable(path, name)
            except (WavError, ValueError, OSError) as exc:
                log.warning("skipping wavetable %s: %s", path.name, exc)
                self._failed[name] = str(exc)

    def names(self) -> list[str]:
        if not self._tables:
            self.scan()
        return list(self._tables)

    def get(self, name: str) -> Wavetable:
        if not self._tables:
            self.scan()
        try:
            return self._tables[name]
        except KeyError:
            raise KeyError(f"unknown wavetable: {name!r}") from None

    @property
    def failures(self) -> dict[str, str]:
        """Files that could not be decoded, by name - surfaced in the UI."""
        return dict(self._failed)

    # ---- import ----

    def import_wav(self, source: Path | str) -> str:
        """Copy a `.wav` into the library and load it. Returns the table name.

        The file is decoded *before* it is copied, so a file that turns out not
        to be a usable wavetable never lands in the library directory.
        """
        if not self._tables:
            self.scan()  # or the built-in would be dropped from a never-scanned library
        path = Path(source)
        name = _unique_name(path.stem, set(self._tables))
        table = load_wavetable(path, name)

        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{name}.wav"
        try:
            if path.resolve() != target.resolve():
                shutil.copyfile(path, target)
        except OSError as exc:
            # the table is loaded and playable; it just will not survive a restart
            log.warning("could not copy %s into the library: %s", path.name, exc)

        self._tables[name] = table
        self._failed.pop(name, None)
        return name


def _unique_name(stem: str, taken: set[str]) -> str:
    """`bells`, then `bells (2)`, so importing twice does not overwrite."""
    base = stem.strip() or "wavetable"
    if base not in taken:
        return base
    for suffix in range(2, 1000):
        candidate = f"{base} ({suffix})"
        if candidate not in taken:
            return candidate
    return f"{base} ({len(taken)})"


class WavetableController:
    """Owns the current table and pushes blended views into the synth.

    Every `WavetableView` is built here, on the calling (UI) thread, and a
    reference to the installed one is kept so the audio thread's pointer swap
    never drops the last reference and never frees memory inside the callback
    (FR-7.7).
    """

    def __init__(
        self,
        synth: Synth,
        library: WavetableLibrary | None = None,
        settings: WavetableSettings | None = None,
    ) -> None:
        self.synth = synth
        self.library = library or WavetableLibrary()
        requested = settings or WavetableSettings()

        self._table = self._resolve(requested.table_name)
        self._position = requested.position
        self._phase = requested.phase
        self._rand_phase = requested.rand_phase
        self._installed: WavetableView = self._table.view_at(self._position)

        self.synth.set_view(self._installed)
        self.synth.set_phase_offset(self._phase)
        self.synth.set_rand_phase(self._rand_phase)

    def _resolve(self, name: str) -> Wavetable:
        """Fall back to the built-in table if a saved name has gone missing,
        the same way an unknown scale name is handled at startup."""
        try:
            return self.library.get(name)
        except KeyError:
            log.warning("unknown wavetable %r; using %s", name, BASIC_TABLE_NAME)
            return self.library.get(BASIC_TABLE_NAME)

    # ---- current state ----

    @property
    def table(self) -> Wavetable:
        return self._table

    @property
    def table_name(self) -> str:
        return self._table.name

    @property
    def position(self) -> float:
        return self._position

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def rand_phase(self) -> float:
        return self._rand_phase

    @property
    def frame_count(self) -> int:
        return self._table.frame_count

    def settings(self) -> WavetableSettings:
        return WavetableSettings(
            table_name=self._table.name,
            position=self._position,
            phase=self._phase,
            rand_phase=self._rand_phase,
        )

    def position_label(self) -> str:
        return self._table.frame_label(self._position)

    # ---- changes from the UI thread ----

    def set_table(self, name: str) -> None:
        self._table = self._resolve(name)
        self._install()

    def set_position(self, position: float) -> None:
        self._position = min(1.0, max(0.0, float(position)))
        self._install()

    def set_phase(self, phase: float) -> None:
        self._phase = float(phase) % 1.0
        self.synth.set_phase_offset(self._phase)

    def set_rand_phase(self, amount: float) -> None:
        self._rand_phase = min(1.0, max(0.0, float(amount)))
        self.synth.set_rand_phase(self._rand_phase)

    def import_wav(self, source: Path | str) -> str:
        """Import a file and switch to it. Raises `WavError` on a bad file."""
        name = self.library.import_wav(source)
        self.set_table(name)
        return name

    def _install(self) -> None:
        self._installed = self._table.view_at(self._position)
        self.synth.set_view(self._installed)

    # ---- drawing data for the UI ----

    def display_cycle(self, points: int = 512) -> np.ndarray:
        """The sounding waveform, phase-rotated as a note would start it.

        Plain float64 in [-1, 1], so the window never touches a live voice or a
        DSP object (NFR-4.1).
        """
        return _resample_cycle(self._installed.cycle, points, self._phase)

    def frame_cycles(self, points: int = 256) -> list[np.ndarray]:
        """Every frame of the current table, for the stacked backdrop."""
        return [_resample_cycle(frame, points, self._phase) for frame in self._table.frames]


def _resample_cycle(cycle: np.ndarray, points: int, phase: float) -> np.ndarray:
    """One cycle resampled to `points`, rotated so it starts at `phase`."""
    count = max(2, int(points))
    offset = (float(phase) % 1.0) * cycle.size
    index = (np.linspace(0.0, cycle.size, count, endpoint=False) + offset) % cycle.size
    wrapped = np.concatenate([cycle, cycle[:1]])
    return np.interp(index, np.arange(wrapped.size, dtype=np.float64), wrapped)


__all__ = [
    "WavetableController",
    "WavetableLibrary",
    "user_wavetable_dir",
]
