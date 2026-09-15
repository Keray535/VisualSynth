"""Glue between gestures and sound: degrees -> notes -> synth voices.

Owns the current scale and root, so the vision side only ever speaks in degrees
and the audio side only ever hears frequencies.
"""

from __future__ import annotations

import threading

from .audio.synth import Synth
from .music.scales import DEGREE_COUNT, Scale, ScaleRegistry, default_registry
from .music.tuning import midi_to_freq, midi_to_name, root_midi
from .vision.finger_state import NoteEvent


class Performance:
    """Current scale/root plus the degree -> voice routing."""

    def __init__(
        self,
        synth: Synth,
        scale_name: str,
        root_pitch_class: int = 0,
        base_octave: int = 4,
        registry: ScaleRegistry | None = None,
    ) -> None:
        self.synth = synth
        self.registry = registry or default_registry()
        self._lock = threading.Lock()
        self._scale: Scale = self.registry.get(scale_name)
        self._root_pitch_class = root_pitch_class % 12
        self._base_octave = base_octave
        self._sounding: set[int] = set()

    # ---- current tuning ----

    @property
    def scale(self) -> Scale:
        return self._scale

    @property
    def scale_name(self) -> str:
        return self._scale.name

    @property
    def root_pitch_class(self) -> int:
        return self._root_pitch_class

    @property
    def base_octave(self) -> int:
        return self._base_octave

    @property
    def root_midi(self) -> int:
        return root_midi(self._root_pitch_class, self._base_octave)

    def degree_midi(self, degree: int) -> int:
        return self._scale.degree_to_midi(self.root_midi, degree)

    def degree_frequency(self, degree: int) -> float:
        return midi_to_freq(self.degree_midi(degree))

    def degree_names(self) -> list[str]:
        """Note name per degree, for the note strip and fingertip badges."""
        return [midi_to_name(self.degree_midi(d)) for d in range(DEGREE_COUNT)]

    # ---- changes from the UI thread ----

    def set_scale(self, name: str) -> None:
        with self._lock:
            self._scale = self.registry.get(name)
            self._retune_sounding()

    def set_root(self, pitch_class: int, base_octave: int | None = None) -> None:
        with self._lock:
            self._root_pitch_class = pitch_class % 12
            if base_octave is not None:
                self._base_octave = base_octave
            self._retune_sounding()

    # ---- events from the vision thread ----

    def handle_events(self, events: list[NoteEvent]) -> None:
        if not events:
            return
        with self._lock:
            for event in events:
                if not 0 <= event.degree < DEGREE_COUNT:
                    continue
                if event.on:
                    self._sounding.add(event.degree)
                    self.synth.note_on(event.degree, self.degree_frequency(event.degree))
                else:
                    self._sounding.discard(event.degree)
                    self.synth.note_off(event.degree)

    def panic(self) -> None:
        with self._lock:
            self._sounding.clear()
            self.synth.panic()

    @property
    def sounding_degrees(self) -> set[int]:
        return set(self._sounding)

    def _retune_sounding(self) -> None:
        """FR-4.6: a scale or root change retunes held notes instead of clicking."""
        for degree in self._sounding:
            self.synth.retune(degree, self.degree_frequency(degree))
