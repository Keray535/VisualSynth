"""Polyphonic wavetable synth - pure DSP, no audio device, no Qt.

`Synth` owns the voices and the command queue. Control threads (vision, UI)
call `note_on` / `note_off` / `panic`; the audio thread calls `process`.
Commands cross the boundary through a queue drained at the top of each block
(FR-5.6), so no control thread ever touches voice state directly.

Wavetable changes travel the same way. Blending a `WavetableView` allocates, so
the control thread builds it and the queued command does nothing but store the
pointer in each voice (FR-7.7).
"""

from __future__ import annotations

import queue
import random
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ..music.scales import DEGREE_COUNT
from .envelope import AdsrSettings
from .voice import Voice
from .wavetable import WavetableView, basic_view


class Command(IntEnum):
    NOTE_ON = 0
    NOTE_OFF = 1
    RETUNE = 2
    PANIC = 3
    MASTER_GAIN = 4
    SET_VIEW = 5
    SET_PHASE = 6
    SET_RAND_PHASE = 7


@dataclass(frozen=True)
class SynthSettings:
    sample_rate: int = 48_000
    block_size: int = 512
    voice_count: int = DEGREE_COUNT
    master_gain: float = 0.5
    adsr: AdsrSettings = field(default_factory=AdsrSettings)


class Synth:
    def __init__(
        self,
        settings: SynthSettings | None = None,
        view: WavetableView | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings or SynthSettings()
        s = self.settings
        # One RNG for every voice: note-ons are all applied on the audio thread
        # while draining the queue, so a single generator needs no locking and
        # still gives each note its own random phase.
        self._rng = rng if rng is not None else random.Random()
        # Default to the built-in table at position 0, which is a saw - a bare
        # Synth() sounds exactly as it did before wavetables existed.
        self._view = view if view is not None else basic_view()
        self.voices = [
            Voice(s.sample_rate, s.block_size, s.adsr, self._view, self._rng)
            for _ in range(s.voice_count)
        ]
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        self._mix = np.empty(s.block_size, dtype=np.float64)
        self._scratch = np.empty(s.block_size, dtype=np.float64)
        self._master_gain = float(s.master_gain)
        self.active_voice_count = 0

    # ---- control-thread API (thread-safe, non-blocking) ----

    def note_on(self, degree: int, frequency: float) -> None:
        self._commands.put((Command.NOTE_ON, degree, frequency))

    def note_off(self, degree: int) -> None:
        self._commands.put((Command.NOTE_OFF, degree, 0.0))

    def retune(self, degree: int, frequency: float) -> None:
        self._commands.put((Command.RETUNE, degree, frequency))

    def panic(self) -> None:
        """Release every voice immediately (FR-5.8)."""
        self._commands.put((Command.PANIC, 0, 0.0))

    def set_master_gain(self, gain: float) -> None:
        self._commands.put((Command.MASTER_GAIN, 0, float(gain)))

    def set_view(self, view: WavetableView) -> None:
        """Install a blended wavetable. Build it on the calling thread; the
        audio thread only stores the reference (FR-7.7)."""
        self._commands.put((Command.SET_VIEW, 0, view))

    def set_phase_offset(self, phase: float) -> None:
        self._commands.put((Command.SET_PHASE, 0, float(phase)))

    def set_rand_phase(self, amount: float) -> None:
        self._commands.put((Command.SET_RAND_PHASE, 0, float(amount)))

    @property
    def master_gain(self) -> float:
        return self._master_gain

    @property
    def view(self) -> WavetableView:
        """The table last installed on the audio thread."""
        return self._view

    # ---- audio-thread API ----

    def process(self, out: np.ndarray) -> None:
        """Render one block of mixed audio into `out` (float64, in place)."""
        n = out.shape[0]
        self._drain_commands()

        mix = self._mix[:n]
        scratch = self._scratch[:n]
        mix[:] = 0.0

        active = 0
        for voice in self.voices:
            if voice.render(scratch):
                np.add(mix, scratch, out=mix)
                active += 1
        self.active_voice_count = active

        if active:
            # equal-power-ish normalisation, then soft clip, then master gain
            np.multiply(mix, 1.0 / np.sqrt(active), out=mix)
            np.tanh(mix, out=mix)
            np.multiply(mix, self._master_gain, out=mix)

        # NFR-3.4: never let a NaN/Inf or an out-of-range sample reach the device
        np.nan_to_num(mix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(mix, -1.0, 1.0, out=mix)
        out[:] = mix

    def _drain_commands(self) -> None:
        while True:
            try:
                kind, degree, value = self._commands.get_nowait()
            except queue.Empty:
                return
            self._apply(kind, degree, value)

    def _apply(self, kind: int, degree: int, value: object) -> None:
        if kind == Command.PANIC:
            for voice in self.voices:
                voice.note_off()
            return
        if kind == Command.MASTER_GAIN:
            self._master_gain = max(0.0, min(1.0, float(value)))
            return
        if kind == Command.SET_VIEW:
            # a pointer store per voice; the sender keeps the previous view
            # alive, so nothing is freed on the audio thread
            self._view = value
            for voice in self.voices:
                voice.set_view(value)
            return
        if kind == Command.SET_PHASE:
            for voice in self.voices:
                voice.set_phase_offset(float(value))
            return
        if kind == Command.SET_RAND_PHASE:
            for voice in self.voices:
                voice.set_rand_phase(float(value))
            return
        if not 0 <= degree < len(self.voices):
            return
        voice = self.voices[degree]
        if kind == Command.NOTE_ON:
            voice.note_on(float(value))
        elif kind == Command.NOTE_OFF:
            voice.note_off()
        elif kind == Command.RETUNE:
            voice.set_frequency(float(value))
