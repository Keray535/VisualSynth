"""Polyphonic saw synth - pure DSP, no audio device, no Qt.

`Synth` owns the voices and the command queue. Control threads (vision, UI)
call `note_on` / `note_off` / `panic`; the audio thread calls `process`.
Commands cross the boundary through a queue drained at the top of each block
(FR-5.6), so no control thread ever touches voice state directly.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ..music.scales import DEGREE_COUNT
from .envelope import AdsrSettings
from .voice import Voice


class Command(IntEnum):
    NOTE_ON = 0
    NOTE_OFF = 1
    RETUNE = 2
    PANIC = 3
    MASTER_GAIN = 4


@dataclass(frozen=True)
class SynthSettings:
    sample_rate: int = 48_000
    block_size: int = 512
    voice_count: int = DEGREE_COUNT
    master_gain: float = 0.5
    adsr: AdsrSettings = field(default_factory=AdsrSettings)


class Synth:
    def __init__(self, settings: SynthSettings | None = None) -> None:
        self.settings = settings or SynthSettings()
        s = self.settings
        self.voices = [
            Voice(s.sample_rate, s.block_size, s.adsr) for _ in range(s.voice_count)
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

    @property
    def master_gain(self) -> float:
        return self._master_gain

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

    def _apply(self, kind: int, degree: int, value: float) -> None:
        if kind == Command.PANIC:
            for voice in self.voices:
                voice.note_off()
            return
        if kind == Command.MASTER_GAIN:
            self._master_gain = max(0.0, min(1.0, value))
            return
        if not 0 <= degree < len(self.voices):
            return
        voice = self.voices[degree]
        if kind == Command.NOTE_ON:
            voice.note_on(value)
        elif kind == Command.NOTE_OFF:
            voice.note_off()
        elif kind == Command.RETUNE:
            voice.set_frequency(value)
