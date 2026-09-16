"""One polyphonic voice: a wavetable oscillator multiplied by an ADSR envelope."""

from __future__ import annotations

import random

import numpy as np

from .envelope import AdsrEnvelope, AdsrSettings, EnvStage
from .wavetable import WavetableOscillator, WavetableView


class Voice:
    """A single sounding degree. Voices are preallocated and never reallocated;
    voice index == degree index, so no stealing logic is needed (FR-5.2)."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        settings: AdsrSettings | None = None,
        view: WavetableView | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.oscillator = WavetableOscillator(sample_rate, block_size, view, rng)
        self.envelope = AdsrEnvelope(sample_rate, block_size, settings)
        self._env_buf = np.empty(block_size, dtype=np.float64)

    @property
    def active(self) -> bool:
        return self.envelope.active

    @property
    def frequency(self) -> float:
        return self.oscillator.frequency

    def note_on(self, frequency: float) -> None:
        self.oscillator.set_frequency(frequency)
        # Restart the phase unless this is a genuine re-trigger of a note that is
        # still being held. A released note keeps `active` true through its whole
        # release tail, and in this instrument that is the normal state a degree
        # is re-struck from - testing `active` alone would swallow almost every
        # phase reset and make Phase and Rand inaudible (FR-7.4).
        if self.envelope.stage in (EnvStage.IDLE, EnvStage.RELEASE):
            self.oscillator.retrigger()
        self.envelope.note_on()

    def note_off(self) -> None:
        self.envelope.note_off()

    def set_frequency(self, frequency: float) -> None:
        """Retune without restarting the envelope (FR-4.6)."""
        self.oscillator.set_frequency(frequency)

    def silence(self) -> None:
        self.envelope.silence()

    def set_envelope(self, settings: AdsrSettings) -> None:
        self.envelope.set_settings(settings)

    # ---- wavetable (FR-7) ----

    def set_view(self, view: WavetableView) -> None:
        self.oscillator.set_view(view)

    def set_phase_offset(self, phase: float) -> None:
        self.oscillator.set_phase_offset(phase)

    def set_rand_phase(self, amount: float) -> None:
        self.oscillator.set_rand_phase(amount)

    def render(self, out: np.ndarray) -> bool:
        """Write this voice's block into `out`. Returns False if silent, in
        which case `out` is left untouched and must not be mixed."""
        if not self.envelope.active:
            return False
        n = out.shape[0]
        env = self._env_buf[:n]
        self.envelope.render(env)
        self.oscillator.render(out)
        np.multiply(out, env, out=out)
        return True
