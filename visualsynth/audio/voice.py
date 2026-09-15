"""One polyphonic voice: a saw oscillator multiplied by an ADSR envelope."""

from __future__ import annotations

import numpy as np

from .envelope import AdsrEnvelope, AdsrSettings
from .oscillator import SawOscillator


class Voice:
    """A single sounding degree. Voices are preallocated and never reallocated;
    voice index == degree index, so no stealing logic is needed (FR-5.2)."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        settings: AdsrSettings | None = None,
    ) -> None:
        self.oscillator = SawOscillator(sample_rate, block_size)
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
        if not self.envelope.active:
            self.oscillator.reset_phase()
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
