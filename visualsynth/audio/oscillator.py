"""Band-limited sawtooth oscillator (PolyBLEP).

A naive ramp aliases badly at high notes; PolyBLEP subtracts a polynomial
approximation of a band-limited step around the wrap point (FR-5.3).

All buffers are preallocated and every numpy op writes through `out=`, so
`render` does not allocate - it runs in the audio callback thread (FR-5.6).
"""

from __future__ import annotations

import numpy as np


class SawOscillator:
    """Mono PolyBLEP sawtooth, rendered a block at a time."""

    def __init__(self, sample_rate: int, block_size: int) -> None:
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        # ramp starts at 1 so sample 0 already advances the phase, which keeps
        # phase continuous across block boundaries.
        self._ramp = np.arange(1, block_size + 1, dtype=np.float64)
        self._phase_buf = np.empty(block_size, dtype=np.float64)
        self._t = np.empty(block_size, dtype=np.float64)
        self._t2 = np.empty(block_size, dtype=np.float64)
        self._mask = np.empty(block_size, dtype=bool)
        self._phase = 0.0
        self._increment = 0.0

    @property
    def frequency(self) -> float:
        return self._increment * self.sample_rate

    def set_frequency(self, hz: float) -> None:
        """Set the pitch. Phase is kept, so retuning does not click."""
        self._increment = max(0.0, float(hz)) / self.sample_rate

    def reset_phase(self, phase: float = 0.0) -> None:
        self._phase = float(phase) % 1.0

    def render(self, out: np.ndarray) -> None:
        """Write one block of band-limited saw into `out` (float64, in place)."""
        n = out.shape[0]
        inc = self._increment
        phase = self._phase_buf[:n]

        np.multiply(self._ramp[:n], inc, out=phase)
        np.add(phase, self._phase, out=phase)
        np.mod(phase, 1.0, out=phase)

        # naive ramp in [-1, 1)
        np.multiply(phase, 2.0, out=out)
        np.subtract(out, 1.0, out=out)

        if inc > 0.0:
            self._apply_polyblep(out, phase, inc, n)

        self._phase = float(phase[n - 1])

    def _apply_polyblep(
        self, out: np.ndarray, phase: np.ndarray, inc: float, n: int
    ) -> None:
        t = self._t[:n]
        t2 = self._t2[:n]
        mask = self._mask[:n]

        # just after the wrap: t = phase / inc in [0, 1) -> blep = 2t - t^2 - 1
        np.divide(phase, inc, out=t)
        np.multiply(t, t, out=t2)
        np.multiply(t, 2.0, out=t)
        np.subtract(t, t2, out=t)
        np.subtract(t, 1.0, out=t)
        np.less(phase, inc, out=mask)
        np.subtract(out, t, out=out, where=mask)

        # just before the wrap: t = (phase - 1) / inc in (-1, 0) -> blep = t^2 + 2t + 1
        np.subtract(phase, 1.0, out=t)
        np.divide(t, inc, out=t)
        np.multiply(t, t, out=t2)
        np.multiply(t, 2.0, out=t)
        np.add(t, t2, out=t)
        np.add(t, 1.0, out=t)
        np.greater(phase, 1.0 - inc, out=mask)
        np.subtract(out, t, out=out, where=mask)
