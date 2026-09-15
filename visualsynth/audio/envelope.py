"""Linear ADSR envelope, rendered a block at a time.

Segments are filled with vectorised ramps rather than a per-sample Python loop,
so the cost per block is a handful of numpy calls (FR-5.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class EnvStage(IntEnum):
    IDLE = 0
    ATTACK = 1
    DECAY = 2
    SUSTAIN = 3
    RELEASE = 4


@dataclass(frozen=True)
class AdsrSettings:
    """Envelope times in seconds, sustain as a linear level in [0, 1]."""

    attack_s: float = 0.008
    decay_s: float = 0.060
    sustain: float = 0.7
    release_s: float = 0.120

    def __post_init__(self) -> None:
        if min(self.attack_s, self.decay_s, self.release_s) < 0.0:
            raise ValueError("envelope times must be >= 0")
        if not 0.0 <= self.sustain <= 1.0:
            raise ValueError("sustain must be in [0, 1]")


class AdsrEnvelope:
    """Stateful ADSR. `note_on` ramps from the current level, so retriggering
    a still-sounding voice does not click."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        settings: AdsrSettings | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self._ramp = np.arange(1, block_size + 1, dtype=np.float64)
        self.stage = EnvStage.IDLE
        self.level = 0.0
        self.set_settings(settings or AdsrSettings())

    def set_settings(self, settings: AdsrSettings) -> None:
        self.settings = settings
        sr = self.sample_rate
        # per-sample slope of each segment; a zero time means "one sample"
        self._attack_rate = 1.0 / max(1.0, settings.attack_s * sr)
        self._decay_rate = (1.0 - settings.sustain) / max(1.0, settings.decay_s * sr)
        self._release_rate = 1.0 / max(1.0, settings.release_s * sr)

    @property
    def active(self) -> bool:
        return self.stage != EnvStage.IDLE

    def note_on(self) -> None:
        self.stage = EnvStage.ATTACK

    def note_off(self) -> None:
        if self.stage != EnvStage.IDLE:
            self.stage = EnvStage.RELEASE

    def silence(self) -> None:
        """Hard reset - used by panic. Callers should fade or accept the click."""
        self.stage = EnvStage.IDLE
        self.level = 0.0

    def render(self, out: np.ndarray) -> bool:
        """Fill `out` with the envelope for this block.

        Returns True if the envelope produced any non-zero output.
        """
        n = out.shape[0]
        i = 0
        produced = False

        while i < n:
            remaining = n - i
            if self.stage == EnvStage.IDLE:
                out[i:] = 0.0
                return produced

            if self.stage == EnvStage.SUSTAIN:
                out[i:] = self.level
                return produced or self.level > 0.0

            if self.stage == EnvStage.ATTACK:
                target, rate, nxt = 1.0, self._attack_rate, EnvStage.DECAY
            elif self.stage == EnvStage.DECAY:
                target, rate, nxt = self.settings.sustain, -self._decay_rate, EnvStage.SUSTAIN
            else:  # RELEASE
                target, rate, nxt = 0.0, -self._release_rate, EnvStage.IDLE

            if rate == 0.0:
                self.level = target
                self.stage = nxt
                continue

            steps = math.ceil((target - self.level) / rate)
            steps = max(0, min(steps, remaining))
            if steps:
                seg = out[i : i + steps]
                np.multiply(self._ramp[:steps], rate, out=seg)
                np.add(seg, self.level, out=seg)
                self.level = float(seg[steps - 1])
                produced = produced or self.level > 0.0 or target > 0.0
                i += steps

            if steps < remaining or steps == 0:
                # segment finished inside this block
                self.level = target
                self.stage = nxt

        return produced
