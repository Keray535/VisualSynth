"""Band-limited wavetable oscillator (FR-7).

A wavetable is a stack of single-cycle frames. The built-in table holds three,
in order: saw, sine, square (FR-7.1). A position in [0, 1] scans the stack and
linearly blends the two frames it falls between (FR-7.3), so position 0 is the
plain saw this instrument shipped with.

Anti-aliasing is by mipmap. Reading a 2048-point table at a high pitch folds
every harmonic above Nyquist back into the signal, so each frame is stored once
per octave band with the harmonics that band cannot carry removed (FR-7.2).
Level `k` keeps `1024 >> k` harmonics, and a note picks its level from the phase
increment alone - no filtering, no per-sample branching.

The tables are built by FFT at load time. `render` itself allocates nothing and
holds no lock, because it runs in the audio callback (FR-5.6).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .wavfile import read_wav

#: Samples per single-cycle frame. 2048 is the de-facto standard for wavetable
#: exports, which is what makes the import split in `split_frames` work.
FRAME_SIZE = 2048

#: One mip level per octave. Level 10 is down to a single harmonic, which covers
#: everything up to Nyquist.
MIP_LEVELS = 11

#: Imported tables are capped so a 256-frame file does not cost 46 MB of mips.
MAX_FRAMES = 64

BASIC_TABLE_NAME = "Basic"


@dataclass(frozen=True)
class WavetableSettings:
    """The persisted oscillator state (FR-6.2). Lives here rather than in
    `config.py` for the same reason `AdsrSettings` does: with its domain."""

    table_name: str = BASIC_TABLE_NAME
    #: Where in the frame stack to read, 0..1 (FR-7.3).
    position: float = 0.0
    #: Start phase in cycles, 0..1 - shown as 0-360 degrees (FR-7.4).
    phase: float = 0.0
    #: Extra start phase drawn per note-on, 0..1 - shown as 0-100 %.
    rand_phase: float = 0.0

    def __post_init__(self) -> None:
        for name in ("position", "phase", "rand_phase"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not str(self.table_name).strip():
            raise ValueError("table_name must not be empty")


@dataclass(frozen=True, eq=False)
class WavetableView:
    """One blended mip stack: what the oscillator actually reads.

    Immutable and shared by every voice. Building one allocates, so it happens
    on the control thread and the audio thread only ever swaps the pointer.
    """

    name: str
    position: float
    mips: np.ndarray  # (MIP_LEVELS, FRAME_SIZE)

    @property
    def cycle(self) -> np.ndarray:
        """The full-bandwidth cycle, for drawing."""
        return self.mips[0]


@dataclass(frozen=True, eq=False)
class Wavetable:
    """A named stack of single-cycle frames plus their per-octave mips."""

    name: str
    frames: np.ndarray  # (F, FRAME_SIZE)
    mips: np.ndarray  # (F, MIP_LEVELS, FRAME_SIZE)

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @classmethod
    def from_frames(cls, name: str, frames: np.ndarray) -> "Wavetable":
        """Normalise every frame and build its mips. Raises on an empty table."""
        data = np.atleast_2d(np.asarray(frames, dtype=np.float64))
        if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != FRAME_SIZE:
            raise ValueError(
                f"wavetable {name!r} must be (frames, {FRAME_SIZE}), got {data.shape}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError(f"wavetable {name!r} contains NaN or Inf samples")

        normalised = np.empty_like(data)
        mips = np.empty((data.shape[0], MIP_LEVELS, FRAME_SIZE), dtype=np.float64)
        for index, frame in enumerate(data):
            normalised[index] = _normalise(frame)
            mips[index] = _build_mips(normalised[index])
        return cls(name=name, frames=normalised, mips=mips)

    def view_at(self, position: float) -> WavetableView:
        """Blend the two frames `position` falls between (FR-7.3)."""
        count = self.frame_count
        clamped = min(1.0, max(0.0, float(position)))
        if count == 1:
            return WavetableView(self.name, clamped, self.mips[0].copy())

        exact = clamped * (count - 1)
        lower = int(math.floor(exact))
        if lower >= count - 1:
            return WavetableView(self.name, clamped, self.mips[count - 1].copy())

        fraction = exact - lower
        blended = self.mips[lower] * (1.0 - fraction) + self.mips[lower + 1] * fraction
        return WavetableView(self.name, clamped, blended)

    def frame_label(self, position: float) -> str:
        """'saw', or 'saw -> sine 37%' mid-blend, for the UI readout."""
        count = self.frame_count
        clamped = min(1.0, max(0.0, float(position)))
        names = _BASIC_FRAME_NAMES if self.name == BASIC_TABLE_NAME else None
        if count == 1:
            return names[0] if names else "frame 1"

        exact = clamped * (count - 1)
        lower = min(count - 2, int(math.floor(exact)))
        fraction = exact - lower

        def label(index: int) -> str:
            if names is not None and index < len(names):
                return names[index]
            return f"frame {index + 1}"

        if fraction <= 0.0005:
            return label(lower)
        if fraction >= 0.9995:
            return label(lower + 1)
        return f"{label(lower)} → {label(lower + 1)}  {fraction * 100:.0f}%"


# ---- built-in table ----

_BASIC_FRAME_NAMES = ("saw", "sine", "square")


def _harmonic_frame(harmonics: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Additive single cycle. `+1j/k` gives -sin(kx), i.e. a rising ramp, so the
    built-in saw matches `SawOscillator`'s polarity rather than inverting it."""
    spectrum = np.zeros(FRAME_SIZE // 2 + 1, dtype=np.complex128)
    spectrum[harmonics] = 1j * amplitudes
    return np.fft.irfft(spectrum, FRAME_SIZE)


@lru_cache(maxsize=1)
def basic_wavetable() -> Wavetable:
    """Saw, sine, square - in that order (FR-7.1)."""
    k = np.arange(1, FRAME_SIZE // 2 + 1)
    odd = k[k % 2 == 1]
    frames = np.stack(
        [
            _harmonic_frame(k, 1.0 / k),  # saw: every harmonic at 1/k
            _harmonic_frame(k[:1], np.ones(1)),  # sine: the fundamental alone
            _harmonic_frame(odd, 1.0 / odd),  # square: odd harmonics at 1/k
        ]
    )
    return Wavetable.from_frames(BASIC_TABLE_NAME, frames)


@lru_cache(maxsize=1)
def basic_view() -> WavetableView:
    """The default voice source: the built-in table at position 0, i.e. a saw."""
    return basic_wavetable().view_at(0.0)


# ---- import ----


def load_wavetable(path: Path | str, name: str | None = None) -> Wavetable:
    """Build a table from a `.wav` file (FR-7.5). Raises `WavError` on bad input."""
    source = Path(path)
    samples, _sample_rate = read_wav(source)
    return Wavetable.from_frames(name or source.stem, split_frames(samples))


def split_frames(
    samples: np.ndarray,
    frame_size: int = FRAME_SIZE,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray:
    """Cut a mono signal into single-cycle frames.

    A length that divides evenly by `frame_size` is a multi-frame wavetable and
    is split as-is; anything else is treated as one cycle and resampled, which
    is the only sane reading of an arbitrary recording.
    """
    data = np.asarray(samples, dtype=np.float64).ravel()
    if data.size == 0:
        raise ValueError("cannot build a wavetable from an empty signal")

    if data.size >= frame_size and data.size % frame_size == 0:
        frames = data.reshape(-1, frame_size)
        if frames.shape[0] > max_frames:
            # keep an evenly spaced subset, so the morph still sweeps the whole
            # table rather than stopping partway through it
            picked = np.linspace(0, frames.shape[0] - 1, max_frames).round().astype(int)
            frames = frames[picked]
        return np.ascontiguousarray(frames)

    # one cycle, resampled - wrap the first sample onto the end so the loop joins
    wrapped = np.concatenate([data, data[:1]])
    source = np.arange(wrapped.size, dtype=np.float64)
    target = np.linspace(0.0, float(data.size), frame_size, endpoint=False)
    return np.interp(target, source, wrapped).reshape(1, frame_size)


# ---- helpers ----


def _normalise(frame: np.ndarray) -> np.ndarray:
    """Remove DC and scale to peak 1.0. A silent frame stays silent."""
    centred = frame - float(np.mean(frame))
    peak = float(np.max(np.abs(centred)))
    if peak <= 1e-12:
        return np.zeros_like(centred)
    return centred / peak


def _build_mips(frame: np.ndarray) -> np.ndarray:
    """Per-octave band-limited copies of one frame (FR-7.2).

    Levels are not renormalised: a band-limited copy is quieter because it holds
    less energy, and rescaling would make the level change audible as a jump in
    loudness when a note crosses an octave boundary.
    """
    spectrum = np.fft.rfft(frame)
    mips = np.empty((MIP_LEVELS, FRAME_SIZE), dtype=np.float64)
    for level in range(MIP_LEVELS):
        keep = max(1, (FRAME_SIZE // 2) >> level)
        trimmed = spectrum.copy()
        trimmed[0] = 0.0  # no DC, ever
        trimmed[keep + 1 :] = 0.0
        mips[level] = np.fft.irfft(trimmed, FRAME_SIZE)
    return mips


def level_for_increment(increment: float) -> int:
    """Mip level for a phase increment in cycles per sample.

    Harmonic `h` of a note at increment `inc` lands at `h * inc` cycles/sample and
    must stay under 0.5, so `h < 1 / (2 * inc)`. Level `k` carries `1024 >> k`
    harmonics, and the smallest safe `k` is `ceil(log2(FRAME_SIZE * inc))`.
    """
    scaled = FRAME_SIZE * float(increment)
    if not scaled > 0.0:
        return 0
    return min(MIP_LEVELS - 1, max(0, int(math.ceil(math.log2(scaled)))))


class WavetableOscillator:
    """Mono wavetable oscillator, rendered a block at a time.

    Same surface as `SawOscillator` plus the table, phase-offset and randomise
    controls, so `Voice` swaps one for the other without further changes.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        view: WavetableView | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        # ramp starts at 1 so sample 0 already advances the phase, which keeps
        # phase continuous across block boundaries (as in SawOscillator).
        self._ramp = np.arange(1, block_size + 1, dtype=np.float64)
        self._phase_buf = np.empty(block_size, dtype=np.float64)
        self._pos = np.empty(block_size, dtype=np.float64)
        self._frac = np.empty(block_size, dtype=np.float64)
        self._tap_a = np.empty(block_size, dtype=np.float64)
        self._tap_b = np.empty(block_size, dtype=np.float64)
        self._index = np.empty(block_size, dtype=np.int64)
        self._phase = 0.0
        self._increment = 0.0
        self._view = view if view is not None else basic_view()
        self._rng = rng if rng is not None else random.Random()
        #: Start phase in cycles, 0..1 (FR-7.4).
        self.phase_offset = 0.0
        #: How much of a cycle a note-on may add on top of `phase_offset`.
        self.rand_phase = 0.0

    # ---- control ----

    @property
    def frequency(self) -> float:
        return self._increment * self.sample_rate

    @property
    def view(self) -> WavetableView:
        return self._view

    def set_frequency(self, hz: float) -> None:
        """Set the pitch. Phase is kept, so retuning does not click."""
        self._increment = max(0.0, float(hz)) / self.sample_rate

    def set_view(self, view: WavetableView) -> None:
        """Swap the table. A pointer store - safe to call from the audio thread,
        which is where the queued command lands (FR-7.7)."""
        self._view = view

    def set_phase_offset(self, phase: float) -> None:
        self.phase_offset = float(phase) % 1.0

    def set_rand_phase(self, amount: float) -> None:
        self.rand_phase = min(1.0, max(0.0, float(amount)))

    def reset_phase(self, phase: float = 0.0) -> None:
        self._phase = float(phase) % 1.0

    def start_phase(self) -> float:
        """Where the next note-on should begin: offset plus the random spread."""
        phase = self.phase_offset
        if self.rand_phase > 0.0:
            phase += self._rng.random() * self.rand_phase
        return phase % 1.0

    def retrigger(self) -> None:
        """Jump to the configured start phase (FR-7.4)."""
        self.reset_phase(self.start_phase())

    # ---- audio thread ----

    def render(self, out: np.ndarray) -> None:
        """Write one block into `out` (float64, in place). Allocates nothing."""
        n = out.shape[0]
        inc = self._increment
        table = self._view.mips[level_for_increment(inc)]

        phase = self._phase_buf[:n]
        np.multiply(self._ramp[:n], inc, out=phase)
        np.add(phase, self._phase, out=phase)
        np.mod(phase, 1.0, out=phase)

        pos = self._pos[:n]
        frac = self._frac[:n]
        lower = self._tap_a[:n]
        upper = self._tap_b[:n]
        index = self._index[:n]

        np.multiply(phase, FRAME_SIZE, out=pos)
        np.floor(pos, out=frac)
        np.copyto(index, frac, casting="unsafe")
        np.subtract(pos, frac, out=frac)

        # linear interpolation between the two neighbouring table samples;
        # `wrap` closes the cycle without a branch on the last index
        np.take(table, index, out=lower, mode="wrap")
        np.add(index, 1, out=index)
        np.take(table, index, out=upper, mode="wrap")
        np.subtract(upper, lower, out=upper)
        np.multiply(upper, frac, out=upper)
        np.add(lower, upper, out=out)

        self._phase = float(phase[n - 1])
