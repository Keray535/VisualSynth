"""Pitch helpers: MIDI note numbers, frequencies and note names.

Pure functions over ints/floats - no audio, no Qt, no camera (NFR-4.1).
"""

from __future__ import annotations

NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

DEFAULT_A4_HZ = 440.0

#: MIDI note number of A4, the reference pitch.
A4_MIDI = 69


def midi_to_freq(midi: float, a4_hz: float = DEFAULT_A4_HZ) -> float:
    """Convert a MIDI note number to a frequency in Hz (FR-4.5)."""
    return a4_hz * 2.0 ** ((midi - A4_MIDI) / 12.0)


def pitch_class_name(pitch_class: int) -> str:
    """Name of a pitch class, 0 = C .. 11 = B."""
    return NOTE_NAMES[pitch_class % 12]


def midi_to_name(midi: int) -> str:
    """Scientific pitch name of a MIDI note, e.g. 60 -> 'C4'."""
    return f"{pitch_class_name(midi)}{midi // 12 - 1}"


def root_midi(pitch_class: int, octave: int) -> int:
    """MIDI note of a root given its pitch class and octave (C4 = 60)."""
    return 12 * (octave + 1) + pitch_class % 12
