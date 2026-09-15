"""Scale library and degree-to-pitch mapping.

A scale is a list of ascending semitone offsets from the root, all < 12, starting at 0.
Degrees wrap across octaves so any scale supports the 10 finger degrees (FR-4.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Ten degrees: right hand thumb..pinky = 0..4, left hand thumb..pinky = 5..9.
DEGREE_COUNT = 10

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "scales.json"


@dataclass(frozen=True)
class Scale:
    """One named scale: ascending semitone offsets from the root."""

    name: str
    family: str
    intervals: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.intervals:
            raise ValueError(f"scale {self.name!r} has no intervals")
        if self.intervals[0] != 0:
            raise ValueError(f"scale {self.name!r} must start at 0")
        if any(not 0 <= i < 12 for i in self.intervals):
            raise ValueError(f"scale {self.name!r} has an interval outside 0..11")
        if any(b <= a for a, b in zip(self.intervals, self.intervals[1:])):
            raise ValueError(f"scale {self.name!r} intervals must ascend strictly")

    def degree_to_offset(self, degree: int) -> int:
        """Semitones above the root for a 0-based degree, wrapping by octave."""
        if degree < 0:
            raise ValueError("degree must be >= 0")
        n = len(self.intervals)
        return self.intervals[degree % n] + 12 * (degree // n)

    def degree_to_midi(self, root: int, degree: int) -> int:
        """MIDI note for a 0-based degree over the given root MIDI note."""
        return root + self.degree_to_offset(degree)

    def degrees_to_midi(self, root: int, count: int = DEGREE_COUNT) -> list[int]:
        """MIDI notes for degrees 0..count-1."""
        return [self.degree_to_midi(root, d) for d in range(count)]


@dataclass(frozen=True)
class ScaleRegistry:
    """All known scales, kept in declaration order and grouped by family."""

    scales: tuple[Scale, ...]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.scales]

    @property
    def families(self) -> list[str]:
        seen: list[str] = []
        for scale in self.scales:
            if scale.family not in seen:
                seen.append(scale.family)
        return seen

    def by_family(self) -> dict[str, list[Scale]]:
        grouped: dict[str, list[Scale]] = {}
        for scale in self.scales:
            grouped.setdefault(scale.family, []).append(scale)
        return grouped

    def get(self, name: str) -> Scale:
        for scale in self.scales:
            if scale.name == name:
                return scale
        raise KeyError(f"unknown scale: {name!r}")

    def __len__(self) -> int:
        return len(self.scales)


def load_registry(path: Path | None = None) -> ScaleRegistry:
    """Load the scale library from JSON. Raises on malformed data."""
    data_path = path or _DATA_PATH
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    scales: list[Scale] = []
    for family in raw["families"]:
        for entry in family["scales"]:
            scales.append(
                Scale(
                    name=entry["name"],
                    family=family["name"],
                    intervals=tuple(entry["intervals"]),
                )
            )
    if not scales:
        raise ValueError(f"no scales found in {data_path}")
    return ScaleRegistry(tuple(scales))


@lru_cache(maxsize=1)
def default_registry() -> ScaleRegistry:
    """The bundled scale library, loaded once."""
    return load_registry()


DEFAULT_SCALE_NAME = "Major (Ionian)"
