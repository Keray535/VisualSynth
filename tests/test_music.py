"""Scale library and pitch mapping (FR-4)."""

from __future__ import annotations

import math

import pytest

from visualsynth.music.scales import (
    DEGREE_COUNT,
    DEFAULT_SCALE_NAME,
    Scale,
    default_registry,
)
from visualsynth.music.tuning import midi_to_freq, midi_to_name, root_midi


@pytest.fixture(scope="module")
def registry():
    return default_registry()


def test_registry_loads_all_families(registry):
    assert len(registry) >= 30
    assert "Major modes" in registry.families
    assert len(registry.by_family()) == len(registry.families)


def test_scale_names_are_unique(registry):
    assert len(set(registry.names)) == len(registry.names)


def test_every_scale_is_well_formed(registry):
    """FR-4.2: ascending, starts at 0, every interval < 12."""
    for scale in registry.scales:
        assert scale.intervals[0] == 0, scale.name
        assert all(0 <= i < 12 for i in scale.intervals), scale.name
        assert list(scale.intervals) == sorted(set(scale.intervals)), scale.name


def test_every_scale_maps_ten_degrees(registry):
    root = root_midi(0, 4)
    for scale in registry.scales:
        notes = scale.degrees_to_midi(root)
        assert len(notes) == DEGREE_COUNT, scale.name
        assert notes == sorted(notes), scale.name
        assert notes[0] == root, scale.name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Major (Ionian)", ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"]),
        ("Minor Pentatonic", ["C4", "D#4", "F4", "G4", "A#4", "C5", "D#5", "F5", "G5", "A#5"]),
        ("Blues", ["C4", "D#4", "F4", "F#4", "G4", "A#4", "C5", "D#5", "F5", "F#5"]),
    ],
)
def test_known_scale_mappings(registry, name, expected):
    root = root_midi(0, 4)
    assert [midi_to_name(m) for m in registry.get(name).degrees_to_midi(root)] == expected


@pytest.mark.parametrize("length", [5, 6, 7, 8, 12])
def test_octave_wrap_rule(registry, length):
    """FR-4.3: degree n (scale length) is exactly one octave above degree 0."""
    scale = next(s for s in registry.scales if len(s.intervals) == length)
    root = root_midi(0, 4)
    assert scale.degree_to_midi(root, length) == root + 12
    assert scale.degree_to_midi(root, length + 1) == scale.degree_to_midi(root, 1) + 12


def test_major_degree_eight_is_the_octave(registry):
    scale = registry.get(DEFAULT_SCALE_NAME)
    root = root_midi(0, 4)
    assert scale.degree_to_midi(root, 7) == root + 12  # 0-based degree 7 = 8th degree


def test_root_midi_and_names():
    assert root_midi(0, 4) == 60
    assert root_midi(9, 4) == 69
    assert midi_to_name(60) == "C4"
    assert midi_to_name(69) == "A4"


def test_midi_to_freq():
    assert midi_to_freq(69) == pytest.approx(440.0)
    assert midi_to_freq(81) == pytest.approx(880.0)
    assert midi_to_freq(60) == pytest.approx(261.6255, abs=1e-3)
    assert midi_to_freq(57) == pytest.approx(220.0)


def test_negative_degree_rejected(registry):
    with pytest.raises(ValueError):
        registry.get(DEFAULT_SCALE_NAME).degree_to_offset(-1)


def test_unknown_scale_rejected(registry):
    with pytest.raises(KeyError):
        registry.get("Nope Major")


@pytest.mark.parametrize(
    "intervals",
    [(), (1, 2, 3), (0, 2, 12), (0, 4, 2)],
)
def test_malformed_scales_rejected(intervals):
    with pytest.raises(ValueError):
        Scale(name="bad", family="test", intervals=intervals)


def test_chromatic_covers_ten_semitones(registry):
    root = root_midi(0, 4)
    notes = registry.get("Chromatic").degrees_to_midi(root)
    assert notes == list(range(root, root + 10))


def test_frequency_doubles_per_octave(registry):
    scale = registry.get("Major Pentatonic")
    root = root_midi(0, 4)
    low = midi_to_freq(scale.degree_to_midi(root, 0))
    high = midi_to_freq(scale.degree_to_midi(root, 5))
    assert math.isclose(high / low, 2.0, rel_tol=1e-9)
