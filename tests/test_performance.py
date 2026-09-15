"""Degree -> note routing and settings persistence (FR-4.6, FR-6.2)."""

from __future__ import annotations

import json

import pytest

from visualsynth.audio.envelope import AdsrSettings
from visualsynth.audio.synth import Synth, SynthSettings
from visualsynth.config import AppConfig
from visualsynth.music.tuning import midi_to_freq, root_midi
from visualsynth.performance import Performance
from visualsynth.vision.finger_state import FingerThresholds, NoteEvent


@pytest.fixture
def performance() -> Performance:
    synth = Synth(SynthSettings(sample_rate=48_000, block_size=128))
    return Performance(synth, "Major (Ionian)", root_pitch_class=0, base_octave=4)


# ---- routing ----


def test_note_on_starts_the_matching_voice(performance):
    performance.handle_events([NoteEvent(degree=2, on=True)])
    performance.synth._drain_commands()
    voice = performance.synth.voices[2]
    assert voice.active
    assert voice.frequency == pytest.approx(midi_to_freq(64))  # E4
    assert performance.sounding_degrees == {2}


def test_note_off_releases_and_forgets_the_degree(performance):
    performance.handle_events([NoteEvent(2, True), NoteEvent(2, False)])
    performance.synth._drain_commands()
    assert performance.sounding_degrees == set()


def test_all_ten_degrees_route_to_their_own_voice(performance):
    performance.handle_events([NoteEvent(d, True) for d in range(10)])
    performance.synth._drain_commands()
    expected = [midi_to_freq(m) for m in performance.scale.degrees_to_midi(root_midi(0, 4))]
    actual = [v.frequency for v in performance.synth.voices]
    assert actual == pytest.approx(expected)


def test_out_of_range_degrees_are_ignored(performance):
    performance.handle_events([NoteEvent(99, True), NoteEvent(-3, True)])
    assert performance.sounding_degrees == set()


def test_panic_clears_everything(performance):
    performance.handle_events([NoteEvent(d, True) for d in range(10)])
    performance.panic()
    performance.synth._drain_commands()
    assert performance.sounding_degrees == set()
    assert all(not v.envelope.active or v.envelope.stage.name == "RELEASE" for v in performance.synth.voices)


# ---- retuning ----


def test_changing_scale_retunes_held_notes(performance):
    performance.handle_events([NoteEvent(1, True)])
    performance.synth._drain_commands()
    assert performance.synth.voices[1].frequency == pytest.approx(midi_to_freq(62))  # D4

    performance.set_scale("Minor Pentatonic")
    performance.synth._drain_commands()
    assert performance.synth.voices[1].frequency == pytest.approx(midi_to_freq(63))  # Eb4
    assert performance.synth.voices[1].active  # still sounding, not restarted


def test_changing_root_transposes_everything(performance):
    performance.set_root(7)  # G
    assert performance.degree_names()[0] == "G4"
    assert performance.degree_midi(0) == root_midi(7, 4)


def test_degree_names_follow_the_scale(performance):
    assert performance.degree_names() == [
        "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5",
    ]
    performance.set_scale("Whole Tone")
    assert performance.degree_names()[:3] == ["C4", "D4", "E4"]


# ---- config persistence ----


def test_config_round_trips_through_json(tmp_path):
    config = AppConfig(
        scale_name="Blues",
        root_pitch_class=5,
        camera_index=2,
        swap_hands=True,
        master_gain=0.25,
        thresholds=FingerThresholds(extend_deg=150.0, curl_deg=130.0),
        adsr=AdsrSettings(attack_s=0.02),
    )
    path = tmp_path / "settings.json"
    config.save(path)

    loaded = AppConfig.load(path)
    assert loaded.scale_name == "Blues"
    assert loaded.root_pitch_class == 5
    assert loaded.camera_index == 2
    assert loaded.swap_hands is True
    assert loaded.master_gain == 0.25
    assert loaded.thresholds == FingerThresholds(extend_deg=150.0, curl_deg=130.0)
    assert loaded.adsr.attack_s == 0.02


def test_missing_file_yields_defaults(tmp_path):
    assert AppConfig.load(tmp_path / "nope.json") == AppConfig()


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert AppConfig.load(path) == AppConfig()


def test_unknown_and_missing_keys_are_tolerated(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"scale_name": "Dorian", "from_a_future_version": 42}), encoding="utf-8"
    )
    config = AppConfig.load(path)
    assert config.scale_name == "Dorian"
    assert config.camera_index == AppConfig().camera_index


def test_bad_nested_values_do_not_break_loading(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"thresholds": {"extend_deg": 100.0, "curl_deg": 170.0}}),
        encoding="utf-8",
    )
    config = AppConfig.load(path)  # curl > extend is invalid and gets dropped
    assert config.thresholds == FingerThresholds()


def test_synth_settings_follow_the_device_rate():
    config = AppConfig(block_size=256, master_gain=0.4)
    settings = config.synth_settings(44_100)
    assert settings.sample_rate == 44_100
    assert settings.block_size == 256
    assert settings.master_gain == 0.4
