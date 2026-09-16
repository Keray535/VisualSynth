"""Wavetable oscillator, WAV import and the control seam (FR-7). No device needed."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

import wav_fixtures
from visualsynth.audio.oscillator import SawOscillator
from visualsynth.audio.synth import Synth, SynthSettings
from visualsynth.audio.wavetable import (
    BASIC_TABLE_NAME,
    FRAME_SIZE,
    MAX_FRAMES,
    MIP_LEVELS,
    Wavetable,
    WavetableOscillator,
    WavetableSettings,
    basic_view,
    basic_wavetable,
    level_for_increment,
    load_wavetable,
    split_frames,
)
from visualsynth.audio.wavfile import WavError, parse_wav
from visualsynth.config import AppConfig
from visualsynth.wavetable_library import WavetableController, WavetableLibrary

SR = 48_000


def render_seconds(osc: WavetableOscillator, seconds: float, block: int) -> np.ndarray:
    n_blocks = int(seconds * SR) // block
    buf = np.empty(block, dtype=np.float64)
    out = np.empty(n_blocks * block, dtype=np.float64)
    for i in range(n_blocks):
        osc.render(buf)
        out[i * block : (i + 1) * block] = buf
    return out


def spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windowed = signal * np.hanning(signal.size)
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / SR)
    return freqs, mag


def aliasing_ratio(signal: np.ndarray, f0: float) -> float:
    """Energy that is not near a harmonic of f0, relative to total energy."""
    freqs, mag = spectrum(signal)
    power = mag**2
    harmonic = np.zeros_like(power, dtype=bool)
    bin_hz = freqs[1] - freqs[0]
    k = 1
    while k * f0 < SR / 2:
        centre = int(round(k * f0 / bin_hz))
        harmonic[max(0, centre - 3) : centre + 4] = True
        k += 1
    harmonic[:3] = True
    return float(power[~harmonic].sum() / power.sum())


def naive_saw(f0: float, n: int) -> np.ndarray:
    return 2.0 * ((np.arange(1, n + 1) * (f0 / SR)) % 1.0) - 1.0


# ---------- built-in table ----------


def test_basic_table_is_saw_sine_square_in_order():
    table = basic_wavetable()
    assert table.name == BASIC_TABLE_NAME
    assert table.frame_count == 3
    assert table.frames.shape == (3, FRAME_SIZE)

    saw, sine, square = table.frames
    # a saw rises monotonically across the cycle; a square does not
    assert saw[FRAME_SIZE // 4] < saw[FRAME_SIZE // 2] < saw[3 * FRAME_SIZE // 4]
    np.testing.assert_allclose(
        sine, -np.sin(2.0 * np.pi * np.arange(FRAME_SIZE) / FRAME_SIZE), atol=1e-9
    )
    # the square's own half-cycles are flat next to the saw's ramp
    first_half = square[FRAME_SIZE // 8 : 3 * FRAME_SIZE // 8]
    assert float(np.std(first_half)) < 0.1
    assert float(np.std(saw[FRAME_SIZE // 8 : 3 * FRAME_SIZE // 8])) > 0.1


def test_every_frame_is_normalised_and_dc_free():
    table = basic_wavetable()
    for frame in table.frames:
        assert float(np.max(np.abs(frame))) == pytest.approx(1.0, abs=1e-9)
        assert float(np.mean(frame)) == pytest.approx(0.0, abs=1e-9)


def test_mips_are_built_per_octave_and_lose_harmonics():
    table = basic_wavetable()
    assert table.mips.shape == (3, MIP_LEVELS, FRAME_SIZE)
    saw_mips = table.mips[0]
    for level in range(1, MIP_LEVELS):
        keep = max(1, (FRAME_SIZE // 2) >> level)
        spectrum_ = np.abs(np.fft.rfft(saw_mips[level]))
        assert float(spectrum_[keep + 1 :].max()) < 1e-9
        assert float(spectrum_[keep]) > 0.0


def test_from_frames_rejects_bad_shapes_and_values():
    with pytest.raises(ValueError):
        Wavetable.from_frames("short", np.zeros((1, 64)))
    with pytest.raises(ValueError):
        Wavetable.from_frames("nan", np.full((1, FRAME_SIZE), np.nan))


# ---------- mip selection ----------


def test_level_rises_one_step_per_octave():
    base = level_for_increment(110.0 / SR)
    assert level_for_increment(220.0 / SR) == base + 1
    assert level_for_increment(440.0 / SR) == base + 2


def test_level_is_clamped_at_both_ends():
    assert level_for_increment(0.0) == 0
    assert level_for_increment(-1.0) == 0
    assert level_for_increment(1e-9) == 0
    assert level_for_increment(0.49) == MIP_LEVELS - 1


def test_chosen_level_never_keeps_a_harmonic_above_nyquist():
    """The whole point of the mips: h * f0 must stay under SR/2 (FR-7.2)."""
    for f0 in (55.0, 110.0, 220.0, 440.0, 1000.0, 4000.0, 9000.0):
        inc = f0 / SR
        kept = max(1, (FRAME_SIZE // 2) >> level_for_increment(inc))
        assert kept * f0 < SR / 2


# ---------- oscillator ----------


def test_wavetable_saw_has_the_same_harmonic_series_as_polyblep():
    osc = WavetableOscillator(SR, 512)
    osc.set_frequency(200.0)
    freqs, mag = spectrum(render_seconds(osc, 0.5, 512))

    def amp_at(hz: float) -> float:
        lo, hi = hz - 20.0, hz + 20.0
        return float(mag[(freqs >= lo) & (freqs <= hi)].max())

    assert freqs[int(np.argmax(mag))] == pytest.approx(200.0, abs=5.0)
    assert amp_at(400.0) / amp_at(200.0) == pytest.approx(0.5, rel=0.15)
    assert amp_at(600.0) / amp_at(200.0) == pytest.approx(1 / 3, rel=0.2)


def test_wavetable_saw_tracks_polyblep_saw():
    """Position 0 must still be the instrument's original sawtooth."""
    wavetable = WavetableOscillator(SR, 512)
    wavetable.set_frequency(440.0)
    polyblep = SawOscillator(SR, 512)
    polyblep.set_frequency(440.0)

    a = render_seconds(wavetable, 0.25, 512)
    b = render_seconds(polyblep, 0.25, 512)
    correlation = float(np.corrcoef(a, b)[0, 1])
    assert correlation > 0.99


@pytest.mark.parametrize("f0", [110.0, 440.0, 1000.0])
def test_mips_suppress_aliasing_versus_naive_saw(f0):
    osc = WavetableOscillator(SR, 512)
    osc.set_frequency(f0)
    rendered = render_seconds(osc, 0.25, 512)
    assert aliasing_ratio(rendered, f0) < 0.05 * aliasing_ratio(naive_saw(f0, rendered.size), f0)


def test_aliasing_is_effectively_gone_high_up():
    osc = WavetableOscillator(SR, 512)
    osc.set_frequency(6000.0)
    assert aliasing_ratio(render_seconds(osc, 0.25, 512), 6000.0) < 1e-6


def test_render_is_block_size_invariant():
    """Phase stays continuous across block boundaries."""
    big = WavetableOscillator(SR, 1024)
    big.set_frequency(333.0)
    one = np.empty(1024)
    big.render(one)

    small = WavetableOscillator(SR, 512)
    small.set_frequency(333.0)
    two = np.empty(1024)
    small.render(two[:512])
    small.render(two[512:])

    np.testing.assert_allclose(one, two, atol=1e-12)


def test_output_stays_in_range_and_finite():
    osc = WavetableOscillator(SR, 256)
    for f0 in (20.0, 440.0, 8000.0, 15000.0, 30000.0):
        osc.set_frequency(f0)
        out = render_seconds(osc, 0.05, 256)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.5


def test_zero_frequency_is_a_frozen_sample_not_a_ramp():
    osc = WavetableOscillator(SR, 128)
    osc.set_frequency(0.0)
    out = np.empty(128)
    osc.render(out)
    assert np.all(out == out[0])  # phase frozen -> constant, no clicks


def test_render_does_not_allocate_per_block():
    osc = WavetableOscillator(SR, 512)
    osc.set_frequency(440.0)
    out = np.empty(512)
    osc.render(out)  # warm up
    before = id(osc._phase_buf), id(osc._pos), id(osc._index), id(osc._tap_a)
    for _ in range(20):
        osc.render(out)
    assert (id(osc._phase_buf), id(osc._pos), id(osc._index), id(osc._tap_a)) == before


# ---------- morph position ----------


def test_position_ends_reproduce_the_first_and_last_frame():
    table = basic_wavetable()
    np.testing.assert_allclose(table.view_at(0.0).cycle, table.mips[0][0])
    np.testing.assert_allclose(table.view_at(1.0).cycle, table.mips[2][0])


def test_position_midpoints_land_on_a_frame_and_blend_between_them():
    table = basic_wavetable()
    np.testing.assert_allclose(table.view_at(0.5).cycle, table.mips[1][0])
    np.testing.assert_allclose(
        table.view_at(0.25).cycle, (table.mips[0][0] + table.mips[1][0]) / 2.0
    )


def test_position_is_clamped():
    table = basic_wavetable()
    np.testing.assert_allclose(table.view_at(-3.0).cycle, table.view_at(0.0).cycle)
    np.testing.assert_allclose(table.view_at(9.0).cycle, table.view_at(1.0).cycle)


def test_a_single_frame_table_morphs_to_itself():
    table = Wavetable.from_frames("one", wav_fixtures.cycle(FRAME_SIZE).reshape(1, -1))
    for position in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(table.view_at(position).cycle, table.mips[0][0])


def test_position_label_names_the_frames_it_sits_between():
    table = basic_wavetable()
    assert table.frame_label(0.0) == "saw"
    assert table.frame_label(0.5) == "sine"
    assert table.frame_label(1.0) == "square"
    assert "saw" in table.frame_label(0.25) and "sine" in table.frame_label(0.25)


def test_morphing_changes_what_the_oscillator_renders():
    table = basic_wavetable()
    osc = WavetableOscillator(SR, 512, table.view_at(0.0))
    osc.set_frequency(220.0)
    saw = render_seconds(osc, 0.2, 512)

    osc.set_view(table.view_at(0.5))
    osc.reset_phase(0.0)
    sine = render_seconds(osc, 0.2, 512)

    # a sine frame has essentially no second harmonic; a saw frame does
    freqs, saw_mag = spectrum(saw)
    _, sine_mag = spectrum(sine)
    window = (freqs >= 420.0) & (freqs <= 460.0)
    assert float(sine_mag[window].max()) < 0.02 * float(saw_mag[window].max())


# ---------- phase and rand ----------


def test_phase_offset_shifts_the_rendered_cycle():
    table = basic_wavetable()
    quarter = FRAME_SIZE // 4

    plain = WavetableOscillator(SR, FRAME_SIZE, table.view_at(0.5))
    plain.set_frequency(SR / FRAME_SIZE)  # exactly one cycle per block
    plain.retrigger()
    a = np.empty(FRAME_SIZE)
    plain.render(a)

    shifted = WavetableOscillator(SR, FRAME_SIZE, table.view_at(0.5))
    shifted.set_frequency(SR / FRAME_SIZE)
    shifted.set_phase_offset(0.25)
    shifted.retrigger()
    b = np.empty(FRAME_SIZE)
    shifted.render(b)

    np.testing.assert_allclose(b, np.roll(a, -quarter), atol=1e-9)


def test_rand_phase_stays_inside_its_range_and_varies():
    osc = WavetableOscillator(SR, 256, rng=random.Random(7))
    osc.set_phase_offset(0.2)
    osc.set_rand_phase(0.5)
    draws = {round(osc.start_phase(), 9) for _ in range(50)}
    assert len(draws) > 10  # actually random, not a constant
    assert all(0.2 <= d < 0.7 for d in draws)


def test_rand_phase_of_zero_is_exactly_the_offset():
    osc = WavetableOscillator(SR, 256, rng=random.Random(7))
    osc.set_phase_offset(0.3)
    osc.set_rand_phase(0.0)
    assert all(osc.start_phase() == pytest.approx(0.3) for _ in range(10))


def test_rand_phase_is_seeded_reproducibly():
    first = WavetableOscillator(SR, 256, rng=random.Random(11))
    second = WavetableOscillator(SR, 256, rng=random.Random(11))
    for osc in (first, second):
        osc.set_rand_phase(1.0)
    assert [first.start_phase() for _ in range(5)] == [second.start_phase() for _ in range(5)]


def test_phase_and_rand_are_clamped_and_wrapped():
    osc = WavetableOscillator(SR, 256)
    osc.set_phase_offset(1.25)
    assert osc.phase_offset == pytest.approx(0.25)
    osc.set_rand_phase(5.0)
    assert osc.rand_phase == 1.0
    osc.set_rand_phase(-1.0)
    assert osc.rand_phase == 0.0


def test_a_released_note_restarts_its_phase_but_a_held_one_does_not():
    """FR-7.4: the release tail keeps a voice `active`, so `active` alone is not
    the test for whether a phase reset is allowed."""
    from visualsynth.audio.envelope import AdsrSettings
    from visualsynth.audio.voice import Voice

    voice = Voice(SR, 256, AdsrSettings(attack_s=0.001, decay_s=0.001, sustain=0.8))
    voice.oscillator.set_phase_offset(0.0)

    voice.note_on(440.0)
    buf = np.empty(256)
    voice.render(buf)
    voice.oscillator.reset_phase(0.42)

    voice.note_on(440.0)  # re-trigger while still held: phase must survive
    assert voice.oscillator._phase == pytest.approx(0.42)

    voice.note_off()
    voice.oscillator.reset_phase(0.42)
    voice.note_on(440.0)  # after release: phase restarts from the offset
    assert voice.oscillator._phase == pytest.approx(0.0)


# ---------- wav reading ----------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 8},
        {"bits": 16},
        {"bits": 24},
        {"bits": 32},
        {"bits": 32, "float_format": True},
        {"bits": 64, "float_format": True},
        {"bits": 16, "extensible": True},
        {"bits": 32, "float_format": True, "extensible": True},
        {"bits": 16, "channels": 2},
        {"bits": 32, "float_format": True, "channels": 2},
    ],
)
def test_every_supported_encoding_decodes_to_the_same_wave(kwargs):
    expected = wav_fixtures.cycle(2048)
    samples, rate = parse_wav(wav_fixtures.encode(expected, **kwargs))
    assert rate == wav_fixtures.SAMPLE_RATE
    assert samples.size == expected.size
    tolerance = 0.02 if kwargs.get("bits") == 8 else 1e-4
    np.testing.assert_allclose(samples, expected, atol=tolerance)


def test_rejects_files_that_are_not_wav():
    with pytest.raises(WavError):
        parse_wav(b"not a wav at all")
    with pytest.raises(WavError):
        parse_wav(b"RIFF\x04\x00\x00\x00OGG ")


def test_rejects_an_unsupported_codec_with_a_readable_message():
    raw = bytearray(wav_fixtures.encode(wav_fixtures.cycle(512)))
    raw[20:22] = (85).to_bytes(2, "little")  # MP3 inside a RIFF container
    with pytest.raises(WavError, match="only PCM and IEEE float"):
        parse_wav(bytes(raw))


def test_rejects_a_file_with_no_audio_data():
    empty = wav_fixtures.encode(wav_fixtures.cycle(512))
    stripped = empty[: empty.index(b"data") + 8]
    with pytest.raises(WavError):
        parse_wav(stripped)


def test_a_truncated_data_chunk_is_read_as_far_as_it_goes():
    """Chunk sizes in real files lie, so the reader must not walk off the end."""
    raw = wav_fixtures.encode(wav_fixtures.cycle(2048), bits=16)
    samples, _rate = parse_wav(raw[: len(raw) - 1000])
    assert 0 < samples.size < 2048


# ---------- frame splitting ----------


@pytest.mark.parametrize(
    ("length", "expected"),
    [(FRAME_SIZE, 1), (2 * FRAME_SIZE, 2), (4 * FRAME_SIZE, 4), (733, 1), (5000, 1)],
)
def test_frame_count_follows_the_file_length(length, expected):
    frames = split_frames(wav_fixtures.cycle(length))
    assert frames.shape == (expected, FRAME_SIZE)


def test_a_single_cycle_file_is_resampled_not_truncated():
    frames = split_frames(wav_fixtures.cycle(512))
    assert frames.shape == (1, FRAME_SIZE)
    np.testing.assert_allclose(frames[0], wav_fixtures.cycle(FRAME_SIZE), atol=2e-5)


def test_a_huge_table_is_capped_but_still_spans_the_whole_file():
    frames = split_frames(np.concatenate([np.full(FRAME_SIZE, i) for i in range(200)]))
    assert frames.shape == (MAX_FRAMES, FRAME_SIZE)
    assert frames[0][0] == 0.0  # first frame kept
    assert frames[-1][0] == 199.0  # last frame kept


def test_an_empty_signal_is_rejected():
    with pytest.raises(ValueError):
        split_frames(np.zeros(0))


def test_load_wavetable_reads_a_file_from_disk(tmp_path):
    path = tmp_path / "two frames.wav"
    wav_fixtures.write(path, wav_fixtures.cycle(2 * FRAME_SIZE), bits=32, float_format=True)
    table = load_wavetable(path)
    assert table.name == "two frames"
    assert table.frame_count == 2


# ---------- library ----------


def test_library_starts_with_only_the_builtin(tmp_path):
    library = WavetableLibrary(tmp_path)
    library.scan()
    assert library.names() == [BASIC_TABLE_NAME]


def test_import_copies_the_file_so_it_survives_a_restart(tmp_path):
    source = tmp_path / "source" / "bells.wav"
    source.parent.mkdir()
    wav_fixtures.write(source, wav_fixtures.cycle(FRAME_SIZE))

    library = WavetableLibrary(tmp_path / "library")
    library.scan()
    assert library.import_wav(source) == "bells"

    reopened = WavetableLibrary(tmp_path / "library")
    reopened.scan()
    assert reopened.names() == [BASIC_TABLE_NAME, "bells"]


def test_importing_the_same_name_twice_does_not_overwrite(tmp_path):
    source = tmp_path / "bells.wav"
    wav_fixtures.write(source, wav_fixtures.cycle(FRAME_SIZE))
    library = WavetableLibrary(tmp_path / "library")
    library.scan()
    assert library.import_wav(source) == "bells"
    assert library.import_wav(source) == "bells (2)"


def test_a_bad_file_never_lands_in_the_library(tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"definitely not audio")
    library = WavetableLibrary(tmp_path / "library")
    library.scan()
    with pytest.raises(WavError):
        library.import_wav(bad)
    assert library.names() == [BASIC_TABLE_NAME]
    assert not (tmp_path / "library" / "broken.wav").exists()


def test_an_undecodable_file_in_the_directory_is_skipped_not_fatal(tmp_path):
    """NFR-3.3: one bad file must not stop the app from starting."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "broken.wav").write_bytes(b"nope")
    wav_fixtures.write(tmp_path / "good.wav", wav_fixtures.cycle(FRAME_SIZE))

    library = WavetableLibrary(tmp_path)
    library.scan()
    assert library.names() == [BASIC_TABLE_NAME, "good"]
    assert "broken" in library.failures


def test_a_user_file_cannot_shadow_the_builtin(tmp_path):
    wav_fixtures.write(tmp_path / f"{BASIC_TABLE_NAME}.wav", wav_fixtures.cycle(FRAME_SIZE))
    library = WavetableLibrary(tmp_path)
    library.scan()
    assert library.get(BASIC_TABLE_NAME).frame_count == 3


def test_unknown_name_raises(tmp_path):
    library = WavetableLibrary(tmp_path)
    library.scan()
    with pytest.raises(KeyError):
        library.get("nothing here")


# ---------- controller ----------


def make_controller(tmp_path, settings: WavetableSettings | None = None):
    synth = Synth(SynthSettings(sample_rate=SR, block_size=256))
    library = WavetableLibrary(tmp_path)
    library.scan()
    return synth, WavetableController(synth, library, settings)


def test_controller_installs_its_table_on_the_synth(tmp_path):
    synth, controller = make_controller(tmp_path)
    out = np.empty(256)
    synth.process(out)
    assert synth.view is controller._installed
    assert controller.settings() == WavetableSettings()


def test_controller_changes_reach_every_voice(tmp_path):
    synth, controller = make_controller(tmp_path)
    controller.set_position(0.5)
    controller.set_phase(0.25)
    controller.set_rand_phase(0.4)
    synth.process(np.empty(256))

    assert synth.view.position == pytest.approx(0.5)
    for voice in synth.voices:
        assert voice.oscillator.view is synth.view
        assert voice.oscillator.phase_offset == pytest.approx(0.25)
        assert voice.oscillator.rand_phase == pytest.approx(0.4)


def test_controller_keeps_the_installed_view_alive(tmp_path):
    """FR-7.7: the audio thread must never drop the last reference to a table."""
    synth, controller = make_controller(tmp_path)
    controller.set_position(0.3)
    installed = controller._installed
    controller.set_position(0.7)
    synth.process(np.empty(256))
    assert installed is not controller._installed  # a new view really was built
    assert synth.view is controller._installed


def test_controller_falls_back_when_a_saved_table_is_gone(tmp_path):
    _synth, controller = make_controller(
        tmp_path, WavetableSettings(table_name="deleted table")
    )
    assert controller.table_name == BASIC_TABLE_NAME


def test_controller_restores_saved_settings(tmp_path):
    saved = WavetableSettings(position=0.75, phase=0.5, rand_phase=0.25)
    _synth, controller = make_controller(tmp_path, saved)
    assert controller.settings() == saved


def test_controller_clamps_out_of_range_input(tmp_path):
    _synth, controller = make_controller(tmp_path)
    controller.set_position(4.0)
    controller.set_rand_phase(-2.0)
    controller.set_phase(1.25)
    assert controller.position == 1.0
    assert controller.rand_phase == 0.0
    assert controller.phase == pytest.approx(0.25)


def test_controller_import_switches_to_the_new_table(tmp_path):
    source = tmp_path / "pad.wav"
    wav_fixtures.write(source, wav_fixtures.cycle(2 * FRAME_SIZE))
    _synth, controller = make_controller(tmp_path / "library")
    assert controller.import_wav(source) == "pad"
    assert controller.table_name == "pad"
    assert controller.frame_count == 2


# ---------- drawing data ----------


def test_display_cycle_is_plain_finite_audio_range_data(tmp_path):
    _synth, controller = make_controller(tmp_path)
    cycle = controller.display_cycle(128)
    assert cycle.shape == (128,)
    assert cycle.dtype == np.float64
    assert np.all(np.isfinite(cycle))
    assert np.max(np.abs(cycle)) <= 1.0


def test_display_cycle_rotates_with_phase(tmp_path):
    _synth, controller = make_controller(tmp_path, WavetableSettings(position=0.5))
    plain = controller.display_cycle(256)
    controller.set_phase(0.25)
    rotated = controller.display_cycle(256)
    np.testing.assert_allclose(rotated, np.roll(plain, -64), atol=1e-3)


def test_frame_cycles_returns_one_entry_per_frame(tmp_path):
    _synth, controller = make_controller(tmp_path)
    cycles = controller.frame_cycles(64)
    assert len(cycles) == 3
    assert all(c.shape == (64,) for c in cycles)


# ---------- settings persistence ----------


def test_wavetable_settings_reject_out_of_range_values():
    for kwargs in ({"position": 1.5}, {"phase": -0.1}, {"rand_phase": 2.0}, {"table_name": " "}):
        with pytest.raises(ValueError):
            WavetableSettings(**kwargs)


def test_config_round_trips_wavetable_settings(tmp_path):
    path = tmp_path / "settings.json"
    config = AppConfig(
        wavetable=WavetableSettings(
            table_name="bells", position=0.4, phase=0.125, rand_phase=0.6
        )
    )
    config.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["wavetable"]["table_name"] == "bells"

    loaded = AppConfig.load(path)
    assert loaded.wavetable == config.wavetable


def test_an_old_settings_file_without_a_wavetable_block_still_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"scale_name": "Dorian", "master_gain": 0.3}), encoding="utf-8")
    loaded = AppConfig.load(path)
    assert loaded.scale_name == "Dorian"
    assert loaded.wavetable == WavetableSettings()


def test_a_corrupt_wavetable_block_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"wavetable": {"position": 99.0}}), encoding="utf-8")
    assert AppConfig.load(path).wavetable == WavetableSettings()


# ---------- synth integration ----------


def test_a_bare_synth_still_sounds_like_the_original_saw():
    synth = Synth(SynthSettings(sample_rate=SR, block_size=256))
    assert synth.view is basic_view()
    assert synth.view.position == 0.0


def test_ten_voices_on_a_morphed_table_stay_in_range():
    synth = Synth(SynthSettings(sample_rate=SR, block_size=256))
    synth.set_view(basic_wavetable().view_at(0.37))
    synth.set_rand_phase(1.0)
    for degree in range(10):
        synth.note_on(degree, 220.0 * (1 + degree * 0.13))
    synth.set_master_gain(1.0)

    out = np.empty(256)
    for _ in range(4):
        synth.process(out)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0
    assert synth.active_voice_count == 10


def test_import_into_a_never_scanned_library_keeps_the_builtin(tmp_path):
    source = tmp_path / "pad.wav"
    wav_fixtures.write(source, wav_fixtures.cycle(FRAME_SIZE))
    library = WavetableLibrary(tmp_path / "library")  # deliberately not scanned
    library.import_wav(source)
    assert library.names() == [BASIC_TABLE_NAME, "pad"]
