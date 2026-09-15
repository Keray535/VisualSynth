"""Audio DSP: oscillator, envelope, synth mixing (FR-5). No device needed."""

from __future__ import annotations

import numpy as np
import pytest

from visualsynth.audio.envelope import AdsrEnvelope, AdsrSettings, EnvStage
from visualsynth.audio.oscillator import SawOscillator
from visualsynth.audio.synth import Synth, SynthSettings
from visualsynth.audio.voice import Voice

SR = 48_000


def render_seconds(osc: SawOscillator, seconds: float, block: int) -> np.ndarray:
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
    harmonic[:3] = True  # ignore DC leakage
    return float(power[~harmonic].sum() / power.sum())


# ---------- oscillator ----------


def test_saw_fundamental_is_the_peak():
    osc = SawOscillator(SR, 512)
    osc.set_frequency(440.0)
    freqs, mag = spectrum(render_seconds(osc, 0.5, 512))
    assert freqs[int(np.argmax(mag))] == pytest.approx(440.0, abs=5.0)


def test_saw_has_harmonic_series_falling_at_6db_per_octave():
    osc = SawOscillator(SR, 512)
    osc.set_frequency(200.0)
    freqs, mag = spectrum(render_seconds(osc, 0.5, 512))

    def amp_at(hz: float) -> float:
        lo, hi = hz - 20.0, hz + 20.0
        return float(mag[(freqs >= lo) & (freqs <= hi)].max())

    # the k-th harmonic of a sawtooth is 1/k of the fundamental
    assert amp_at(400.0) / amp_at(200.0) == pytest.approx(0.5, rel=0.15)
    assert amp_at(600.0) / amp_at(200.0) == pytest.approx(1 / 3, rel=0.2)


@pytest.mark.parametrize("f0", [1000.0, 3000.0, 6000.0])
def test_polyblep_suppresses_aliasing_versus_naive_saw(f0):
    osc = SawOscillator(SR, 512)
    osc.set_frequency(f0)
    blep = render_seconds(osc, 0.25, 512)

    n = blep.size
    phase = (np.arange(1, n + 1) * (f0 / SR)) % 1.0
    naive = 2.0 * phase - 1.0

    assert aliasing_ratio(blep, f0) < 0.25 * aliasing_ratio(naive, f0)


def test_render_is_block_size_invariant():
    """Phase stays continuous across block boundaries."""
    big = SawOscillator(SR, 1024)
    big.set_frequency(333.0)
    one = np.empty(1024)
    big.render(one)

    small = SawOscillator(SR, 512)
    small.set_frequency(333.0)
    two = np.empty(1024)
    small.render(two[:512])
    small.render(two[512:])

    np.testing.assert_allclose(one, two, atol=1e-12)


def test_output_stays_in_range_and_finite():
    osc = SawOscillator(SR, 256)
    for f0 in (20.0, 440.0, 8000.0, 15000.0):
        osc.set_frequency(f0)
        out = render_seconds(osc, 0.05, 256)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.5


def test_zero_frequency_is_silent_ramp_free():
    osc = SawOscillator(SR, 128)
    osc.set_frequency(0.0)
    out = np.empty(128)
    osc.render(out)
    assert np.allclose(out, -1.0)  # phase frozen at 0 -> constant, no clicks


def test_render_does_not_allocate_per_block():
    osc = SawOscillator(SR, 512)
    osc.set_frequency(440.0)
    out = np.empty(512)
    osc.render(out)  # warm up
    before = id(osc._phase_buf), id(osc._t), id(osc._mask)
    for _ in range(20):
        osc.render(out)
    assert (id(osc._phase_buf), id(osc._t), id(osc._mask)) == before


# ---------- envelope ----------


def test_envelope_stages_and_levels():
    settings = AdsrSettings(attack_s=0.01, decay_s=0.01, sustain=0.5, release_s=0.01)
    env = AdsrEnvelope(SR, 1024, settings)
    buf = np.empty(1024)

    assert not env.active
    env.render(buf)
    assert np.all(buf == 0.0)

    env.note_on()
    assert env.active
    env.render(buf)  # 1024 samples > attack (480) + decay (480)
    assert buf.max() == pytest.approx(1.0, abs=1e-3)
    assert env.stage == EnvStage.SUSTAIN
    assert env.level == pytest.approx(0.5, abs=1e-3)

    env.render(buf)
    assert np.allclose(buf, 0.5)

    env.note_off()
    env.render(buf)
    assert env.stage == EnvStage.IDLE
    assert buf[-1] == pytest.approx(0.0, abs=1e-6)
    assert not env.active


def test_envelope_attack_takes_the_configured_time():
    env = AdsrEnvelope(SR, 4096, AdsrSettings(attack_s=0.05, decay_s=1.0, sustain=1.0))
    env.note_on()
    buf = np.empty(4096)
    env.render(buf)
    peak_index = int(np.argmax(buf >= 0.999))
    assert peak_index == pytest.approx(0.05 * SR, rel=0.02)


def test_envelope_is_monotonic_during_attack_and_release():
    env = AdsrEnvelope(SR, 2048, AdsrSettings(attack_s=0.02, decay_s=0.02, sustain=0.6))
    env.note_on()
    buf = np.empty(2048)
    env.render(buf)
    attack = buf[: int(0.02 * SR)]
    assert np.all(np.diff(attack) >= -1e-12)

    env.note_off()
    env.render(buf)
    release = buf[: int(0.12 * SR)]
    assert np.all(np.diff(release) <= 1e-12)


def test_retrigger_starts_from_current_level_not_zero():
    env = AdsrEnvelope(SR, 512, AdsrSettings(attack_s=0.05, decay_s=0.05, sustain=0.5))
    env.note_on()
    buf = np.empty(512)
    env.render(buf)
    level = env.level
    assert 0.0 < level < 1.0
    env.note_off()
    env.note_on()
    env.render(buf)
    assert buf[0] == pytest.approx(level, abs=0.05)  # no jump back to zero


def test_silence_resets_immediately():
    env = AdsrEnvelope(SR, 256, AdsrSettings())
    env.note_on()
    env.render(np.empty(256))
    env.silence()
    assert env.stage == EnvStage.IDLE and env.level == 0.0


def test_invalid_envelope_settings_rejected():
    with pytest.raises(ValueError):
        AdsrSettings(attack_s=-1.0)
    with pytest.raises(ValueError):
        AdsrSettings(sustain=1.5)


# ---------- voice ----------


def test_silent_voice_reports_inactive_and_leaves_buffer_alone():
    voice = Voice(SR, 256)
    buf = np.full(256, 7.0)
    assert voice.render(buf) is False
    assert np.all(buf == 7.0)


def test_voice_produces_sound_then_stops():
    voice = Voice(
        SR, 512, AdsrSettings(attack_s=0.001, decay_s=0.001, sustain=0.8, release_s=0.005)
    )
    voice.note_on(440.0)
    buf = np.empty(512)
    assert voice.render(buf) is True
    assert np.max(np.abs(buf)) > 0.1

    voice.note_off()
    for _ in range(5):
        voice.render(buf)
    assert voice.active is False


# ---------- synth ----------


def make_synth(**kwargs) -> Synth:
    settings = SynthSettings(
        sample_rate=SR,
        block_size=256,
        adsr=AdsrSettings(attack_s=0.001, decay_s=0.001, sustain=0.8, release_s=0.005),
        **kwargs,
    )
    return Synth(settings)


def test_synth_starts_silent():
    synth = make_synth()
    out = np.empty(256)
    synth.process(out)
    assert np.all(out == 0.0)
    assert synth.active_voice_count == 0


def test_note_on_sounds_and_note_off_releases():
    synth = make_synth()
    out = np.empty(256)
    synth.note_on(0, 440.0)
    synth.process(out)
    assert np.max(np.abs(out)) > 0.01
    assert synth.active_voice_count == 1

    synth.note_off(0)
    for _ in range(6):
        synth.process(out)
    assert synth.active_voice_count == 0
    assert np.allclose(out, 0.0)


def test_ten_voices_mix_without_clipping():
    synth = make_synth()
    out = np.empty(256)
    for degree in range(10):
        synth.note_on(degree, 220.0 * (1 + degree * 0.13))
    synth.set_master_gain(1.0)
    for _ in range(4):
        synth.process(out)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0
    assert synth.active_voice_count == 10


def test_panic_releases_every_voice():
    synth = make_synth()
    out = np.empty(256)
    for degree in range(10):
        synth.note_on(degree, 330.0)
    synth.process(out)
    synth.panic()
    for _ in range(6):
        synth.process(out)
    assert synth.active_voice_count == 0
    assert np.allclose(out, 0.0)


def test_commands_apply_in_order_within_one_block():
    synth = make_synth()
    out = np.empty(256)
    synth.note_on(3, 440.0)
    synth.note_off(3)
    synth.process(out)
    for _ in range(6):
        synth.process(out)
    assert synth.voices[3].active is False


def test_retune_keeps_the_voice_sounding():
    synth = make_synth()
    out = np.empty(256)
    synth.note_on(1, 440.0)
    synth.process(out)
    synth.retune(1, 660.0)
    synth.process(out)
    assert synth.voices[1].frequency == pytest.approx(660.0)
    assert synth.voices[1].active


def test_out_of_range_degree_is_ignored():
    synth = make_synth()
    out = np.empty(256)
    synth.note_on(99, 440.0)
    synth.note_on(-1, 440.0)
    synth.process(out)
    assert synth.active_voice_count == 0


def test_master_gain_is_clamped_and_applied():
    synth = make_synth()
    out = np.empty(256)
    synth.note_on(0, 440.0)
    synth.set_master_gain(5.0)
    synth.process(out)
    assert synth.master_gain == 1.0
    loud = np.max(np.abs(out))

    synth.set_master_gain(0.1)
    synth.process(out)
    assert np.max(np.abs(out)) < loud


def test_nan_from_a_voice_never_reaches_the_output():
    """NFR-3.4: a NaN/Inf must not reach the device."""
    synth = make_synth()

    class BrokenVoice:
        active = True
        frequency = 0.0

        def render(self, out: np.ndarray) -> bool:
            out[:] = np.nan
            out[0] = np.inf
            return True

    synth.voices[0] = BrokenVoice()
    out = np.empty(256)
    synth.process(out)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0
