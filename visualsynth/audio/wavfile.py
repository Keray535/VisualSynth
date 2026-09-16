"""Minimal RIFF/WAVE reader for imported wavetables (FR-7.5).

The stdlib `wave` module is not enough here. It rejects IEEE-float files with
`unknown format: 3`, and for `WAVE_FORMAT_EXTENSIBLE` it assumes PCM without
reading the SubFormat GUID - so a 32-bit float wavetable export, which is what
Serum and most editors write, would either raise or decode into garbage.

Pure stdlib + numpy, so `requirements.txt` stays pinned as it is (NFR-5.2), and
no Qt or OpenCV enters the audio layer (NFR-4.1).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

#: Biggest file we will read, so a mistaken drag-and-drop of a 2 GB recording
#: fails fast with a message instead of exhausting memory.
MAX_BYTES = 64 * 1024 * 1024


class WavError(ValueError):
    """The file is not a WAV we can decode. The message is shown to the user."""


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Read `path` as mono float64 in [-1, 1]. Returns (samples, sample_rate)."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise WavError(f"Could not open {source.name}: {exc}") from exc
    if size > MAX_BYTES:
        raise WavError(
            f"{source.name} is {size / 1e6:.0f} MB; wavetables are expected to be "
            f"under {MAX_BYTES // (1024 * 1024)} MB."
        )
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise WavError(f"Could not read {source.name}: {exc}") from exc
    return parse_wav(raw, source.name)


def parse_wav(raw: bytes, name: str = "file") -> tuple[np.ndarray, int]:
    """Decode WAV bytes. Split out from `read_wav` so tests need no temp files."""
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise WavError(f"{name} is not a RIFF/WAVE file.")

    fmt: tuple[int, int, int, int] | None = None
    data: bytes | None = None

    # RIFF is a chunk list: 4-byte id, 4-byte little-endian size, payload padded
    # to an even length. Sizes in the wild lie, so every read is bounds-checked.
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        (chunk_size,) = struct.unpack_from("<I", raw, offset + 4)
        body = offset + 8
        end = min(body + chunk_size, len(raw))
        if chunk_id == b"fmt ":
            fmt = _parse_fmt(raw[body:end], name)
        elif chunk_id == b"data":
            data = raw[body:end]
        offset = body + chunk_size + (chunk_size & 1)

    if fmt is None:
        raise WavError(f"{name} has no format chunk.")
    if not data:
        raise WavError(f"{name} contains no audio data.")

    format_tag, channels, sample_rate, bits = fmt
    samples = _decode(data, format_tag, bits, name)
    if channels > 1:
        usable = (samples.size // channels) * channels
        if usable == 0:
            raise WavError(f"{name} contains no complete audio frames.")
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        raise WavError(f"{name} contains no audio data.")
    return samples, sample_rate


def _parse_fmt(body: bytes, name: str) -> tuple[int, int, int, int]:
    if len(body) < 16:
        raise WavError(f"{name} has a truncated format chunk.")
    format_tag, channels, sample_rate, _bytes_per_s, _align, bits = struct.unpack_from(
        "<HHIIHH", body, 0
    )
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        # the real format is the first two bytes of the 16-byte SubFormat GUID,
        # which sits 24 bytes into the chunk
        if len(body) < 26:
            raise WavError(f"{name} has a truncated extensible format chunk.")
        (format_tag,) = struct.unpack_from("<H", body, 24)
    if channels < 1:
        raise WavError(f"{name} reports {channels} audio channels.")
    if sample_rate < 1:
        raise WavError(f"{name} reports a sample rate of {sample_rate} Hz.")
    return format_tag, channels, sample_rate, bits


def _decode(data: bytes, format_tag: int, bits: int, name: str) -> np.ndarray:
    """Raw sample bytes -> float64 in [-1, 1], interleaved channels intact."""
    if format_tag == WAVE_FORMAT_PCM:
        if bits == 8:  # 8-bit PCM is unsigned, every other width is signed
            return (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        if bits == 16:
            return _trim(data, 2, np.dtype("<i2")).astype(np.float64) / 32768.0
        if bits == 24:
            return _decode_24(data)
        if bits == 32:
            return _trim(data, 4, np.dtype("<i4")).astype(np.float64) / 2147483648.0
        raise WavError(f"{name} is {bits}-bit PCM, which is not supported.")

    if format_tag == WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            return _trim(data, 4, np.dtype("<f4")).astype(np.float64)
        if bits == 64:
            return _trim(data, 8, np.dtype("<f8")).astype(np.float64)
        raise WavError(f"{name} is {bits}-bit float, which is not supported.")

    raise WavError(
        f"{name} uses audio format {format_tag}; only PCM and IEEE float WAV "
        "files can be imported."
    )


def _trim(data: bytes, width: int, dtype: np.dtype) -> np.ndarray:
    """Drop a trailing partial sample, which a truncated file leaves behind."""
    usable = (len(data) // width) * width
    if usable == 0:
        return np.empty(0, dtype=dtype)
    return np.frombuffer(data[:usable], dtype=dtype)


def _decode_24(data: bytes) -> np.ndarray:
    """24-bit PCM: three little-endian bytes per sample, sign in the top byte."""
    usable = (len(data) // 3) * 3
    if usable == 0:
        return np.empty(0, dtype=np.float64)
    packed = np.frombuffer(data[:usable], dtype=np.uint8).reshape(-1, 3)
    # the high byte is read as int8 so numpy sign-extends it for us
    value = (
        packed[:, 0].astype(np.int32)
        | (packed[:, 1].astype(np.int32) << 8)
        | (packed[:, 2].view(np.int8).astype(np.int32) << 16)
    )
    return value.astype(np.float64) / 8388608.0
