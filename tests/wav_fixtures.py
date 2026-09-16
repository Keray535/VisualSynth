"""Synthetic WAV files, so the importer is testable without any audio assets.

The stdlib `wave` module can only *write* PCM, and the formats most likely to
break the reader are the ones it cannot write at all - IEEE float and
`WAVE_FORMAT_EXTENSIBLE`. So every file here is assembled as raw RIFF bytes.
"""

from __future__ import annotations

import struct

import numpy as np

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

SAMPLE_RATE = 44_100

#: The GUID tail every EXTENSIBLE sub-format shares.
_GUID_TAIL = bytes.fromhex("0000100080001AA000389B71")


def cycle(samples: int = 2048, harmonic: int = 1) -> np.ndarray:
    """A single sine cycle in [-1, 1], the signal every fixture encodes."""
    return np.sin(2.0 * np.pi * harmonic * np.arange(samples) / samples)


def encode(
    samples: np.ndarray,
    bits: int = 16,
    float_format: bool = False,
    channels: int = 1,
    extensible: bool = False,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Build a complete RIFF/WAVE file from mono samples in [-1, 1]."""
    data = _pack(np.asarray(samples, dtype=np.float64), bits, float_format)
    if channels > 1:
        # duplicate into interleaved channels, so the reader has to mean them back
        width = bits // 8
        frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, width)
        data = np.tile(frames, (1, channels)).tobytes()

    tag = WAVE_FORMAT_IEEE_FLOAT if float_format else WAVE_FORMAT_PCM
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        WAVE_FORMAT_EXTENSIBLE if extensible else tag,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    if extensible:
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + _GUID_TAIL
    return _riff(fmt, data)


def _pack(samples: np.ndarray, bits: int, float_format: bool) -> bytes:
    if float_format:
        if bits == 32:
            return samples.astype("<f4").tobytes()
        if bits == 64:
            return samples.astype("<f8").tobytes()
        raise ValueError(f"no float encoding for {bits} bits")
    if bits == 8:
        return (np.round(samples * 127.0) + 128.0).astype(np.uint8).tobytes()
    if bits == 16:
        return np.round(samples * 32767.0).astype("<i2").tobytes()
    if bits == 24:
        packed = np.round(samples * 8388607.0).astype("<i4")
        return np.frombuffer(packed.tobytes(), dtype=np.uint8).reshape(-1, 4)[:, :3].tobytes()
    if bits == 32:
        return np.round(samples * 2147483647.0).astype("<i4").tobytes()
    raise ValueError(f"no PCM encoding for {bits} bits")


def _riff(fmt: bytes, data: bytes) -> bytes:
    chunks = b"WAVE"
    chunks += b"fmt " + struct.pack("<I", len(fmt)) + fmt + (b"\x00" if len(fmt) & 1 else b"")
    chunks += b"data" + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(chunks)) + chunks


def write(path, samples: np.ndarray, **kwargs) -> str:
    """Write a fixture to `path` and return it, for the loader tests."""
    path.write_bytes(encode(samples, **kwargs))
    return str(path)
