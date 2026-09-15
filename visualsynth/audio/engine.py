"""Real-time audio output: wraps a `Synth` in a sounddevice output stream.

This is the only audio module that touches hardware; everything DSP lives in
`synth.py` and is testable headlessly (NFR-4.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .synth import Synth, SynthSettings

log = logging.getLogger(__name__)

#: Host APIs preferred on Windows, best first. MME adds tens of ms (NFR-1.2).
_PREFERRED_HOST_APIS = ("Windows WASAPI", "Windows WDM-KS", "ASIO", "Windows DirectSound")


@dataclass(frozen=True)
class OutputDevice:
    index: int
    name: str
    host_api: str
    default_samplerate: float
    max_channels: int

    @property
    def label(self) -> str:
        return f"{self.name} ({self.host_api})"


def list_output_devices() -> list[OutputDevice]:
    """Every device that can play audio, in sounddevice index order."""
    host_apis = sd.query_hostapis()
    devices: list[OutputDevice] = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_output_channels"] < 1:
            continue
        devices.append(
            OutputDevice(
                index=index,
                name=info["name"],
                host_api=host_apis[info["hostapi"]]["name"],
                default_samplerate=float(info["default_samplerate"]),
                max_channels=int(info["max_output_channels"]),
            )
        )
    return devices


def preferred_output_device() -> OutputDevice | None:
    """Pick a sane output: the default device of the fastest available host API.

    Windows reports the same speaker several times, once per host API. MME
    measures ~100 ms of output latency on this machine, WASAPI ~10 ms, so the
    host API matters far more than the device (NFR-1.2). Device names are
    truncated to 31 characters under MME, which is why this matches on host API
    defaults rather than on names.
    """
    devices = list_output_devices()
    if not devices:
        return None
    by_index = {d.index: d for d in devices}

    host_apis = sd.query_hostapis()
    for api_name in _PREFERRED_HOST_APIS:
        for api in host_apis:
            if api["name"] != api_name:
                continue
            index = api.get("default_output_device", -1)
            if index is not None and index >= 0 and index in by_index:
                return by_index[index]

    try:
        default_index = sd.default.device[1]
    except (TypeError, IndexError):
        default_index = None
    return by_index.get(default_index, devices[0])


def resolve_output_config(
    device: int | None = None,
) -> tuple[int | None, int]:
    """Return (device index, sample rate) to build the synth with.

    The sample rate follows the device, because WASAPI shared mode only accepts
    the rate the Windows mixer is running at.
    """
    if device is None:
        chosen = preferred_output_device()
    else:
        chosen = next((d for d in list_output_devices() if d.index == device), None)
    if chosen is None:
        return device, 48_000
    return chosen.index, int(round(chosen.default_samplerate))


class AudioEngine:
    """Owns the output stream and pumps the synth from the audio callback."""

    def __init__(
        self,
        settings: SynthSettings | None = None,
        synth: Synth | None = None,
        device: int | None = None,
        channels: int = 2,
    ) -> None:
        self.settings = settings or (synth.settings if synth else SynthSettings())
        self.synth = synth or Synth(self.settings)
        self.device = device
        self.channels = channels
        self.xruns = 0
        self._stream: sd.OutputStream | None = None
        self._mono = np.empty(self.settings.block_size, dtype=np.float64)

    # ---- lifecycle ----

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.settings.sample_rate,
            blocksize=self.settings.block_size,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "audio started: %s Hz, block %s, latency %.1f ms",
            self.settings.sample_rate,
            self.settings.block_size,
            self.output_latency_ms,
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def restart(self, device: int | None = None) -> None:
        """Move to another output device without losing synth state (NFR-3.2)."""
        self.stop()
        if device is not None:
            self.device = device
        self.synth.panic()
        self.xruns = 0
        self.start()

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ---- status ----

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def output_latency_ms(self) -> float:
        if self._stream is None:
            return 0.0
        return float(self._stream.latency) * 1000.0

    # ---- audio thread ----

    def _callback(self, outdata, frames: int, _time, status) -> None:
        if status:
            if status.output_underflow:
                self.xruns += 1
            else:
                log.debug("audio status: %s", status)
        mono = self._mono[:frames]
        self.synth.process(mono)
        outdata[:, 0] = mono
        for ch in range(1, outdata.shape[1]):
            outdata[:, ch] = outdata[:, 0]
