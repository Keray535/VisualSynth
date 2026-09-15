# VisualSynth

A desktop instrument you play with your hands in front of a webcam. Each finger is one
degree of a scale you pick in the UI; holding a finger **extended** sustains that note as a
sawtooth wave, curling it stops the note.

```
Right hand:  thumb=1  index=2  middle=3  ring=4  pinky=5
Left hand:   thumb=6  index=7  middle=8  ring=9  pinky=10
```

Open hand = five-note chord. Fist = silence. Both hands = all ten degrees at once.
Full requirements live in [requirements.md](requirements.md).

## Setup

```sh
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\fetch_models.py     # downloads hand_landmarker.task (7.8 MB)
.venv\Scripts\python -m visualsynth
```

Tested on Windows 11 with Python 3.14.5.

## Using it

| Control | What it does |
|---|---|
| Scale / Root | Retunes every degree instantly, even mid-note |
| Camera | Switches capture device without restarting |
| Swap left / right hands | Fixes inverted handedness if your camera reports it the other way |
| Audio output | Picks the device; WASAPI is chosen by default because MME adds ~100 ms |
| Volume | Master gain |
| `Esc` | Panic - releases every voice |
| `Space` | Mute toggle |
| `Ctrl+,` | Detection thresholds (angles, debounce, idle rate) |

Settings persist to `%APPDATA%/visualsynth/settings.json`.

## Layout

```
visualsynth/
  app.py           wiring: audio engine + performance + vision thread + window
  config.py        AppConfig dataclass, JSON persistence
  performance.py   degrees -> notes -> voices; owns the current scale and root
  music/           scale library (scales.json), degree->MIDI mapping, tuning
  audio/           PolyBLEP saw, ADSR, 10-voice synth, sounddevice engine
  vision/          camera, MediaPipe tracker, tracker pool, finger state machine, QThread worker
  ui/              main window, video overlay, note strip, settings dialog, theme
```

Three threads, one direction of flow: the **vision thread** captures and detects, hands note
events straight to the synth's lock-free queue, and emits frames to the **UI thread**; the
**audio callback thread** only ever reads that queue. The audio engine imports no Qt and no
OpenCV, so it is testable with no device attached.

Degrees wrap across octaves, so every scale supports ten fingers regardless of its length:

```
midi = root + intervals[d % n] + 12 * (d // n)
```

On C major that makes degree 8 the octave; on a pentatonic the left hand is simply the
right hand an octave up.

## Tests

```sh
.venv\Scripts\python -m pytest tests
```

115 tests, all headless - synthetic hand landmarks stand in for the camera, and the synth
renders into a buffer instead of a device. Coverage: `music/` 99-100%, `audio/` DSP 95-100%,
`vision/` state machines 96-100%.

## Measured performance on this machine

Intel 20-core laptop CPU, 640x480 @ 30 fps camera, WASAPI output:

| Metric | Measured | Target (requirements.md) |
|---|---|---|
| Detection rate, two hands | 26-30 /s (3 tracker workers) | >= 25 FPS (NFR-2.1) - met |
| MediaPipe inference, per frame | 85-95 ms | - |
| Audio output latency | 22 ms (WASAPI shared, 512-frame blocks) | <= 15 ms (NFR-1.2) - **not met** |
| Synth CPU, 10 voices | 2.5% of one core | - |
| Audio xruns | 0 | 0 (NFR-1.3) - met |
| Note-on latency, end to end | ~150 ms | <= 100 ms (NFR-1.1) - **not met** |

Two honest gaps:

- **Inference is the bottleneck.** MediaPipe 1.0.1's CPU path here is single-threaded and
  costs roughly 42 ms *per hand*, so one tracker instance peaks near 10 detections/s with two
  hands in frame. Measured alternatives that did **not** help: lower capture resolution,
  different detection/tracking confidences, `LIVE_STREAM` mode, and the GPU delegate (not
  compiled into the Windows wheel). MediaPipe 0.10.35 was checked too - the legacy
  `mp.solutions.hands` API with its `model_complexity=0` lite model no longer exists there.
  The fix that did work is `vision/tracker_pool.py`: several tracker instances in threads,
  fed round-robin. The MediaPipe call goes through ctypes, which releases the GIL, so they
  really run in parallel - 8.6/s with one worker, 17.2/s with two, 22-30/s with three.
  That raises throughput; it does not lower the ~90 ms each frame costs, which is why
  note-on latency sits near 150 ms rather than 100 ms.
- **WASAPI shared mode** reports 22 ms of output latency regardless of block size. Exclusive
  mode would be lower but takes the device away from every other app, so it is not the default.

Tuning knobs in `settings.json` if your machine differs: `tracker_workers` (3),
`idle_inference_fps` (12 - detection is throttled while no hand is in frame),
`preview_fps` (30), `block_size` (512).

## Scope

Ten polyphonic saw voices at fixed velocity. Deliberately not included for now: velocity or
dynamics, gesture-driven octave shifting, filters, other waveforms, effects, MIDI, recording,
and `.exe` packaging.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/fetch_models.py` | Downloads the MediaPipe hand model |
| `scripts/demo_scale.py` | Plays a scale through the audio engine - checks sound with no camera |
| `scripts/debug_vision.py` | Prints the live 10-degree finger mask and detection rate |
