# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the repo root with the venv interpreter (Windows paths shown; the venv is `.venv/`):

```sh
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\fetch_models.py     # downloads visualsynth/models/hand_landmarker.task (gitignored, 7.8 MB)
.venv\Scripts\python -m visualsynth              # run the app
.venv\Scripts\python -m pytest tests             # full suite (191 tests, all headless)
```

Single test / single file:

```sh
.venv\Scripts\python -m pytest tests/test_vision.py
.venv\Scripts\python -m pytest tests/test_vision.py::test_degenerate_angle_is_zero_not_nan
.venv\Scripts\python -m pytest tests -k "debounce"
.venv\Scripts\python -m pytest tests --cov=visualsynth --cov-report=term-missing
```

`pytest.ini` sets `pythonpath = . tests`, so tests import both `visualsynth.*` and the bare
`hand_fixtures` module.

Hardware smoke checks (need a real device, not part of the suite):

```sh
.venv\Scripts\python scripts\demo_scale.py "Minor Pentatonic" 9   # audio only, no camera
.venv\Scripts\python scripts\debug_vision.py 20 0                 # camera only, prints the degree mask
```

There is no linter or formatter configured.

## Architecture

Three threads, one direction of flow:

```
vision thread (VisionWorker, QThread)
  camera.read -> TrackerPool.submit/poll -> GestureMapper.update -> [NoteEvent]
       |                                                                 |
       | frameReady signal (FrameUpdate)                                 v
       v                                          Performance.handle_events -> Synth queue
  UI thread (MainWindow)                                                       |
                                                        audio callback thread -+-> Synth.process
```

- The vision thread calls `Performance.handle_events` **directly** (not via a Qt signal) — that is
  the shortest path to sound. Only frames and status go to the UI as signals.
- The audio callback thread never touches voice state directly. `Synth` exposes
  `note_on/note_off/retune/panic/set_master_gain` from any thread; those push onto a
  `queue.SimpleQueue` drained at the top of `Synth.process`. The callback allocates nothing
  per block (preallocated `_mix` / `_scratch` / `_mono` buffers).

### Layer rules (NFR-4.1, enforced by convention — keep them)

- `music/` — pure functions over ints/floats. No numpy state, no I/O.
- `audio/` — DSP; **no Qt, no OpenCV imports**. `engine.py` is the only file touching hardware;
  everything in `synth.py`/`voice.py`/`wavetable.py`/`oscillator.py`/`envelope.py` renders into a
  buffer and is tested with no device attached. `wavfile.py` is stdlib + numpy only.
- `vision/` — `landmarks.py` and `finger_state.py` are pure geometry/state machines over numpy
  arrays and are tested against synthetic landmarks (`tests/hand_fixtures.py`); `camera.py`,
  `hand_tracker.py`, `tracker_pool.py`, `worker.py` touch OpenCV/MediaPipe/Qt.
- `ui/` — Qt only; reads `Performance` and `AudioEngine`, never DSP internals. `WavetablePanel`
  is a child of `MainWindow`, not a window: `Ctrl+T` toggles its visibility (`wavetable_panel_visible`
  in the config) and there is no detach path to keep alive.
- `performance.py` is the seam: vision speaks only in **degrees**, audio hears only **frequencies**.
- `wavetable_library.py` is the same kind of seam for the oscillator: the UI names a table, a
  position and a phase, and gets plain numpy arrays back for drawing. It never touches a `Voice`.

### Degree indexing

Degrees are **0-based internally** (0–4 right hand thumb→pinky, 5–9 left hand), and displayed as
1–10 in the UI and docs. `DEGREE_COUNT = 10` lives in `music/scales.py`. Voice index == degree
index, so there is no voice-stealing logic — a re-triggered degree always reuses its own voice.

Degrees wrap by octave so any scale length supports ten fingers:

```
midi = root_midi + intervals[d % n] + 12 * (d // n)
```

### Non-obvious constraints

- **MediaPipe VIDEO mode needs strictly increasing timestamps.** `VisionWorker._timestamp_ms`
  guarantees this by bumping any non-increasing value.
- **Pool results finish out of order.** `TrackerPool.poll()` sorts by timestamp, and the caller
  must drop anything older than the newest result already applied (`last_applied_ms` in
  `VisionWorker.run`). Do not remove that check.
- **`TrackerPool` exists because MediaPipe CPU inference is single-threaded** (~42 ms per hand).
  Its ctypes call releases the GIL, so Python threads really parallelize: ~8.6/s with one worker,
  ~22/s with three. Adding workers raises throughput but not per-frame latency.
- **Never leave a stuck note.** Every failure path releases: camera error, tracker stall
  (`_age_hands` vs `hand_lost_ms`), thread shutdown, hand swap, scale change. When adding a new
  exit path, call `mapper.release_all()` / `performance.panic()`.
- **Hysteresis + asymmetric debounce** in `finger_state.py`: extend above 160°, curl below 140°;
  note-on immediate, note-off after `off_debounce_ms`. Debounce is measured in **milliseconds, not
  frames**, because detection rate swings with scene complexity. The debounce window starts at the
  *previous* observation, not the current frame.
- **Landmark geometry uses `landmarks_px` (aspect-corrected), never `landmarks_norm`.** The
  normalized array is for drawing only; measuring angles on it skews them by the frame aspect ratio.
- **The synth mixer must stay finite**: `nan_to_num` + `clip` at the end of `Synth.process`
  (NFR-3.4) — a NaN reaching the device is a speaker hazard.
- **Audio sample rate follows the device**, because WASAPI shared mode only accepts the Windows
  mixer rate. `resolve_output_config()` returns `(device_index, sample_rate)`; build
  `SynthSettings` from that, not from a hardcoded 48 kHz.
- **Host API matters more than the device** on Windows (MME ~100 ms vs WASAPI ~10 ms), which is why
  `preferred_output_device()` matches on host-API defaults rather than device names — MME truncates
  names to 31 characters.
- **Wavetable views are built on the control thread, never in the callback.** Blending two
  frames' mip stacks allocates, so `WavetableController` does it and pushes the finished
  `WavetableView` through the synth's command queue; `_apply` only stores the pointer. The
  controller keeps a reference to the installed view so the audio thread's overwrite never drops
  the last reference and frees memory inside the callback (FR-7.7).
- **Mip level selection is `ceil(log2(FRAME_SIZE * inc))`**, clamped to `[0, MIP_LEVELS - 1]`.
  That is the smallest octave band that keeps every harmonic under Nyquist; loosening it aliases
  and tightening it dulls the bass. Mip levels are deliberately **not** renormalised — a
  band-limited copy is quieter, and rescaling would make octave crossings jump in loudness.
- **`Voice.note_on` resets phase when the envelope is `IDLE` or `RELEASE`**, not when it is
  merely inactive. A released note stays `active` for its whole 120 ms release tail, which is the
  normal state a degree is re-struck from, so testing `active` would swallow nearly every phase
  reset and make Phase/Rand inaudible (FR-7.4).
- **The stdlib `wave` module is not usable here.** It raises on IEEE-float WAVs
  (`unknown format: 3`) and misreads `WAVE_FORMAT_EXTENSIBLE` as PCM without checking the
  SubFormat GUID — which is exactly what a 32-bit float wavetable export is. `audio/wavfile.py`
  parses RIFF directly instead; do not "simplify" it back to `wave`.
- **`AppConfig.from_dict` is deliberately tolerant** of missing and unknown keys so an old
  `settings.json` still loads after the config grows. Adding a nested dataclass field requires an
  entry in `_nested_type()`. Settings live at `%APPDATA%/visualsynth/settings.json` and are saved
  on window close.

## Requirements traceability

`requirements.md` is the spec, and code comments/docstrings cite its IDs (`FR-3.4`, `NFR-1.2`,
`UX-2.3`). Keep that convention when adding behaviour — cite the requirement the code satisfies, and
update `requirements.md` if the behaviour is new. Two targets are knowingly unmet on the reference
machine (audio output latency, end-to-end note-on latency); README.md's "Measured performance"
section explains why and what was already tried, so don't re-litigate those without new measurements.

Out of scope by decision: velocity/dynamics, gesture octave shifting, filters, effects, MIDI,
recording, `.exe` packaging. Wavetable position/phase/rand are UI controls, not gesture-mapped.

## Testing conventions

- Everything is headless. Vision tests build synthetic hands at requested joint angles via
  `tests/hand_fixtures.py` (`make_hand`, `open_hand`, `fist`); tracker tests stub `HandTracker` with
  a `FakeTracker` that sleeps like inference does; audio tests render into numpy buffers and assert
  on spectra (e.g. aliasing ratio for the PolyBLEP saw).
- New vision logic belongs in `finger_state.py`/`landmarks.py` where it can be tested without
  MediaPipe; keep `hand_tracker.py` a thin adapter.
- Wavetable tests build WAV files as raw RIFF bytes (`tests/wav_fixtures.py`) because `wave` can
  only write PCM, and the float/extensible paths are the ones most likely to break.
- There are no UI tests. The wavetable panel is checked by hand with
  `QT_QPA_PLATFORM=offscreen` and `QWidget.grab()` when its painting changes.
