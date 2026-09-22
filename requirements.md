# VisualSynth — Requirements

A standalone desktop Python application that watches the user's hands through a webcam and
plays scale degrees as sawtooth-wave tones. Each finger is one degree of a user-selected scale:
right hand thumb→pinky = degrees 1–5, left hand thumb→pinky = degrees 6–10. A finger held
**extended** sustains its note; curling it stops the note.

## Locked design decisions

| Topic | Decision |
|---|---|
| Trigger rule | Finger **extended = note sustains**; curled = silent. Fist = silence. Open hand = 5-note chord. |
| Hand mapping | Right hand = degrees 1–5, left hand = degrees 6–10 |
| Polyphony | Full polyphonic, 10 simultaneous voices |
| Expression | **None for now** — no velocity control, no gesture octave control, no filter |
| Waveform | Wavetable oscillator. Built-in table: saw, sine, square, morphed continuously. Users can import their own `.wav` tables. |
| GUI | PySide6 |
| Distribution | Run from source in a venv |

## Target environment

- Windows 11, Python 3.14.5
- `mediapipe 1.0.1`, `opencv-python 5.0.0.93`, `sounddevice 0.5.6`, `PySide6 6.11.2`, `numpy 2.5.3`

---

## 1. Functional Requirements

### FR-1 Camera capture
- **FR-1.1** App opens a webcam via OpenCV (`cv2.VideoCapture`, `CAP_DSHOW` backend on Windows) at a target 640×480 @ 30 FPS.
- **FR-1.2** The available camera devices are enumerated and selectable at runtime; switching cameras must not restart the app.
- **FR-1.3** Frames are horizontally mirrored (`cv2.flip(frame, 1)`) so the preview behaves like a mirror.
- **FR-1.4** If the camera is missing, busy, or denied, the app stays alive and shows a recoverable error state with a Retry action (see UX-6).

### FR-2 Hand tracking
- **FR-2.1** MediaPipe Hand Landmarker runs in `VIDEO` running-mode with `num_hands=2`, producing 21 landmarks per hand plus a handedness label and score.
- **FR-2.2** Handedness ("Left"/"Right") assigns each detected hand to its degree block: Right → degrees 1–5, Left → degrees 6–10. A `swap_hands` setting inverts this if the camera orientation makes labels wrong.
- **FR-2.3** If two hands of the same reported handedness appear, the one with the higher handedness score keeps the block; the other is ignored.
- **FR-2.4** When a hand is lost for longer than `hand_lost_ms` (default 150 ms), **all notes of that hand are released** — no stuck notes.

### FR-3 Finger extension detection
- **FR-3.1** For index/middle/ring/pinky, extension is decided from the interior joint angle MCP–PIP–TIP.
- **FR-3.2** Hysteresis: a finger becomes *extended* above **160°** and only becomes *curled* below **140°**. This prevents chattering at the threshold.
- **FR-3.3** The thumb uses a combined test: interior angle MCP–IP–TIP above threshold **AND** normalized abduction distance `dist(thumb_tip, index_mcp) / palm_size` above threshold, where `palm_size = dist(wrist, middle_mcp)` (scale-invariant).
- **FR-3.4** Asymmetric debounce: **note-on** fires immediately, **note-off** requires the finger to stay curled for `off_debounce_ms` (default 60 ms), so brief tracking dropouts do not cut a sustained note. The window is measured in milliseconds, not frames, because the detection rate varies with scene complexity.
- **FR-3.5** The detector emits discrete `NoteOn(degree)` / `NoteOff(degree)` events, not per-frame level state, so the audio side is edge-driven.
- **FR-3.6** All thresholds live in a config object and are tunable without code edits.

### FR-4 Scales and pitch mapping
- **FR-4.1** A scale library covers all commonly known scales, stored as data (`visualsynth/data/scales.json`), grouped by family:
  - *Major modes*: Ionian/Major, Dorian, Phrygian, Lydian, Mixolydian, Aeolian/Natural Minor, Locrian
  - *Minor variants*: Harmonic Minor, Melodic Minor (ascending), Harmonic Major
  - *Pentatonic/blues*: Major Pentatonic, Minor Pentatonic, Blues (hexatonic), Major Blues
  - *Symmetric*: Whole Tone, Chromatic, Diminished (H-W), Diminished (W-H)
  - *Exotic/world*: Phrygian Dominant (Hijaz), Double Harmonic (Byzantine), Hungarian Minor, Neapolitan Major, Neapolitan Minor, Hirajoshi, In Sen, Yo, Iwato
  - *Jazz*: Lydian Dominant, Altered (Super Locrian), Bebop Dominant
- **FR-4.2** Each scale is a list of ascending semitone offsets from the root, all `< 12`, first element `0`.
- **FR-4.3** Degree → MIDI note mapping wraps across octaves so every scale supports 10 degrees regardless of its length:
  ```
  midi = root_midi + intervals[d % n] + 12 * (d // n)     # d = 0..9, n = len(intervals)
  ```
  So on Major Pentatonic (n=5) the right hand is one octave and the left hand the next; on Major (n=7) degree 8 is the octave.
- **FR-4.4** Root pitch class (C, C♯ … B) is selectable in the UI. Base octave is a **static config value (default 4, i.e. root C4 = MIDI 60)** and is deliberately *not* gesture-controlled in this version.
- **FR-4.5** Frequency from MIDI: `f = 440 * 2 ** ((midi - 69) / 12)` (A4 = 440 Hz, configurable).
- **FR-4.6** Changing scale or root while notes are sounding retunes/restarts cleanly without clicks.

### FR-5 Audio synthesis
- **FR-5.1** Real-time output through `sounddevice.OutputStream` — `float32`, 48 kHz default, block size configurable (default 512).
- **FR-5.2** **Fixed voice allocation**: 10 voices, voice index = degree index. No voice stealing; a re-triggered degree always reuses its own voice.
- **FR-5.3** Each voice is a **band-limited wavetable oscillator** (see FR-7), so high notes do not alias. Position 0 of the built-in table is a sawtooth, which is what the instrument plays with no configuration.
- **FR-5.4** Each voice has a linear **ADSR** envelope (defaults: A 8 ms, D 60 ms, S 0.7, R 120 ms) to avoid clicks. Velocity is fixed at 1.0.
- **FR-5.5** Mixing: `mix = sum(active_voices) / sqrt(max(1, n_active))`, then a `tanh` soft-clip limiter, then master gain.
- **FR-5.6** Note events cross into the audio callback through a thread-safe queue drained non-blockingly at the top of each callback; the callback allocates nothing per block.
- **FR-5.7** Output device is selectable; the host API is displayed (prefer WASAPI on Windows over MME).
- **FR-5.8** **Panic / All-notes-off**: a button and the `Esc` key immediately release every voice.

### FR-6 Application shell
- **FR-6.1** Entry point `python -m visualsynth` launches the window.
- **FR-6.2** Settings (scale, root, camera index, audio device, master gain, thresholds, swap_hands, wavetable table/position/phase/rand) persist to `%APPDATA%/visualsynth/settings.json` and reload at startup.
- **FR-6.3** Clean shutdown: audio stream closed, camera released, vision thread joined.
- **FR-6.4** A helper script downloads the MediaPipe `hand_landmarker.task` model into `visualsynth/models/`; the app fails with a clear, actionable message if the model is missing.

### FR-7 Wavetable oscillator
- **FR-7.1** The voice source is a wavetable: a stack of 2048-sample single-cycle frames. The built-in table holds three, **in order: saw, sine, square**.
- **FR-7.2** Anti-aliasing is by **mipmap**. Each frame is stored once per octave band with the harmonics that band cannot carry removed; level `k` keeps `1024 >> k` harmonics and a note selects its level from the phase increment alone: `level = clamp(ceil(log2(2048 * inc)), 0, 10)`. No harmonic above Nyquist is ever read.
- **FR-7.3** A **position** in [0, 1] scans the frame stack and linearly blends the two frames it falls between, so a held note morphs continuously rather than stepping between frames.
- **FR-7.4** **Phase** (0–1, shown as 0–360°) sets the start phase of a note. **Rand phase** (0–1, shown as 0–100 %) adds `U(0, rand)` on top at each note-on. Phase restarts when a note begins from silence or from its release tail; a re-trigger of a still-held note keeps its phase, so no click is introduced.
- **FR-7.5** Users can import their own wavetables as `.wav`. PCM (8/16/24/32-bit) and IEEE float (32/64-bit) are supported, including `WAVE_FORMAT_EXTENSIBLE`; stereo is mean-summed to mono and every frame is peak-normalised and DC-removed. A length that divides evenly by 2048 is split into that many frames (capped at 64, evenly spaced); any other length is treated as one cycle and resampled.
- **FR-7.6** Imported files are copied into `%APPDATA%/visualsynth/wavetables/`, so a table survives a restart and a moved source file. A file that will not decode is skipped with a message and never blocks startup.
- **FR-7.7** Table and position changes cross into the audio thread through the same command queue as notes, carrying an already-blended table: the callback only stores a pointer, and the sender keeps the previous table referenced so nothing is freed on the audio thread.

### Out of scope (this version)
Velocity/dynamics, gesture-driven octave shifting, filters, effects, MIDI in/out, recording/export, `.exe` packaging. Wavetable position, phase and rand are set in the UI and are **not** gesture-controlled.

---

## 2. Non-Functional Requirements

### NFR-1 Latency
- **NFR-1.1** Finger-extend → audible note-on: **≤ 100 ms p95**.
- **NFR-1.2** Audio output latency ≤ 15 ms.
- **NFR-1.3** **Zero audio underruns (xruns)** over a 10-minute session; the xrun counter is visible in the UI.

### NFR-2 Throughput & resources
- **NFR-2.1** Detection sustains ≥ 25 detections/s with two hands visible.
- **NFR-2.2** Detection is throttled while no hand is in frame, so an idle app does not burn a core.
- **NFR-2.3** The UI thread never blocks on camera or audio work; frame drops degrade the preview, never the audio.

### NFR-3 Robustness
- **NFR-3.1** No stuck notes under any tracking failure, device unplug, or scale change.
- **NFR-3.2** Camera unplug/replug and audio device change are recoverable at runtime without restarting.
- **NFR-3.3** Unhandled exceptions in the vision thread are caught, logged, and surfaced in the UI — they never kill the audio stream.
- **NFR-3.4** Output signal is always finite and within [-1, 1]; a NaN/Inf guard in the mixer prevents speaker-damaging noise.

### NFR-4 Architecture quality
- **NFR-4.1** Strict layer separation: `vision/`, `music/`, `audio/`, `ui/`. The audio engine has **no** Qt or OpenCV imports; the music layer is pure functions over ints.
- **NFR-4.2** Every layer is testable headlessly — audio renders to a buffer without a device; vision logic runs on synthetic landmark fixtures without a camera.
- **NFR-4.3** Type hints throughout, dataclasses for state/config.
- **NFR-4.4** Test coverage ≥ 80 % on `music/` and `audio/` DSP, ≥ 60 % on `vision/` state machines.

### NFR-5 Portability & setup
- **NFR-5.1** Runs on Windows 11 with Python 3.14 from a venv; no admin rights required.
- **NFR-5.2** Dependencies pinned in `requirements.txt`; setup is 3 documented commands.
- **NFR-5.3** No OS-specific code outside a thin backend-selection layer (camera backend, audio host API).

> Measured results against NFR-1 and NFR-2, including the two targets this hardware does not
> meet and why, are in [README.md](README.md#measured-performance-on-this-machine).

---

## 3. UI/UX Requirements

### UX-1 Window & layout
Single window, default 1100×720, resizable, minimum 900×600, dark theme.

```
┌────────────────────────────────────────────┬────────────────────┐
│                                            │  Scale             │
│           CAMERA PREVIEW (mirrored)        │  [Major        ▾]  │
│           + hand skeleton overlay          │  Root  [C ▾]       │
│           + fingertip degree/note badges   │  ──────────────    │
│                                            │  Camera [0 ▾]      │
│                                            │  Audio  [WASAPI▾]  │
│                                            │  Volume ▬▬▬●──     │
│                                            │  ──────────────    │
│                                            │  FPS 29 | 12 ms    │
│                                            │  xruns 0           │
│                                            │  [ PANIC (Esc) ]   │
├────────────────────────────────────────────┴────────────────────┤
│ LEFT HAND  ⑩G ⑨F ⑧E ⑦D ⑥C  │  ①C ②D ③E ④F ⑤G  RIGHT HAND     │
└──────────────────────────────────────────────────────────────────┘
```

### UX-2 Live video feedback
- **UX-2.1** Preview is mirrored, letterboxed to preserve aspect ratio, ≥ 640×480 rendered.
- **UX-2.2** Hand skeleton drawn as landmark points + bone connections.
- **UX-2.3** Each fingertip carries a badge with the **degree number and note name** (e.g. "③ E4"). A sounding finger's badge is filled and its bone chain thickens; a curled finger's badge is dimmed.
- **UX-2.4** Active state is **never signalled by color alone** — fill, outline weight and the degree number all change (color-blind safe).
- **UX-2.5** Left and right hand get distinct accent colors plus distinct badge shapes (circle vs rounded square).

### UX-3 Note strip
- **UX-3.1** A fixed 10-slot strip along the bottom, left hand on the left and right hand on the right, mirroring the on-screen hand positions. Within each hand the thumb sits nearest the centre, as the fingers do on screen.
- **UX-3.2** Each slot shows degree number + resolved note name (e.g. "⑦ D5") and lights while its voice is sounding.
- **UX-3.3** Slot labels update immediately when scale or root changes, even with no hand visible.

### UX-4 Controls
- **UX-4.1** Scale selector is a combo box **grouped by family** with type-to-search.
- **UX-4.2** Root note selector: C, C♯/D♭ … B.
- **UX-4.3** Camera and audio-output device selectors list friendly device names; audio also shows host API.
- **UX-4.4** Master volume slider.
- **UX-4.5** Keyboard: `Esc` = panic, `Space` = mute toggle, `Ctrl+,` = detection settings, `Ctrl+T` = show / hide the wavetable panel.
- **UX-4.6** Every control change takes effect immediately; nothing requires a restart.

### UX-5 Status & diagnostics
Always-visible compact readout: capture FPS, detection rate, inference time (ms), audio buffer size, xrun count, hands detected, active voices. Out-of-band values turn into a warning style.

### UX-6 Empty, error & guidance states
- **UX-6.1** No camera / permission denied → overlay card: cause, one concrete fix, **Retry** (the vision thread retries automatically). Audio keeps working.
- **UX-6.2** No hand detected for > 2 s → soft hint: "Show your hand to the camera".
- **UX-6.3** Low landmark confidence → hint: "Low tracking confidence — improve lighting".
- **UX-6.4** Missing model file → message with the exact command to fetch it.
- **UX-6.5** Audio device lost → automatic fallback to the default device.

### UX-7 First-run experience
- **UX-7.1** Cold start to first preview frame < 3 s.
- **UX-7.2** First launch shows a one-time hint over the preview: fingers = degrees, right = 1–5, left = 6–10.
- **UX-7.3** Defaults are playable with zero configuration: C Major, camera 0, default output device.

### UX-8 Wavetable panel
- **UX-8.1** A panel **fixed into the main window** under the preview, on a draggable split, so a held chord can be heard morphing while it is edited and nothing covers the preview. It is never a separate window and cannot be detached; `Ctrl+T` (or the panel button) only hides and shows it, and that choice persists in settings.
- **UX-8.2** The frame stack is drawn in table order, receding up and to the right, with the sounding blended cycle highlighted at the depth the current position sits at. A table with more frames than fits legibly is thinned to an evenly spaced subset.
- **UX-8.3** Position slider with a readout naming the frames it is between (`saw`, or `saw → sine 37%`).
- **UX-8.4** Phase slider, 0–360°. The drawn cycle rotates with it and a marker shows where a note starts, labelled in degrees so the cue is not position-only.
- **UX-8.5** Rand slider, 0–100 %, drawn as a range strip from the start marker to the furthest a note-on can land.
- **UX-8.6** Table selector lists the built-in table and every imported one; `Load .wav...` imports a new one and switches to it.
- **UX-8.7** An import that fails shows the reason inline; the current table keeps sounding.
- **UX-8.8** Reset returns table, position, phase and rand to their defaults.
