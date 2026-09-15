"""Play a scale through the audio engine - checks sound with no camera involved.

Usage:  python scripts/demo_scale.py ["Scale name"] [root_pitch_class]
Example: python scripts/demo_scale.py "Minor Pentatonic" 9
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualsynth.audio.engine import AudioEngine, resolve_output_config
from visualsynth.config import AppConfig
from visualsynth.music.scales import DEGREE_COUNT
from visualsynth.performance import Performance
from visualsynth.vision.finger_state import NoteEvent

NOTE_S = 0.45
CHORD_S = 1.6


def main(scale_name: str, root_pitch_class: int) -> None:
    config = AppConfig(master_gain=0.4)
    device, rate = resolve_output_config(config.audio_device)
    engine = AudioEngine(settings=config.synth_settings(rate), device=device)
    engine.start()

    performance = Performance(engine.synth, scale_name, root_pitch_class, config.base_octave)
    names = performance.degree_names()
    print(f"{scale_name} from {names[0]} @ {rate} Hz, {engine.output_latency_ms:.0f} ms latency")
    print("degrees:", "  ".join(f"{d + 1}:{n}" for d, n in enumerate(names)))

    try:
        print("\nascending...")
        for degree in range(DEGREE_COUNT):
            performance.handle_events([NoteEvent(degree, True)])
            time.sleep(NOTE_S)
            performance.handle_events([NoteEvent(degree, False)])

        time.sleep(0.4)
        print("chord: degrees 1, 3, 5, 8")
        for degree in (0, 2, 4, 7):
            performance.handle_events([NoteEvent(degree, True)])
        time.sleep(CHORD_S)
        performance.panic()
        time.sleep(0.4)

        print("all ten voices")
        for degree in range(DEGREE_COUNT):
            performance.handle_events([NoteEvent(degree, True)])
        time.sleep(CHORD_S)
        performance.panic()
        time.sleep(0.5)
        print(f"\nxruns: {engine.xruns}")
    finally:
        performance.panic()
        time.sleep(0.3)
        engine.stop()


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Major (Ionian)"
    root = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(name, root)
