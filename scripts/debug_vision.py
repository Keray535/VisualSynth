"""Headless vision check: prints the 10-degree finger mask and FPS.

Usage:  python scripts/debug_vision.py [seconds] [camera_index]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualsynth.vision.camera import Camera
from visualsynth.vision.finger_state import GestureMapper
from visualsynth.vision.hand_tracker import HandTracker


def main(seconds: float, camera_index: int) -> None:
    mapper = GestureMapper()
    frames = 0
    inference_total = 0.0
    started = time.perf_counter()
    last_print = 0.0

    with Camera(index=camera_index) as camera, HandTracker() as tracker:
        print(f"camera {camera_index}: {camera.width}x{camera.height} @ {camera.fps:.0f} fps")
        while time.perf_counter() - started < seconds:
            frame = camera.read()
            now_ms = int(time.monotonic() * 1000)
            if frame is None:
                continue
            observations = tracker.process(frame, now_ms)
            mapper.update(observations, now_ms)
            frames += 1
            inference_total += tracker.last_inference_ms

            elapsed = time.perf_counter() - started
            if elapsed - last_print >= 0.5:
                last_print = elapsed
                active = mapper.active_degrees()
                mask = "".join("#" if d in active else "." for d in range(10))
                hands = ",".join(f"{o.handedness}:{o.score:.2f}" for o in observations) or "-"
                print(f"[{elapsed:5.1f}s] {mask}  hands={hands:24s} inf={tracker.last_inference_ms:5.1f}ms")

    elapsed = time.perf_counter() - started
    print(f"\n{frames} frames in {elapsed:.1f}s -> {frames / elapsed:.1f} FPS, "
          f"mean inference {inference_total / max(1, frames):.1f} ms")


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(duration, index)
