"""Download the MediaPipe hand landmark model into visualsynth/models/ (FR-6.4).

Usage:  python scripts/fetch_models.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "visualsynth" / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def fetch(force: bool = False) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force:
        print(f"already present: {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes)")
        return MODEL_PATH
    print(f"downloading {MODEL_URL}")
    tmp = MODEL_PATH.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
        data = response.read()
    if len(data) < 1_000_000:
        raise RuntimeError(f"suspiciously small download: {len(data)} bytes")
    tmp.write_bytes(data)
    tmp.replace(MODEL_PATH)
    digest = hashlib.sha256(data).hexdigest()[:16]
    print(f"saved {MODEL_PATH} ({len(data):,} bytes, sha256:{digest}...)")
    return MODEL_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    try:
        fetch(force=args.force)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"failed: {exc}", file=sys.stderr)
        sys.exit(1)
