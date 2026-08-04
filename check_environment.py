from __future__ import annotations

import sys

import cv2
import numpy as np

from trackers import TRACKERS, build_tracker, validate_tracker_environment


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    validate_tracker_environment()

    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    ok: list[str] = []
    failed: list[str] = []
    for name in TRACKERS:
        try:
            t = build_tracker(name)
            t.initialize(dummy, (20.0, 20.0, 24.0, 24.0))
            t.update(dummy)
            ok.append(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")

    if failed:
        print("FAIL: some trackers could not be instantiated or updated:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)

    print(f"OK: {', '.join(ok)} are available and functional.")


if __name__ == "__main__":
    main()
