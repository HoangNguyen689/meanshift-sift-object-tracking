from __future__ import annotations

import sys

import cv2

from trackers import validate_tracker_environment


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    validate_tracker_environment()
    print("OK: MeanShift, CBWH-MeanShift, CAMShift, SOAMST, SIFT, MeanShift+SIFT and KCF are available.")


if __name__ == "__main__":
    main()
