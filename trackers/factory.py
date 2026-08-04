from trackers.base import BaseTracker
from trackers.histogram import MeanShiftTracker, CBWHMeanShiftTracker, CamShiftTracker
from trackers.soamst import SOAMSTTracker
from trackers.sift import SIFTTracker
from trackers.kcf import KCFTracker
from trackers.hybrid import HybridTracker

import cv2


TRACKERS = ["MeanShift", "CBWH-MeanShift", "CAMShift", "SOAMST", "SIFT", "MeanShift+SIFT", "KCF"]

DISPLAY_NAMES = {
    "MeanShift": "MeanShift",
    "CBWH-MeanShift": "CBWH-MeanShift",
    "CAMShift": "CAMShift",
    "SOAMST": "SOAMST",
    "SIFT": "SIFT",
    "MeanShift+SIFT": "MeanShift+SIFT",
    "KCF": "KCF",
}

TRACKER_IMPLEMENTATION_VERSIONS = {
    "MeanShift": "v1",
    "CBWH-MeanShift": "v1",
    "CAMShift": "v2-constrained-size",
    "SOAMST": "v1-moment-rgb16",
    "SIFT": "v2-local-affine",
    "MeanShift+SIFT": "v2-local-affine",
    "KCF": "v1",
}


def validate_tracker_environment() -> None:
    """Fail fast before a long benchmark if required OpenCV modules are missing."""
    missing: list[str] = []
    if not hasattr(cv2, "SIFT_create"):
        missing.append("SIFT")
    if missing:
        raise RuntimeError(
            "Missing OpenCV tracker support: " + ", ".join(missing) + ". "
            "Activate the project virtual environment, remove conflicting OpenCV wheels, "
            "and install requirements.txt."
        )

def build_tracker(name: str) -> BaseTracker:
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "meanshift":
        return MeanShiftTracker()
    if normalized in {"cbwh", "cbwh-meanshift", "cbwh+meanshift"}:
        return CBWHMeanShiftTracker()
    if normalized == "camshift":
        return CamShiftTracker()
    if normalized == "soamst":
        return SOAMSTTracker()
    if normalized == "sift":
        return SIFTTracker()
    if normalized == "kcf":
        return KCFTracker()
    if normalized in {"hybrid", "meanshift+sift", "meanshift-sift"}:
        return HybridTracker()
    raise ValueError(f"Unknown tracker: {name}")
