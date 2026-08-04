from trackers.base import BBox, TrackResult, clip_bbox, BaseTracker
from trackers.histogram import MeanShiftTracker
from trackers.sift import SIFTTracker

from typing import Optional

import cv2
import numpy as np


class HybridTracker(BaseTracker):
    name = "MeanShift+SIFT"

    def __init__(self, sift_interval: int = 5, confidence_threshold: float = 0.65) -> None:
        self.mean_shift = MeanShiftTracker()
        self.sift = SIFTTracker()
        self.sift_interval = int(sift_interval)
        self.confidence_threshold = float(confidence_threshold)
        self.frame_index = 0
        self.last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self.mean_shift.initialize(frame, bbox)
        self.sift.initialize(frame, bbox)
        self.frame_index = 0
        self.last_bbox = clip_bbox(bbox, frame.shape)

    def update(self, frame: np.ndarray) -> TrackResult:
        self.frame_index += 1
        previous_bbox = self.last_bbox
        ms_result = self.mean_shift.update(frame)
        confidence = float(ms_result.info.get("hist_similarity", 0.0))
        run_sift = (
            self.frame_index % self.sift_interval == 0
            or confidence < self.confidence_threshold
        )

        sift_result: Optional[TrackResult] = None
        if run_sift:
            # A reliable MeanShift estimate narrows the SIFT search region.  When
            # histogram confidence is low, retain the hybrid tracker's previous
            # position so that SIFT can expand its own local/global search rather
            # than being forced around a potentially drifted MeanShift window.
            hint = ms_result.bbox if confidence >= 0.35 else previous_bbox
            self.sift.set_search_hint(hint, frame.shape)
            sift_result = self.sift.update(frame)
            if sift_result.ok:
                self.mean_shift.set_window(sift_result.bbox, frame.shape)
                self.last_bbox = sift_result.bbox
                return TrackResult(
                    bbox=sift_result.bbox,
                    ok=True,
                    info={
                        "source": str(sift_result.info.get("source", "sift_relocalization")),
                        "hist_similarity": confidence,
                        "matches": int(sift_result.info.get("matches", 0)),
                        "inliers": int(sift_result.info.get("inliers", 0)),
                        "inlier_ratio": float(sift_result.info.get("inlier_ratio", 0.0)),
                    },
                )

        self.last_bbox = ms_result.bbox
        return TrackResult(
            bbox=ms_result.bbox,
            ok=ms_result.ok,
            info={
                "source": "meanshift",
                "hist_similarity": confidence,
                "sift_triggered": run_sift,
                "sift_ok": bool(sift_result.ok) if sift_result is not None else False,
                "matches": int(sift_result.info.get("matches", 0)) if sift_result else 0,
                "inliers": int(sift_result.info.get("inliers", 0)) if sift_result else 0,
            },
        )


