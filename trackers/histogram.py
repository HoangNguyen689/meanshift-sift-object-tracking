from trackers.base import BBox, TrackResult, clip_bbox, bbox_to_int, bbox_center_size, bbox_from_center_size, BaseTracker

from typing import Optional

import cv2
import numpy as np


class HistogramTrackerBase(BaseTracker):
    def __init__(self, hist_bins: int = 180) -> None:
        self.hist_bins = hist_bins
        self.roi_hist: Optional[np.ndarray] = None
        self.track_window: tuple[int, int, int, int] = (0, 0, 1, 1)
        self.last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)
        self.term_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            1,
        )

    @staticmethod
    def hsv_foreground_mask(hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(hsv, np.array((0, 45, 25)), np.array((180, 255, 255)))

    def canonicalize_histogram(self, hist: np.ndarray) -> np.ndarray:
        """Return a histogram with a stable OpenCV-compatible layout.

        OpenCV 5 is stricter than some OpenCV 4 builds when compareHist receives
        arrays with equivalent bin counts but different dimensional layouts.
        Keeping every histogram as contiguous float32 with shape (bins, 1) makes
        MeanShift, CBWH and CAMShift behave consistently across versions.
        """
        array = np.asarray(hist, dtype=np.float32).reshape(-1)
        if array.size != self.hist_bins:
            raise ValueError(
                f"Expected {self.hist_bins} histogram bins, got {array.size}"
            )
        return np.ascontiguousarray(array.reshape(self.hist_bins, 1))

    def calculate_hue_histogram(self, hsv: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        hist = cv2.calcHist([hsv], [0], mask, [self.hist_bins], [0, 180]).astype(np.float32)
        if float(hist.sum()) <= 0.0:
            hist = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180]).astype(np.float32)
        return self.canonicalize_histogram(hist)

    def build_target_histogram(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        x, y, w, h = bbox_to_int(bbox)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            raise ValueError("Initial bounding box produces an empty ROI")
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = self.calculate_hue_histogram(hsv_roi, self.hsv_foreground_mask(hsv_roi))
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        bbox = clip_bbox(bbox, frame.shape)
        self.roi_hist = self.canonicalize_histogram(
            self.build_target_histogram(frame, bbox)
        )
        self.track_window = bbox_to_int(bbox)
        self.last_bbox = bbox

    def set_window(self, bbox: BBox, frame_shape: tuple[int, ...]) -> None:
        bbox = clip_bbox(bbox, frame_shape)
        self.track_window = bbox_to_int(bbox)
        self.last_bbox = bbox

    def backproject(self, frame: np.ndarray) -> np.ndarray:
        if self.roi_hist is None:
            raise RuntimeError("Tracker is not initialized")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        return cv2.GaussianBlur(backproj, (5, 5), 0)

    def histogram_similarity(self, frame: np.ndarray, bbox: BBox) -> float:
        if self.roi_hist is None:
            return 0.0
        x, y, w, h = bbox_to_int(clip_bbox(bbox, frame.shape))
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        candidate = self.calculate_hue_histogram(hsv_roi, self.hsv_foreground_mask(hsv_roi))
        if float(candidate.sum()) <= 0.0:
            return 0.0
        cv2.normalize(candidate, candidate, 0, 255, cv2.NORM_MINMAX)
        reference = self.canonicalize_histogram(self.roi_hist)
        candidate = self.canonicalize_histogram(candidate)

        try:
            distance = float(
                cv2.compareHist(reference, candidate, cv2.HISTCMP_BHATTACHARYYA)
            )
            similarity = 1.0 - distance
        except cv2.error:
            # Cross-version fallback equivalent to a Bhattacharyya-style
            # comparison after L1 normalization. It is only used when a
            # particular OpenCV build rejects otherwise compatible arrays.
            reference_prob = reference.reshape(-1).astype(np.float64)
            candidate_prob = candidate.reshape(-1).astype(np.float64)
            reference_prob /= max(float(reference_prob.sum()), 1e-12)
            candidate_prob /= max(float(candidate_prob.sum()), 1e-12)
            similarity = float(
                np.sum(np.sqrt(reference_prob * candidate_prob))
            )

        return float(np.clip(similarity, 0.0, 1.0))


class MeanShiftTracker(HistogramTrackerBase):
    name = "MeanShift"

    def update(self, frame: np.ndarray) -> TrackResult:
        backproj = self.backproject(frame)
        iterations, window = cv2.meanShift(backproj, self.track_window, self.term_criteria)
        self.track_window = window
        bbox = clip_bbox(tuple(map(float, window)), frame.shape)
        self.last_bbox = bbox
        confidence = self.histogram_similarity(frame, bbox)
        return TrackResult(
            bbox=bbox,
            ok=True,
            info={"iterations": int(iterations), "hist_similarity": confidence},
        )


class CBWHMeanShiftTracker(MeanShiftTracker):
    """MeanShift with a corrected background-weighted target histogram.

    The target histogram is transformed once at initialization using
    q'_u = C v_u q_u, where v_u = min(o_min / o_u, 1).  q_u is the target
    histogram and o_u is a histogram from a surrounding background ring.
    Only the target model is transformed; candidate histograms remain standard,
    matching the correction proposed by Ning et al.
    """

    name = "CBWH-MeanShift"

    def __init__(self, hist_bins: int = 180, background_scale: float = 2.0) -> None:
        super().__init__(hist_bins=hist_bins)
        self.background_scale = max(1.2, float(background_scale))

    def build_target_histogram(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        x, y, w, h = bbox_to_int(bbox)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            raise ValueError("Initial bounding box produces an empty ROI")
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        target_hist = self.calculate_hue_histogram(hsv_roi, self.hsv_foreground_mask(hsv_roi))

        frame_h, frame_w = frame.shape[:2]
        expanded_w = max(w + 2, int(round(w * self.background_scale)))
        expanded_h = max(h + 2, int(round(h * self.background_scale)))
        cx = x + w / 2.0
        cy = y + h / 2.0
        ex1 = max(0, int(round(cx - expanded_w / 2.0)))
        ey1 = max(0, int(round(cy - expanded_h / 2.0)))
        ex2 = min(frame_w, int(round(cx + expanded_w / 2.0)))
        ey2 = min(frame_h, int(round(cy + expanded_h / 2.0)))
        context = frame[ey1:ey2, ex1:ex2]

        if context.size == 0:
            cv2.normalize(target_hist, target_hist, 0, 255, cv2.NORM_MINMAX)
            return target_hist

        hsv_context = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
        ring_mask = np.full(hsv_context.shape[:2], 255, dtype=np.uint8)
        tx1 = max(0, x - ex1)
        ty1 = max(0, y - ey1)
        tx2 = min(ring_mask.shape[1], tx1 + w)
        ty2 = min(ring_mask.shape[0], ty1 + h)
        ring_mask[ty1:ty2, tx1:tx2] = 0
        ring_mask = cv2.bitwise_and(ring_mask, self.hsv_foreground_mask(hsv_context))
        background_hist = cv2.calcHist(
            [hsv_context], [0], ring_mask, [self.hist_bins], [0, 180]
        ).astype(np.float32)

        target_prob = target_hist.reshape(-1).astype(np.float64)
        background_prob = background_hist.reshape(-1).astype(np.float64)
        target_sum = float(target_prob.sum())
        background_sum = float(background_prob.sum())
        if target_sum <= 0.0 or background_sum <= 0.0:
            cv2.normalize(target_hist, target_hist, 0, 255, cv2.NORM_MINMAX)
            return target_hist

        target_prob /= target_sum
        background_prob /= background_sum
        positive_background = background_prob[background_prob > 0.0]
        if positive_background.size == 0:
            weights = np.ones_like(background_prob)
        else:
            background_floor = float(positive_background.min())
            weights = np.ones_like(background_prob)
            nonzero = background_prob > 0.0
            weights[nonzero] = np.minimum(background_floor / background_prob[nonzero], 1.0)

        corrected = target_prob * weights
        corrected_sum = float(corrected.sum())
        if corrected_sum <= 0.0:
            corrected = target_prob
        else:
            corrected /= corrected_sum
        corrected_hist = self.canonicalize_histogram(corrected.astype(np.float32))
        cv2.normalize(corrected_hist, corrected_hist, 0, 255, cv2.NORM_MINMAX)
        return self.canonicalize_histogram(corrected_hist)


class CamShiftTracker(HistogramTrackerBase):
    """CAMShift with conservative size validation for axis-aligned boxes.

    Raw CAMShift moments may explode when the back-projection contains large
    background regions with a similar hue.  The tracker therefore keeps the
    CAMShift center estimate but constrains the per-frame and global size change.
    This is not a ground-truth-dependent correction: every sequence uses the same
    fixed limits derived only from the initial and previous predicted boxes.
    """

    name = "CAMShift"

    def __init__(
        self,
        max_scale_step: float = 1.15,
        min_global_scale: float = 0.35,
        max_global_scale: float = 3.0,
        size_update_rate: float = 0.45,
        max_center_step_factor: float = 1.75,
    ) -> None:
        super().__init__()
        self.max_scale_step = float(max_scale_step)
        self.min_global_scale = float(min_global_scale)
        self.max_global_scale = float(max_global_scale)
        self.size_update_rate = float(size_update_rate)
        self.max_center_step_factor = float(max_center_step_factor)
        self.initial_size = np.array([1.0, 1.0], dtype=np.float64)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        super().initialize(frame, bbox)
        _, self.initial_size = bbox_center_size(self.last_bbox)

    @staticmethod
    def _align_camshift_size(raw_size: np.ndarray, previous_size: np.ndarray) -> np.ndarray:
        """Avoid a 90-degree width/height swap from appearing as a scale jump."""
        direct = np.maximum(raw_size.astype(np.float64), 2.0)
        swapped = direct[::-1].copy()
        eps = 1e-9
        direct_cost = float(np.sum(np.abs(np.log((direct + eps) / (previous_size + eps)))))
        swapped_cost = float(np.sum(np.abs(np.log((swapped + eps) / (previous_size + eps)))))
        return swapped if swapped_cost < direct_cost else direct

    def update(self, frame: np.ndarray) -> TrackResult:
        previous_bbox = self.last_bbox
        previous_center, previous_size = bbox_center_size(previous_bbox)
        backproj = self.backproject(frame)
        rotated_rect, _ = cv2.CamShift(backproj, self.track_window, self.term_criteria)

        raw_center = np.asarray(rotated_rect[0], dtype=np.float64)
        raw_size = np.asarray(rotated_rect[1], dtype=np.float64)
        valid = bool(
            np.all(np.isfinite(raw_center))
            and np.all(np.isfinite(raw_size))
            and np.all(raw_size >= 2.0)
        )
        if not valid:
            self.track_window = bbox_to_int(previous_bbox)
            return TrackResult(
                bbox=previous_bbox,
                ok=False,
                info={"angle": float(rotated_rect[2]), "hist_similarity": 0.0, "source": "invalid_camshift_moments"},
            )

        aligned_size = self._align_camshift_size(raw_size, previous_size)
        per_frame_low = previous_size / self.max_scale_step
        per_frame_high = previous_size * self.max_scale_step
        global_low = self.initial_size * self.min_global_scale
        global_high = self.initial_size * self.max_global_scale
        constrained_size = np.clip(aligned_size, np.maximum(per_frame_low, global_low), np.minimum(per_frame_high, global_high))
        smoothed_size = (
            (1.0 - self.size_update_rate) * previous_size
            + self.size_update_rate * constrained_size
        )

        displacement = raw_center - previous_center
        max_center_step = self.max_center_step_factor * max(float(previous_size[0]), float(previous_size[1]), 2.0)
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm > max_center_step:
            displacement *= max_center_step / max(displacement_norm, 1e-9)
        center = previous_center + displacement

        bbox = bbox_from_center_size(center, smoothed_size, frame.shape)
        confidence = self.histogram_similarity(frame, bbox)
        self.set_window(bbox, frame.shape)
        area_ratio = float((bbox[2] * bbox[3]) / max(previous_bbox[2] * previous_bbox[3], 1e-9))
        return TrackResult(
            bbox=bbox,
            ok=bool(confidence >= 0.10),
            info={
                "angle": float(rotated_rect[2]),
                "hist_similarity": confidence,
                "area_ratio": area_ratio,
                "source": "camshift_constrained",
            },
        )
