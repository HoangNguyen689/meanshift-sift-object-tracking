from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

BBox = tuple[float, float, float, float]


def clip_bbox(bbox: BBox, frame_shape: tuple[int, ...], min_size: float = 2.0) -> BBox:
    """Clip x, y, w, h to the image while keeping a non-empty rectangle."""
    height, width = frame_shape[:2]
    x, y, w, h = map(float, bbox)
    x = min(max(x, 0.0), max(width - min_size, 0.0))
    y = min(max(y, 0.0), max(height - min_size, 0.0))
    w = min(max(w, min_size), max(width - x, min_size))
    h = min(max(h, min_size), max(height - y, min_size))
    return x, y, w, h


def bbox_to_int(bbox: BBox) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))


def polygon_to_bbox(points: np.ndarray, frame_shape: tuple[int, ...]) -> Optional[BBox]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4 or not np.all(np.isfinite(pts)):
        return None
    area = abs(float(cv2.contourArea(pts)))
    if area < 25.0:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 3 or h < 3:
        return None
    frame_h, frame_w = frame_shape[:2]
    if w > frame_w * 1.25 or h > frame_h * 1.25:
        return None
    return clip_bbox((float(x), float(y), float(w), float(h)), frame_shape)


@dataclass
class TrackResult:
    bbox: BBox
    ok: bool
    info: dict[str, Any]


@dataclass
class FeatureModel:
    keypoints: list[Any]
    descriptors: np.ndarray
    corners: np.ndarray


def bbox_center_size(bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = map(float, bbox)
    return (
        np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64),
        np.array([w, h], dtype=np.float64),
    )


def bbox_from_center_size(
    center: np.ndarray,
    size: np.ndarray,
    frame_shape: tuple[int, ...],
) -> BBox:
    width = max(2.0, float(size[0]))
    height = max(2.0, float(size[1]))
    return clip_bbox(
        (float(center[0] - width / 2.0), float(center[1] - height / 2.0), width, height),
        frame_shape,
    )


def expand_bbox(bbox: BBox, factor: float, frame_shape: tuple[int, ...]) -> BBox:
    center, size = bbox_center_size(bbox)
    expanded = np.maximum(size * max(1.0, float(factor)), 8.0)
    return bbox_from_center_size(center, expanded, frame_shape)


class BaseTracker:
    name = "base"

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        raise NotImplementedError

    def update(self, frame: np.ndarray) -> TrackResult:
        raise NotImplementedError


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


class SOAMSTTracker(BaseTracker):
    """Scale and Orientation Adaptive Mean Shift Tracking.

    This implementation follows the moment-based procedure described by Ning
    et al. The target is represented by a 16 x 16 x 16 RGB histogram. For each
    frame, a candidate ellipse slightly larger than the previous target is used
    to build the candidate model. Pixel weights are computed as
    sqrt(q_u / p_u), the center is updated by the first-order moments, and the
    zeroth/second-order moments together with the Bhattacharyya coefficient are
    used to estimate area, axis lengths and orientation.

    Fixed numerical guards are applied uniformly to every sequence: bounded
    per-frame scale change, bounded global scale, covariance regularization and
    temporal smoothing. These guards do not use ground truth and prevent a
    nearly singular moment matrix from collapsing or exploding the box.
    """

    name = "SOAMST"

    def __init__(
        self,
        bins_per_channel: int = 16,
        area_sigma: float = 1.0,
        candidate_margin: float = 5.0,
        epsilon: float = 0.1,
        max_iterations: int = 15,
        max_scale_step: float = 1.20,
        min_global_scale: float = 0.35,
        max_global_scale: float = 3.0,
        shape_update_rate: float = 0.55,
        angle_update_rate: float = 0.65,
        max_weight: float = 12.0,
    ) -> None:
        self.bins_per_channel = int(bins_per_channel)
        self.hist_bins = self.bins_per_channel ** 3
        self.area_sigma = max(float(area_sigma), 1e-6)
        self.candidate_margin = max(float(candidate_margin), 0.0)
        self.epsilon = max(float(epsilon), 0.0)
        self.max_iterations = max(1, int(max_iterations))
        self.max_scale_step = max(float(max_scale_step), 1.001)
        self.min_global_scale = max(float(min_global_scale), 0.05)
        self.max_global_scale = max(float(max_global_scale), self.min_global_scale)
        self.shape_update_rate = float(np.clip(shape_update_rate, 0.0, 1.0))
        self.angle_update_rate = float(np.clip(angle_update_rate, 0.0, 1.0))
        self.max_weight = max(float(max_weight), 1.0)

        self.target_prob = np.full(self.hist_bins, 1.0 / self.hist_bins, dtype=np.float64)
        self.center = np.array([0.0, 0.0], dtype=np.float64)
        self.axes = np.array([1.0, 1.0], dtype=np.float64)  # semi-major, semi-minor
        self.initial_axes = self.axes.copy()
        self.angle_deg = 0.0
        self.area_calibration = 1.0
        self.last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def _feature_indices(self, image: np.ndarray) -> np.ndarray:
        shift = 8 - int(round(np.log2(self.bins_per_channel)))
        if shift < 0 or (1 << (8 - shift)) != self.bins_per_channel:
            # The project uses 16 bins per channel. Keep a generic fallback for
            # other values without changing the public constructor contract.
            quantized = np.minimum(
                (image.astype(np.int32) * self.bins_per_channel) // 256,
                self.bins_per_channel - 1,
            )
        else:
            quantized = image.astype(np.int32) >> shift
        blue = quantized[..., 0]
        green = quantized[..., 1]
        red = quantized[..., 2]
        return (red * self.bins_per_channel + green) * self.bins_per_channel + blue

    @staticmethod
    def _nearest_equivalent_angle(angle_deg: float, reference_deg: float) -> float:
        angle = float(angle_deg)
        while angle - reference_deg > 90.0:
            angle -= 180.0
        while angle - reference_deg < -90.0:
            angle += 180.0
        return angle

    @staticmethod
    def _rotated_rect_bbox(
        center: np.ndarray,
        axes: np.ndarray,
        angle_deg: float,
        frame_shape: tuple[int, ...],
    ) -> BBox:
        rectangle = (
            (float(center[0]), float(center[1])),
            (max(2.0, 2.0 * float(axes[0])), max(2.0, 2.0 * float(axes[1]))),
            float(angle_deg),
        )
        points = cv2.boxPoints(rectangle)
        x, y, w, h = cv2.boundingRect(points.astype(np.float32))
        return clip_bbox((float(x), float(y), float(w), float(h)), frame_shape)

    def _ellipse_samples(
        self,
        frame: np.ndarray,
        center: np.ndarray,
        axes: np.ndarray,
        angle_deg: float,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        candidate_axes = np.maximum(axes + float(margin), 2.0)
        theta = np.deg2rad(float(angle_deg))
        cosine = float(np.cos(theta))
        sine = float(np.sin(theta))

        # Axis-aligned bounds of the rotated ellipse.
        half_width = np.sqrt((candidate_axes[0] * cosine) ** 2 + (candidate_axes[1] * sine) ** 2)
        half_height = np.sqrt((candidate_axes[0] * sine) ** 2 + (candidate_axes[1] * cosine) ** 2)
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, int(np.floor(center[0] - half_width - 1.0)))
        y1 = max(0, int(np.floor(center[1] - half_height - 1.0)))
        x2 = min(frame_w, int(np.ceil(center[0] + half_width + 1.0)))
        y2 = min(frame_h, int(np.ceil(center[1] + half_height + 1.0)))
        if x2 <= x1 or y2 <= y1:
            empty = np.empty((0,), dtype=np.float64)
            return empty, empty, empty, empty, empty.astype(np.int32)

        roi = frame[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx.astype(np.float64) + 0.5 - float(center[0])
        dy = yy.astype(np.float64) + 0.5 - float(center[1])
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        radius_sq = (local_x / candidate_axes[0]) ** 2 + (local_y / candidate_axes[1]) ** 2
        mask = radius_sq <= 1.0
        if not np.any(mask):
            empty = np.empty((0,), dtype=np.float64)
            return empty, empty, empty, empty, empty.astype(np.int32)

        kernel = np.maximum(1.0 - radius_sq, 0.0)
        feature_indices = self._feature_indices(roi)
        return (
            xx[mask].astype(np.float64) + 0.5,
            yy[mask].astype(np.float64) + 0.5,
            kernel[mask].astype(np.float64),
            radius_sq[mask].astype(np.float64),
            feature_indices[mask].astype(np.int32),
        )

    def _probability_model(
        self,
        frame: np.ndarray,
        center: np.ndarray,
        axes: np.ndarray,
        angle_deg: float,
        margin: float,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        samples = self._ellipse_samples(frame, center, axes, angle_deg, margin)
        _, _, kernel, _, bins = samples
        if bins.size == 0:
            return np.zeros(self.hist_bins, dtype=np.float64), samples
        histogram = np.bincount(
            bins,
            weights=kernel,
            minlength=self.hist_bins,
        ).astype(np.float64)
        total = float(histogram.sum())
        if total <= 1e-12:
            histogram = np.bincount(bins, minlength=self.hist_bins).astype(np.float64)
            total = float(histogram.sum())
        if total > 0.0:
            histogram /= total
        return histogram, samples

    def _weight_statistics(
        self,
        frame: np.ndarray,
        center: np.ndarray,
        axes: np.ndarray,
        angle_deg: float,
    ) -> Optional[dict[str, Any]]:
        candidate_prob, samples = self._probability_model(
            frame,
            center,
            axes,
            angle_deg,
            self.candidate_margin,
        )
        xs, ys, _, _, bins = samples
        if bins.size == 0 or float(candidate_prob.sum()) <= 0.0:
            return None

        rho = float(np.sum(np.sqrt(np.maximum(self.target_prob, 0.0) * candidate_prob)))
        rho = float(np.clip(rho, 0.0, 1.0))
        denominator = np.maximum(candidate_prob[bins], 1e-12)
        weights = np.sqrt(np.maximum(self.target_prob[bins], 0.0) / denominator)
        weights = np.minimum(weights, self.max_weight)
        m00 = float(weights.sum())
        if not np.isfinite(m00) or m00 <= 1e-9:
            return None

        centroid = np.array(
            [float(np.dot(weights, xs) / m00), float(np.dot(weights, ys) / m00)],
            dtype=np.float64,
        )
        dx = xs - centroid[0]
        dy = ys - centroid[1]
        covariance = np.array(
            [
                [float(np.dot(weights, dx * dx) / m00), float(np.dot(weights, dx * dy) / m00)],
                [float(np.dot(weights, dx * dy) / m00), float(np.dot(weights, dy * dy) / m00)],
            ],
            dtype=np.float64,
        )
        covariance += np.eye(2, dtype=np.float64) * 1e-6
        correction = float(np.exp((rho - 1.0) / self.area_sigma))
        raw_area = max(m00 * correction, 1e-6)
        return {
            "center": centroid,
            "covariance": covariance,
            "rho": rho,
            "m00": m00,
            "raw_area": raw_area,
        }

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        bbox = clip_bbox(bbox, frame.shape)
        x, y, width, height = map(float, bbox)
        self.center = np.array([x + width / 2.0, y + height / 2.0], dtype=np.float64)
        if width >= height:
            self.axes = np.array([width / 2.0, height / 2.0], dtype=np.float64)
            self.angle_deg = 0.0
        else:
            self.axes = np.array([height / 2.0, width / 2.0], dtype=np.float64)
            self.angle_deg = 90.0
        self.initial_axes = self.axes.copy()

        target_prob, _ = self._probability_model(
            frame,
            self.center,
            self.axes,
            self.angle_deg,
            0.0,
        )
        if float(target_prob.sum()) <= 0.0:
            raise ValueError("SOAMST initial ellipse produces an empty target model")
        self.target_prob = target_prob

        # Eq. (12)--(13) estimates an ellipse area. A one-time normalization
        # makes the discretized first-frame estimate exactly match the initial
        # ellipse area while keeping every later update ground-truth-free.
        initial_stats = self._weight_statistics(frame, self.center, self.axes, self.angle_deg)
        initial_area = float(np.pi * self.axes[0] * self.axes[1])
        if initial_stats is not None and float(initial_stats["raw_area"]) > 1e-9:
            self.area_calibration = initial_area / float(initial_stats["raw_area"])
        else:
            self.area_calibration = 1.0
        self.last_bbox = self._rotated_rect_bbox(
            self.center, self.axes, self.angle_deg, frame.shape
        )

    def update(self, frame: np.ndarray) -> TrackResult:
        previous_center = self.center.copy()
        previous_axes = self.axes.copy()
        previous_angle = float(self.angle_deg)
        center = previous_center.copy()
        iterations = 0
        stats: Optional[dict[str, Any]] = None

        for iteration in range(self.max_iterations):
            stats = self._weight_statistics(frame, center, previous_axes, previous_angle)
            if stats is None:
                break
            new_center = np.asarray(stats["center"], dtype=np.float64)
            displacement = float(np.linalg.norm(new_center - center))
            center = new_center
            iterations = iteration + 1
            if displacement < self.epsilon:
                break

        stats = self._weight_statistics(frame, center, previous_axes, previous_angle)
        if stats is None:
            return TrackResult(
                bbox=self.last_bbox,
                ok=False,
                info={"source": "soamst_invalid_weight_image", "iterations": iterations},
            )

        covariance = np.asarray(stats["covariance"], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 1e-6)
        eigenvectors = eigenvectors[:, order]
        lambdas = np.sqrt(eigenvalues)

        area = max(
            self.area_calibration * float(stats["raw_area"]),
            np.pi * 2.0 * 2.0,
        )
        raw_major = np.sqrt(area * lambdas[0] / (np.pi * max(lambdas[1], 1e-9)))
        raw_minor = np.sqrt(area * lambdas[1] / (np.pi * max(lambdas[0], 1e-9)))
        raw_axes = np.array([raw_major, raw_minor], dtype=np.float64)

        per_frame_low = previous_axes / self.max_scale_step
        per_frame_high = previous_axes * self.max_scale_step
        global_low = self.initial_axes * self.min_global_scale
        global_high = self.initial_axes * self.max_global_scale
        lower = np.maximum(per_frame_low, global_low)
        upper = np.minimum(per_frame_high, global_high)
        constrained_axes = np.clip(raw_axes, lower, upper)
        axes = (
            (1.0 - self.shape_update_rate) * previous_axes
            + self.shape_update_rate * constrained_axes
        )

        major_vector = eigenvectors[:, 0]
        raw_angle = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])))
        raw_angle = self._nearest_equivalent_angle(raw_angle, previous_angle)
        angle = previous_angle + self.angle_update_rate * (raw_angle - previous_angle)

        center[0] = float(np.clip(center[0], 0.0, max(frame.shape[1] - 1.0, 0.0)))
        center[1] = float(np.clip(center[1], 0.0, max(frame.shape[0] - 1.0, 0.0)))
        self.center = center
        self.axes = np.maximum(axes, 2.0)
        self.angle_deg = float(angle)
        self.last_bbox = self._rotated_rect_bbox(
            self.center, self.axes, self.angle_deg, frame.shape
        )

        rho = float(stats["rho"])
        return TrackResult(
            bbox=self.last_bbox,
            ok=bool(np.isfinite(rho) and rho >= 0.05),
            info={
                "source": "soamst",
                "iterations": iterations,
                "bhattacharyya": rho,
                "m00": float(stats["m00"]),
                "estimated_area": area,
                "semi_major": float(self.axes[0]),
                "semi_minor": float(self.axes[1]),
                "angle": self.angle_deg,
            },
        )


class LocalFeatureTracker(BaseTracker):
    """Descriptor tracker with local search, stable geometry and cautious model update.

    The original implementation extracted descriptors from a small unscaled ROI,
    searched the whole frame at reduced resolution, and required a full homography.
    That combination can leave many frames with no valid update.  This version:

    * upsamples small initial targets before descriptor extraction;
    * searches a full-resolution local region first;
    * falls back to the immutable first-frame model for global re-detection;
    * estimates a similarity transform with RANSAC, which is better conditioned
      than an eight-parameter homography for small numbers of matches;
    * updates only the active model after a high-confidence result, while retaining
      the original model as a drift-resistant re-detection reference.
    """

    feature_name = "feature"

    def __init__(
        self,
        detector: Any,
        matcher: cv2.DescriptorMatcher,
        ratio_threshold: float,
        min_matches: int,
        min_inliers: int,
        ransac_threshold: float,
        processing_scale: float,
        local_search_factor: float,
        template_min_side: int,
        max_template_scale: float,
        model_update_interval: int,
        global_search_interval: int = 20,
    ) -> None:
        self.detector = detector
        self.matcher = matcher
        self.ratio_threshold = float(ratio_threshold)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.ransac_threshold = float(ransac_threshold)
        self.processing_scale = float(processing_scale)
        self.local_search_factor = float(local_search_factor)
        self.template_min_side = int(template_min_side)
        self.max_template_scale = float(max_template_scale)
        self.model_update_interval = max(1, int(model_update_interval))
        self.global_search_interval = max(1, int(global_search_interval))
        self.original_model: Optional[FeatureModel] = None
        self.active_model: Optional[FeatureModel] = None
        self.last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)
        self.frame_index = 0
        self.consecutive_failures = 0
        self.successful_updates = 0

    def _create_model(self, frame: np.ndarray, bbox: BBox) -> Optional[FeatureModel]:
        bbox = clip_bbox(bbox, frame.shape)
        x, y, w, h = bbox_to_int(bbox)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        smallest_side = max(1, min(gray.shape[:2]))
        scale = min(
            self.max_template_scale,
            max(1.0, self.template_min_side / float(smallest_side)),
        )
        if scale > 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or keypoints is None or len(keypoints) < 4:
            return None
        model_h, model_w = gray.shape[:2]
        corners = np.float32(
            [[[0.0, 0.0]], [[float(model_w), 0.0]], [[float(model_w), float(model_h)]], [[0.0, float(model_h)]]]
        )
        return FeatureModel(list(keypoints), np.ascontiguousarray(descriptors), corners)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        bbox = clip_bbox(bbox, frame.shape)
        model = self._create_model(frame, bbox)
        self.original_model = model
        self.active_model = model
        self.last_bbox = bbox
        self.frame_index = 0
        self.consecutive_failures = 0
        self.successful_updates = 0

    def set_search_hint(self, bbox: BBox, frame_shape: tuple[int, ...]) -> None:
        """Move only the search prior; descriptor models are left unchanged."""
        self.last_bbox = clip_bbox(bbox, frame_shape)

    def _detect_in_region(
        self,
        frame: np.ndarray,
        search_bbox: BBox,
        scale: float,
    ) -> tuple[list[Any], Optional[np.ndarray], tuple[float, float], float]:
        x, y, w, h = bbox_to_int(clip_bbox(search_bbox, frame.shape))
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return [], None, (float(x), float(y)), scale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        effective_scale = float(scale)
        if effective_scale != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=effective_scale,
                fy=effective_scale,
                interpolation=cv2.INTER_AREA if effective_scale < 1.0 else cv2.INTER_CUBIC,
            )
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return list(keypoints or []), descriptors, (float(x), float(y)), effective_scale

    def _required_matches(self, model: FeatureModel) -> int:
        adaptive = max(4, int(round(len(model.keypoints) * 0.12)))
        return min(self.min_matches, adaptive)

    @staticmethod
    def _candidate_geometry_valid(candidate: BBox, previous: BBox, global_search: bool) -> bool:
        _, candidate_size = bbox_center_size(candidate)
        _, previous_size = bbox_center_size(previous)
        ratios = candidate_size / np.maximum(previous_size, 1e-9)
        if global_search:
            return bool(np.all((ratios >= 0.25) & (ratios <= 4.0)))
        return bool(np.all((ratios >= 0.45) & (ratios <= 2.2)))

    @staticmethod
    def _stabilize_bbox(
        candidate: BBox,
        previous: BBox,
        frame_shape: tuple[int, ...],
        global_search: bool,
    ) -> BBox:
        candidate_center, candidate_size = bbox_center_size(candidate)
        previous_center, previous_size = bbox_center_size(previous)
        if global_search:
            low, high, size_rate, center_rate = 0.40, 2.50, 0.70, 1.00
        else:
            low, high, size_rate, center_rate = 0.70, 1.45, 0.45, 0.90
        constrained_size = np.clip(candidate_size, previous_size * low, previous_size * high)
        size = (1.0 - size_rate) * previous_size + size_rate * constrained_size
        center = (1.0 - center_rate) * previous_center + center_rate * candidate_center
        return bbox_from_center_size(center, size, frame_shape)

    def _match_model(
        self,
        model: FeatureModel,
        frame: np.ndarray,
        search_bbox: BBox,
        search_scale: float,
        source: str,
        global_search: bool,
    ) -> TrackResult:
        keypoints, descriptors, offset, effective_scale = self._detect_in_region(
            frame, search_bbox, search_scale
        )
        if descriptors is None or len(keypoints) < 2:
            return TrackResult(
                self.last_bbox,
                False,
                {"matches": 0, "inliers": 0, "reason": f"{source}_no_descriptors", "source": source},
            )
        try:
            raw_matches = self.matcher.knnMatch(model.descriptors, descriptors, k=2)
        except cv2.error:
            return TrackResult(
                self.last_bbox,
                False,
                {"matches": 0, "inliers": 0, "reason": f"{source}_matcher_failed", "source": source},
            )

        good: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if second.distance <= 1e-12:
                continue
            if first.distance < self.ratio_threshold * second.distance:
                good.append(first)

        required_matches = self._required_matches(model)
        if len(good) < required_matches:
            return TrackResult(
                self.last_bbox,
                False,
                {
                    "matches": len(good),
                    "inliers": 0,
                    "reason": f"{source}_too_few_ratio_matches",
                    "source": source,
                    "required_matches": required_matches,
                },
            )

        src = np.float32([model.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_points = []
        for match in good:
            point = keypoints[match.trainIdx].pt
            dst_points.append(
                (
                    point[0] / effective_scale + offset[0],
                    point[1] / effective_scale + offset[1],
                )
            )
        dst = np.float32(dst_points).reshape(-1, 1, 2)

        affine, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=2500,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None or mask is None:
            return TrackResult(
                self.last_bbox,
                False,
                {"matches": len(good), "inliers": 0, "reason": f"{source}_affine_failed", "source": source},
            )

        inliers = int(mask.ravel().sum())
        required_inliers = min(self.min_inliers, max(3, required_matches - 2))
        inlier_ratio = inliers / max(len(good), 1)
        if inliers < required_inliers or inlier_ratio < 0.35:
            return TrackResult(
                self.last_bbox,
                False,
                {
                    "matches": len(good),
                    "inliers": inliers,
                    "inlier_ratio": inlier_ratio,
                    "reason": f"{source}_too_few_inliers",
                    "source": source,
                },
            )

        transformed = cv2.transform(model.corners, affine)
        candidate = polygon_to_bbox(transformed, frame.shape)
        if candidate is None or not self._candidate_geometry_valid(candidate, self.last_bbox, global_search):
            return TrackResult(
                self.last_bbox,
                False,
                {
                    "matches": len(good),
                    "inliers": inliers,
                    "inlier_ratio": inlier_ratio,
                    "reason": f"{source}_invalid_geometry",
                    "source": source,
                },
            )

        bbox = self._stabilize_bbox(candidate, self.last_bbox, frame.shape, global_search)
        return TrackResult(
            bbox,
            True,
            {
                "matches": len(good),
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "reason": "ok",
                "source": source,
            },
        )

    def _maybe_update_active_model(self, frame: np.ndarray, result: TrackResult) -> None:
        if self.successful_updates % self.model_update_interval != 0:
            return
        inliers = int(result.info.get("inliers", 0))
        inlier_ratio = float(result.info.get("inlier_ratio", 0.0))
        if inliers < max(6, self.min_inliers) or inlier_ratio < 0.55:
            return
        model = self._create_model(frame, result.bbox)
        if model is not None and len(model.keypoints) >= 4:
            self.active_model = model

    def update(self, frame: np.ndarray) -> TrackResult:
        self.frame_index += 1
        if self.original_model is None:
            return TrackResult(
                self.last_bbox,
                False,
                {"matches": 0, "inliers": 0, "reason": "template_has_too_few_keypoints", "source": "none"},
            )

        search_factor = min(
            8.0,
            self.local_search_factor * (1.0 + 0.45 * min(self.consecutive_failures, 4)),
        )
        local_region = expand_bbox(self.last_bbox, search_factor, frame.shape)
        attempts: list[tuple[FeatureModel, BBox, float, str, bool]] = []
        if self.active_model is not None:
            attempts.append((self.active_model, local_region, 1.0, "local_active", False))
        if self.active_model is not self.original_model:
            attempts.append((self.original_model, local_region, 1.0, "local_original", False))

        should_global_search = (
            self.consecutive_failures >= 1
            or self.frame_index % self.global_search_interval == 0
        )
        if should_global_search:
            frame_h, frame_w = frame.shape[:2]
            full_frame = (0.0, 0.0, float(frame_w), float(frame_h))
            attempts.append((self.original_model, full_frame, self.processing_scale, "global_original", True))

        last_failure: Optional[TrackResult] = None
        for model, region, scale, source, global_search in attempts:
            result = self._match_model(model, frame, region, scale, source, global_search)
            if result.ok:
                self.last_bbox = result.bbox
                self.consecutive_failures = 0
                self.successful_updates += 1
                self._maybe_update_active_model(frame, result)
                return result
            last_failure = result

        self.consecutive_failures += 1
        if last_failure is None:
            last_failure = TrackResult(
                self.last_bbox,
                False,
                {"matches": 0, "inliers": 0, "reason": "no_search_attempt", "source": "none"},
            )
        return TrackResult(self.last_bbox, False, dict(last_failure.info))


class SIFTTracker(LocalFeatureTracker):
    name = "SIFT"
    feature_name = "SIFT"

    def __init__(
        self,
        ratio_threshold: float = 0.80,
        min_matches: int = 6,
        min_inliers: int = 4,
        ransac_threshold: float = 4.0,
        processing_scale: float = 0.85,
    ) -> None:
        super().__init__(
            detector=cv2.SIFT_create(
                nfeatures=800,
                contrastThreshold=0.020,
                edgeThreshold=10,
                sigma=1.6,
            ),
            matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
            ratio_threshold=ratio_threshold,
            min_matches=min_matches,
            min_inliers=min_inliers,
            ransac_threshold=ransac_threshold,
            processing_scale=processing_scale,
            local_search_factor=3.5,
            template_min_side=96,
            max_template_scale=4.0,
            model_update_interval=8,
            global_search_interval=20,
        )



class KCFTracker(BaseTracker):
    """Compact CPU KCF implementation using grayscale features and a Gaussian kernel.

    The implementation follows the standard cyclic-correlation formulation: a fixed
    search window is represented by a Hann-windowed grayscale feature map, training
    and detection are performed in the Fourier domain, and the model is updated by
    linear interpolation.  Scale is intentionally fixed so the baseline remains a
    direct test of translation tracking.
    """

    name = "KCF"

    def __init__(
        self,
        padding: float = 1.8,
        kernel_sigma: float = 0.5,
        regularization: float = 1e-4,
        interpolation_factor: float = 0.075,
        output_sigma_factor: float = 0.1,
        max_model_side: int = 160,
    ) -> None:
        self.padding = float(padding)
        self.kernel_sigma = float(kernel_sigma)
        self.regularization = float(regularization)
        self.interpolation_factor = float(interpolation_factor)
        self.output_sigma_factor = float(output_sigma_factor)
        self.max_model_side = int(max_model_side)
        self.center = np.array([0.0, 0.0], dtype=np.float64)  # x, y
        self.target_size = np.array([1.0, 1.0], dtype=np.float64)  # w, h
        self.search_size = np.array([2.0, 2.0], dtype=np.float64)
        self.model_size = (2, 2)  # width, height
        self.hann: Optional[np.ndarray] = None
        self.yf: Optional[np.ndarray] = None
        self.model_x: Optional[np.ndarray] = None
        self.model_alphaf: Optional[np.ndarray] = None
        self.last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    @staticmethod
    def _extract_patch(
        frame: np.ndarray,
        center: np.ndarray,
        search_size: np.ndarray,
        model_size: tuple[int, int],
    ) -> np.ndarray:
        search_w = max(2, int(round(search_size[0])))
        search_h = max(2, int(round(search_size[1])))
        x1 = int(round(center[0] - search_w / 2.0))
        y1 = int(round(center[1] - search_h / 2.0))
        x2 = x1 + search_w
        y2 = y1 + search_h
        frame_h, frame_w = frame.shape[:2]
        left = max(0, -x1)
        top = max(0, -y1)
        right = max(0, x2 - frame_w)
        bottom = max(0, y2 - frame_h)
        padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REPLICATE)
        x1 += left
        x2 += left
        y1 += top
        y2 += top
        patch = padded[y1:y2, x1:x2]
        if patch.size == 0:
            raise RuntimeError("KCF search patch is empty")
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if (gray.shape[1], gray.shape[0]) != model_size:
            gray = cv2.resize(gray, model_size, interpolation=cv2.INTER_LINEAR)
        return gray.astype(np.float32) / 255.0

    def _features(self, frame: np.ndarray) -> np.ndarray:
        if self.hann is None:
            raise RuntimeError("KCF is not initialized")
        patch = self._extract_patch(frame, self.center, self.search_size, self.model_size)
        patch = patch - float(patch.mean())
        std = float(patch.std())
        if std > 1e-6:
            patch = patch / std
        return patch * self.hann

    def _gaussian_correlation(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        xf = np.fft.fft2(x)
        zf = np.fft.fft2(z)
        correlation = np.fft.ifft2(xf * np.conj(zf)).real
        squared_distance = (
            float(np.sum(x * x)) + float(np.sum(z * z)) - 2.0 * correlation
        ) / float(x.size)
        squared_distance = np.maximum(squared_distance, 0.0)
        kernel = np.exp(-squared_distance / max(self.kernel_sigma**2, 1e-8))
        return np.fft.fft2(kernel)

    def _train(self, x: np.ndarray, first: bool) -> None:
        if self.yf is None:
            raise RuntimeError("KCF label is not initialized")
        kf = self._gaussian_correlation(x, x)
        alphaf = self.yf / (kf + self.regularization)
        if first or self.model_x is None or self.model_alphaf is None:
            self.model_x = x
            self.model_alphaf = alphaf
        else:
            rate = self.interpolation_factor
            self.model_x = (1.0 - rate) * self.model_x + rate * x
            self.model_alphaf = (1.0 - rate) * self.model_alphaf + rate * alphaf

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        bbox = clip_bbox(bbox, frame.shape)
        x, y, w, h = bbox
        self.center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
        self.target_size = np.array([w, h], dtype=np.float64)
        self.search_size = np.maximum(self.target_size * (1.0 + self.padding), 8.0)

        scale = min(1.0, self.max_model_side / float(max(self.search_size)))
        model_w = max(8, int(round(self.search_size[0] * scale)))
        model_h = max(8, int(round(self.search_size[1] * scale)))
        self.model_size = (model_w, model_h)

        hann_x = np.hanning(model_w).astype(np.float32)
        hann_y = np.hanning(model_h).astype(np.float32)
        self.hann = np.outer(hann_y, hann_x).astype(np.float32)

        output_sigma = max(
            1.0,
            np.sqrt(self.target_size[0] * self.target_size[1])
            * self.output_sigma_factor
            * scale,
        )
        yy, xx = np.mgrid[
            -(model_h // 2) : model_h - (model_h // 2),
            -(model_w // 2) : model_w - (model_w // 2),
        ]
        labels = np.exp(-0.5 * (xx * xx + yy * yy) / (output_sigma**2)).astype(np.float32)
        self.yf = np.fft.fft2(np.fft.ifftshift(labels))
        x_features = self._features(frame)
        self._train(x_features, first=True)
        self.last_bbox = bbox

    def update(self, frame: np.ndarray) -> TrackResult:
        if self.model_x is None or self.model_alphaf is None:
            raise RuntimeError("KCF is not initialized")
        z = self._features(frame)
        kzf = self._gaussian_correlation(z, self.model_x)
        response = np.fft.ifft2(self.model_alphaf * kzf).real
        peak_y, peak_x = np.unravel_index(int(np.argmax(response)), response.shape)
        model_w, model_h = self.model_size
        if peak_x > model_w // 2:
            peak_x -= model_w
        if peak_y > model_h // 2:
            peak_y -= model_h
        dx = float(peak_x) * self.search_size[0] / float(model_w)
        dy = float(peak_y) * self.search_size[1] / float(model_h)
        self.center += np.array([dx, dy], dtype=np.float64)

        x = self.center[0] - self.target_size[0] / 2.0
        y = self.center[1] - self.target_size[1] / 2.0
        bbox = clip_bbox((x, y, self.target_size[0], self.target_size[1]), frame.shape)
        bx, by, bw, bh = bbox
        self.center = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float64)
        self.last_bbox = bbox

        x_features = self._features(frame)
        self._train(x_features, first=False)
        peak = float(response.max())
        return TrackResult(bbox=bbox, ok=np.isfinite(peak), info={"source": "kcf", "response_peak": peak})


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


TRACKERS = ["MeanShift", "CBWH-MeanShift", "CAMShift", "SOAMST", "SIFT", "MeanShift+SIFT", "KCF"]

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
