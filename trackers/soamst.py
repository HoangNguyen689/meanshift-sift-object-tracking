from typing import Any, Optional

from trackers.base import BBox, TrackResult, clip_bbox, bbox_from_center_size, BaseTracker

import cv2
import numpy as np


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
