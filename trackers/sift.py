from typing import Any, Optional

from trackers.base import BBox, TrackResult, clip_bbox, bbox_to_int, polygon_to_bbox, bbox_center_size, bbox_from_center_size, expand_bbox, BaseTracker, FeatureModel

import cv2
import numpy as np


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



