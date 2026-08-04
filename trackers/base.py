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

