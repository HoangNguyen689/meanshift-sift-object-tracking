from __future__ import annotations

from typing import Iterable

import numpy as np

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = max(aw * ah + bw * bh - inter, 1e-12)
    return float(inter / union)


def center_error(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ac = np.array([ax + aw / 2.0, ay + ah / 2.0], dtype=np.float64)
    bc = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float64)
    return float(np.linalg.norm(ac - bc))


def success_auc(values: Iterable[float], thresholds: np.ndarray | None = None) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return 0.0
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    curve = np.array([(data >= threshold).mean() for threshold in thresholds], dtype=np.float64)
    return float(np.trapezoid(curve, thresholds))


def precision_at(errors: Iterable[float], threshold: float = 20.0) -> float:
    data = np.asarray(list(errors), dtype=np.float64)
    if data.size == 0:
        return 0.0
    return float((data <= threshold).mean())
