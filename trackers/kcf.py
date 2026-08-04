from typing import Optional

from trackers.base import BBox, TrackResult, clip_bbox, BaseTracker

import cv2
import numpy as np


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


