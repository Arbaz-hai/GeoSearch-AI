from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ChangeModelResult:
    change_map: np.ndarray
    confidence: float
    change_type: str
    backend: str


def _ndvi(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.shape[0] < 4:
        return np.zeros(values.shape[1:], dtype=np.float32)
    red = values[2]
    nir = values[3]
    return (nir - red) / np.maximum(nir + red, 1e-6)


class ChangeModel:
    def __init__(self, model_path: str | Path, offline: bool = True):
        self.model_path = Path(model_path)
        self.offline = offline
        self.backend = "local-quality-aware-change-fallback"

    def predict(self, before: np.ndarray, after: np.ndarray, valid_mask: np.ndarray | None = None) -> ChangeModelResult:
        left = np.asarray(before, dtype=np.float32)
        right = np.asarray(after, dtype=np.float32)
        bands = min(left.shape[0], right.shape[0])
        delta = np.abs(right[:bands] - left[:bands])
        valid_delta = np.isfinite(delta)
        difference = np.where(valid_delta.sum(axis=0) > 0, np.where(valid_delta, delta, 0).sum(axis=0) / np.maximum(valid_delta.sum(axis=0), 1), np.nan)
        finite = difference[np.isfinite(difference)]
        if finite.size < 16:
            return ChangeModelResult(np.zeros_like(difference, dtype=np.float32), 0.0, "unknown", self.backend)
        baseline = float(np.median(finite))
        scale = float(np.percentile(finite, 95) - baseline)
        normalized = np.clip((difference - baseline) / max(scale, 1e-6), 0, 1)
        if valid_mask is not None:
            normalized = np.where(valid_mask, normalized, 0)
        active = normalized >= 0.55
        confidence = float(np.mean(normalized[active])) if active.any() else 0.0
        before_ndvi = _ndvi(left)
        after_ndvi = _ndvi(right)
        ndvi_delta = float(np.nanmean(after_ndvi - before_ndvi))
        mean_delta = float(np.nanmean(right[:bands]) - np.nanmean(left[:bands]))
        if ndvi_delta < -0.12:
            label = "clearance"
        elif ndvi_delta > 0.12:
            label = "water-extent"
        elif mean_delta > 0.08:
            label = "construction"
        elif mean_delta < -0.08:
            label = "road-development"
        else:
            label = "appearance"
        return ChangeModelResult(normalized.astype(np.float32), confidence, label, self.backend)
