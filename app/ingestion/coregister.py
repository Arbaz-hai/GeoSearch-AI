from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CoregistrationResult:
    aligned: np.ndarray
    shift_y: float
    shift_x: float
    quality: float


def _shift(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    result = np.full_like(image, np.nan)
    source_y_start = max(0, -shift_y)
    source_y_end = image.shape[-2] - max(0, shift_y)
    source_x_start = max(0, -shift_x)
    source_x_end = image.shape[-1] - max(0, shift_x)
    target_y_start = max(0, shift_y)
    target_y_end = target_y_start + max(0, source_y_end - source_y_start)
    target_x_start = max(0, shift_x)
    target_x_end = target_x_start + max(0, source_x_end - source_x_start)
    result[..., target_y_start:target_y_end, target_x_start:target_x_end] = image[..., source_y_start:source_y_end, source_x_start:source_x_end]
    return result


def _band_mean(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(values)
    return np.where(valid.sum(axis=0) > 0, np.where(valid, values, 0).sum(axis=0) / np.maximum(valid.sum(axis=0), 1), np.nan)


def coregister(reference: np.ndarray, moving: np.ndarray, max_shift: int = 12) -> CoregistrationResult:
    ref = _band_mean(reference)
    candidate = _band_mean(moving)
    valid = np.isfinite(ref) & np.isfinite(candidate)
    if valid.sum() < 32:
        return CoregistrationResult(np.asarray(moving, dtype=np.float32), 0.0, 0.0, 0.0)
    reference_filled = np.nan_to_num(ref, nan=float(np.nanmean(ref[valid])))
    moving_filled = np.nan_to_num(candidate, nan=float(np.nanmean(candidate[valid])))
    best_shift = (0, 0)
    best_score = -np.inf
    for shift_y in range(-max_shift, max_shift + 1):
        for shift_x in range(-max_shift, max_shift + 1):
            shifted = _shift(moving_filled, shift_y, shift_x)
            overlap = np.isfinite(shifted) & np.isfinite(reference_filled)
            if overlap.sum() < 32:
                continue
            left = reference_filled[overlap]
            right = shifted[overlap]
            denominator = np.linalg.norm(left - left.mean()) * np.linalg.norm(right - right.mean())
            score = float(np.dot(left - left.mean(), right - right.mean()) / denominator) if denominator else -np.inf
            if score > best_score:
                best_score = score
                best_shift = (shift_y, shift_x)
    aligned = _shift(np.asarray(moving, dtype=np.float32), *best_shift)
    return CoregistrationResult(aligned, float(best_shift[0]), float(best_shift[1]), float(max(best_score, 0.0)))
