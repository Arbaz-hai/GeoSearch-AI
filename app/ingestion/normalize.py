from __future__ import annotations

import numpy as np


def normalize_reflectance(image: np.ndarray, lower_percentile: float = 2, upper_percentile: float = 98) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    result = np.full_like(source, np.nan, dtype=np.float32)
    for band in range(source.shape[0]):
        values = source[band]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [lower_percentile, upper_percentile])
        if high <= low:
            result[band] = np.where(np.isfinite(values), 0.5, np.nan)
        else:
            result[band] = np.clip((values - low) / (high - low), 0, 1)
    return result


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_data = np.asarray(source, dtype=np.float32)
    reference_data = np.asarray(reference, dtype=np.float32)
    matched = np.full_like(source_data, np.nan, dtype=np.float32)
    bands = min(source_data.shape[0], reference_data.shape[0])
    quantiles = np.linspace(0, 100, 256)
    for band in range(bands):
        current = source_data[band]
        target = reference_data[band]
        valid_current = np.isfinite(current)
        valid_target = np.isfinite(target)
        if valid_current.sum() < 8 or valid_target.sum() < 8:
            matched[band] = current
            continue
        current_values = current[valid_current]
        source_quantiles = np.percentile(current_values, quantiles)
        target_quantiles = np.percentile(target[valid_target], quantiles)
        matched_band = np.interp(current_values, source_quantiles, target_quantiles)
        result = current.copy()
        result[valid_current] = matched_band
        matched[band] = result
    if source_data.shape[0] > bands:
        matched[bands:] = source_data[bands:]
    return matched
