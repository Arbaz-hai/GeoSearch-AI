from __future__ import annotations

import numpy as np


INVALID_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}


def sentinel_scl_mask(scl: np.ndarray) -> np.ndarray:
    return ~np.isin(scl, list(INVALID_SCL_CLASSES))


def estimate_cloud_probability(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    channels = data.shape[0]
    blue = data[0]
    visible = data[: min(3, channels)]
    valid = np.isfinite(visible)
    count = valid.sum(axis=0)
    filled = np.where(valid, visible, 0)
    mean = filled.sum(axis=0) / np.maximum(count, 1)
    variance = np.where(valid, (visible - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    spread = np.sqrt(variance)
    finite_mean = mean[count > 0]
    finite_spread = spread[count > 0]
    finite_blue = blue[np.isfinite(blue)]
    if not finite_mean.size or not finite_blue.size:
        return np.ones(data.shape[-2:], dtype=np.float32)
    mean_low, mean_high = np.percentile(finite_mean, [10, 98])
    blue_low, blue_high = np.percentile(finite_blue, [10, 98])
    spread_high = np.percentile(finite_spread, 95) if finite_spread.size else 1.0
    brightness = (mean - mean_low) / max(mean_high - mean_low, 1e-6)
    whiteness = 1 - np.clip(spread / max(spread_high, 1e-6), 0, 1)
    blue_score = (np.nan_to_num(blue, nan=blue_low) - blue_low) / max(blue_high - blue_low, 1e-6)
    return np.clip(0.45 * brightness + 0.35 * whiteness + 0.2 * blue_score, 0, 1)


def s2cloudless_probability(image: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 10:
        return estimate_cloud_probability(values)
    try:
        from s2cloudless import S2PixelCloudDetector

        detector = S2PixelCloudDetector(threshold=threshold, average_over=4, dilation_size=2, all_bands=False)
        bands = np.moveaxis(np.nan_to_num(values[:10], nan=0.0), 0, -1)[np.newaxis, ...]
        return detector.get_cloud_probability_maps(bands)[0]
    except Exception:
        return estimate_cloud_probability(values)


def quality_mask(image: np.ndarray, sensor: str, cloud_threshold: float = 0.65, scl: np.ndarray | None = None, fmask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    valid = np.all(np.isfinite(values), axis=0) if values.ndim == 3 else np.isfinite(values)
    sensor_token = sensor.lower()
    if "sentinel-2" in sensor_token and scl is not None:
        valid &= sentinel_scl_mask(scl)
    if "landsat" in sensor_token and fmask is not None:
        valid &= fmask == 0
    
    # Only apply heuristic probability filter if 10+ bands or extreme values exist
    if values.shape[0] >= 10:
        probability = s2cloudless_probability(values, cloud_threshold)
        valid &= probability <= cloud_threshold
    else:
        # Standard RGB / 3-4 band image: mask only true saturated blown-out white regions
        brightness = np.nanmean(values[:3], axis=0) if values.ndim == 3 else values
        # If image is scaled 0-255 or 0-1
        is_255 = float(np.nanmax(values)) > 2.0
        sat_val = 250.0 if is_255 else 0.98
        is_cloud = (brightness >= sat_val) & (np.nanstd(values[:3], axis=0) < (5.0 if is_255 else 0.05))
        valid &= ~is_cloud

    return valid


def apply_quality_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32).copy()
    if data.ndim == 3:
        data[:, ~mask] = np.nan
    else:
        data[~mask] = np.nan
    return data


def valid_fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0
