from __future__ import annotations

import numpy as np


def dark_object_subtraction(image: np.ndarray, percentile: float = 1.0) -> np.ndarray:
    """
    Apply Dark Object Subtraction (DOS) per band as an atmospheric path-radiance correction.
    Subtracts the 1st percentile (darkest non-zero pixel) from each band.
    """
    arr = np.asarray(image, dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    for c in range(arr.shape[0]):
        band = arr[c]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        dark_val = float(np.percentile(finite, percentile))
        arr[c] = np.maximum(arr[c] - dark_val, 0.0)

    return arr


def histogram_match(source: np.ndarray, reference: np.ndarray, num_quantiles: int = 256) -> np.ndarray:
    """
    Match cumulative distribution function (CDF) of source to reference per band.
    Used before ChangeFormer to normalize seasonal, atmospheric, and illumination differences.
    """
    src = np.asarray(source, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)

    if src.ndim == 2:
        src = src[np.newaxis, :, :]
    if ref.ndim == 2:
        ref = ref[np.newaxis, :, :]

    bands = min(src.shape[0], ref.shape[0])
    quantiles = np.linspace(0.0, 1.0, num_quantiles)

    for b in range(bands):
        s_band = src[b]
        r_band = ref[b]

        s_valid = s_band[np.isfinite(s_band)]
        r_valid = r_band[np.isfinite(r_band)]

        if s_valid.size < 16 or r_valid.size < 16:
            continue

        s_quantiles = np.quantile(s_valid, quantiles)
        r_quantiles = np.quantile(r_valid, quantiles)

        # Map source pixels through interpolation between CDFs
        flat_s = s_band.flatten()
        mapped_flat = np.interp(flat_s, s_quantiles, r_quantiles)
        src[b] = mapped_flat.reshape(s_band.shape)

    return src


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Clip and normalize multi-band raster to [0, 1] per band with 2-98% percentile stretch.
    """
    arr = np.asarray(image, dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    for c in range(arr.shape[0]):
        band = arr[c]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        p2, p98 = np.percentile(finite, (2, 98))
        if p98 > p2:
            arr[c] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
        else:
            arr[c] = np.clip(band, 0.0, 1.0)

    return arr
