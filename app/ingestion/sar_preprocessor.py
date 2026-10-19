from __future__ import annotations

import numpy as np


def radiometric_calibration(sar_dn: np.ndarray, calibration_factor: float = 1.0) -> np.ndarray:
    """
    Convert SAR digital numbers (DN) to sigma-naught (backscatter coefficient in dB).
    sigma_0_dB = 10 * log10(DN^2 * calibration_factor + 1e-7)
    Clamps output to typical [-35 dB, +5 dB] dynamic range.
    """
    dn = np.asarray(sar_dn, dtype=np.float32)
    power = (dn ** 2) * calibration_factor
    sigma_0_db = 10.0 * np.log10(np.maximum(power, 1e-7))
    return np.clip(sigma_0_db, -35.0, 5.0)


def lee_speckle_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Adaptive Lee Speckle Filter for synthetic aperture radar (SAR) imagery.
    Preserves edges and structural targets while aggressively smoothing homogeneous speckle noise.
    """
    arr = np.asarray(image, dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    pad = kernel_size // 2
    c, h, w = arr.shape
    filtered = np.zeros_like(arr)

    for band_idx in range(c):
        band = arr[band_idx]
        padded = np.pad(band, pad, mode="reflect")

        # Compute rolling window mean & variance
        local_mean = np.zeros_like(band)
        local_var = np.zeros_like(band)

        # Sliding window statistics
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                patch = padded[dy : dy + h, dx : dx + w]
                local_mean += patch
                local_var += patch ** 2

        num_pixels = kernel_size * kernel_size
        local_mean /= num_pixels
        local_var = np.maximum(local_var / num_pixels - local_mean ** 2, 1e-6)

        # Overall noise variance estimate (assuming Rayleigh/gamma distributed speckle)
        overall_noise_var = np.nanvar(band) if np.nanvar(band) > 0 else 1.0

        # Lee weighting coefficient W = local_var / (local_var + overall_noise_var)
        w_coeff = local_var / (local_var + overall_noise_var)
        filtered[band_idx] = local_mean + w_coeff * (band - local_mean)

    return filtered


def preprocess_sar(
    sar_array: np.ndarray,
    calibration_factor: float = 1.0,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Full Sentinel-1 GRD SAR Preprocessing pipeline:
    1. Radiometric calibration to sigma-0 backscatter (dB)
    2. Adaptive Lee filter for speckle noise suppression
    3. Min-max normalization for downstream neural encoder consumption
    """
    calibrated = radiometric_calibration(sar_array, calibration_factor=calibration_factor)
    filtered = lee_speckle_filter(calibrated, kernel_size=kernel_size)
    # Normalize dB range [-35, 5] to [0, 1]
    normalized = np.clip((filtered + 35.0) / 40.0, 0.0, 1.0)
    return normalized
