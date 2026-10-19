from __future__ import annotations

import numpy as np


def compute_shadow_mask(
    image: np.ndarray,
    solar_azimuth_deg: float | None = None,
    solar_elevation_deg: float | None = None,
    dsm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a boolean shadow mask (True = shadow-affected pixel).
    
    If solar geometry and DSM (Digital Surface Model) are provided, compute ray-cast geometric shadow.
    Otherwise, fall back to physics-informed multi-spectral shadow detection:
      Shadow pixels have low visible brightness, very low NIR reflectance, but are not deep water.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    h, w = arr.shape[1], arr.shape[2]

    # Geometric shadow if DSM and solar elevation available
    if dsm is not None and solar_elevation_deg is not None and solar_azimuth_deg is not None:
        try:
            elev_rad = np.radians(max(solar_elevation_deg, 5.0))
            azim_rad = np.radians(solar_azimuth_deg)
            # Simple directional slope/shadow projection
            dx = int(round(-np.sin(azim_rad) / np.tan(elev_rad)))
            dy = int(round(np.cos(azim_rad) / np.tan(elev_rad)))
            shifted_dsm = np.roll(np.roll(dsm, dy, axis=0), dx, axis=1)
            shadow_mask = shifted_dsm > dsm
            return shadow_mask
        except Exception:
            pass

    # Spectral fallback shadow detection
    # Typical RGB / NIR bands
    red = arr[2] if arr.shape[0] > 2 else arr[0]
    green = arr[1] if arr.shape[0] > 1 else arr[0]
    blue = arr[0]
    nir = arr[3] if arr.shape[0] > 3 else arr[-1]

    brightness = (red + green + blue) / 3.0
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    ndwi = (green - nir) / np.maximum(green + nir, 1e-6)

    # Shadow: low brightness, not open water (NDWI not extremely high), low NIR
    b_p15 = np.nanpercentile(brightness, 15) if brightness.size > 0 else 0.1
    shadow_candidate = (brightness < b_p15) & (nir < b_p15 * 1.2) & (ndwi < 0.3)

    return shadow_candidate.astype(bool)


def shadow_fraction(mask: np.ndarray) -> float:
    """Compute the percentage/fraction of pixels flagged as shadow."""
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.mean(mask.astype(np.float32)))
