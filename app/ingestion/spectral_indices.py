from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(slots=True)
class SpectralIndices:
    ndvi_mean: float
    ndvi_std: float
    ndvi_p10: float
    ndvi_p90: float
    ndwi_mean: float
    ndwi_std: float
    ndwi_p10: float
    ndwi_p90: float
    ndbi_mean: float
    ndbi_std: float
    ndbi_p10: float
    ndbi_p90: float
    evi_mean: float
    evi_std: float
    evi_p10: float
    evi_p90: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ndvi_mean": self.ndvi_mean,
            "ndvi_std": self.ndvi_std,
            "ndvi_p10": self.ndvi_p10,
            "ndvi_p90": self.ndvi_p90,
            "ndwi_mean": self.ndwi_mean,
            "ndwi_std": self.ndwi_std,
            "ndwi_p10": self.ndwi_p10,
            "ndwi_p90": self.ndwi_p90,
            "ndbi_mean": self.ndbi_mean,
            "ndbi_std": self.ndbi_std,
            "ndbi_p10": self.ndbi_p10,
            "ndbi_p90": self.ndbi_p90,
            "evi_mean": self.evi_mean,
            "evi_std": self.evi_std,
            "evi_p10": self.evi_p10,
            "evi_p90": self.evi_p90,
        }

    @classmethod
    def zero(cls) -> SpectralIndices:
        return cls(
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        )


def _compute_stats(index_arr: np.ndarray) -> tuple[float, float, float, float]:
    finite = index_arr[np.isfinite(index_arr)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_val = float(np.nanmean(finite))
    std_val = float(np.nanstd(finite))
    p10_val = float(np.nanpercentile(finite, 10))
    p90_val = float(np.nanpercentile(finite, 90))
    return round(mean_val, 4), round(std_val, 4), round(p10_val, 4), round(p90_val, 4)


def compute_spectral_indices(
    image: np.ndarray,
    band_map: dict[str, int] | None = None,
) -> SpectralIndices:
    """
    Compute NDVI, NDWI, NDBI, and EVI for a given CHW float32 raster.
    Default band_map matches typical Sentinel-2 / multispectral arrangements:
      blue: 0, green: 1, red: 2, nir: 3, swir: 4
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    num_bands = arr.shape[0]
    mapping = band_map or {
        "blue": 0,
        "green": 1,
        "red": 2,
        "nir": 3,
        "swir": 4,
    }

    # Extract bands safely
    blue = arr[mapping["blue"]] if mapping.get("blue", -1) < num_bands else None
    green = arr[mapping["green"]] if mapping.get("green", -1) < num_bands else None
    red = arr[mapping["red"]] if mapping.get("red", -1) < num_bands else None
    nir = arr[mapping["nir"]] if mapping.get("nir", -1) < num_bands else None
    swir = arr[mapping["swir"]] if mapping.get("swir", -1) < num_bands else None

    # NDVI = (NIR - Red) / (NIR + Red)
    if nir is not None and red is not None:
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    else:
        ndvi = np.zeros(arr.shape[1:], dtype=np.float32)

    # NDWI = (Green - NIR) / (Green + NIR)
    if green is not None and nir is not None:
        ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
    else:
        ndwi = np.zeros(arr.shape[1:], dtype=np.float32)

    # NDBI = (SWIR - NIR) / (SWIR + NIR)
    if swir is not None and nir is not None:
        ndbi = (swir - nir) / np.maximum(swir + nir, 1e-6)
    else:
        ndbi = np.zeros(arr.shape[1:], dtype=np.float32)

    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    if nir is not None and red is not None and blue is not None:
        denom = nir + 6.0 * red - 7.5 * blue + 1.0
        evi = 2.5 * (nir - red) / np.where(np.abs(denom) < 1e-5, 1e-5, denom)
    elif nir is not None and red is not None:
        evi = 2.5 * (nir - red) / np.maximum(nir + 2.4 * red + 1.0, 1e-6)
    else:
        evi = np.zeros(arr.shape[1:], dtype=np.float32)

    ndvi_stats = _compute_stats(ndvi)
    ndwi_stats = _compute_stats(ndwi)
    ndbi_stats = _compute_stats(ndbi)
    evi_stats = _compute_stats(evi)

    return SpectralIndices(
        ndvi_mean=ndvi_stats[0],
        ndvi_std=ndvi_stats[1],
        ndvi_p10=ndvi_stats[2],
        ndvi_p90=ndvi_stats[3],
        ndwi_mean=ndwi_stats[0],
        ndwi_std=ndwi_stats[1],
        ndwi_p10=ndwi_stats[2],
        ndwi_p90=ndwi_stats[3],
        ndbi_mean=ndbi_stats[0],
        ndbi_std=ndbi_stats[1],
        ndbi_p10=ndbi_stats[2],
        ndbi_p90=ndbi_stats[3],
        evi_mean=evi_stats[0],
        evi_std=evi_stats[1],
        evi_p10=evi_stats[2],
        evi_p90=evi_stats[3],
    )
