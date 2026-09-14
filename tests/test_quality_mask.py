from __future__ import annotations

import numpy as np

from app.ingestion.quality_mask import quality_mask, sentinel_scl_mask, valid_fraction


def test_scl_cloud_and_shadow_classes_are_invalid():
    scl = np.array([[4, 8], [3, 6]], dtype=np.uint8)
    mask = sentinel_scl_mask(scl)
    assert mask.tolist() == [[True, False], [False, True]]


def test_quality_mask_excludes_non_finite_values():
    image = np.ones((3, 4, 4), dtype=np.float32)
    image[:, 0, 0] = np.nan
    mask = quality_mask(image, "Sentinel-2 L2A", cloud_threshold=1.0)
    assert not mask[0, 0]
    assert valid_fraction(mask) == 15 / 16
