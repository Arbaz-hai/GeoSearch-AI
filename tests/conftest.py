from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services import create_services
from app.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        root_dir=tmp_path,
        project_name="Test Satellite Search",
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        tile_dir=data_dir / "tiles",
        model_dir=data_dir / "models",
        index_dir=data_dir / "index",
        qdrant_path=data_dir / "index" / "qdrant",
        database_url="sqlite:///data/index/metadata.db",
        tile_size=32,
        tile_stride=32,
        embedding_dimension=32,
        confidence_threshold=0.0,
        cloud_probability_threshold=1.0,
        minimum_valid_fraction=0.1,
        model_paths={"remoteclip": "data/models/remoteclip", "prithvi": "data/models/prithvi", "changeformer": "data/models/changeformer"},
    )


@pytest.fixture
def services(settings: Settings):
    settings.ensure_directories()
    return create_services(settings)


def write_scene(path: Path, changed: bool = False) -> Path:
    height = width = 64
    y, x = np.mgrid[:height, :width]
    base = (35 + x * 2 + y).astype(np.uint8)
    image = np.stack([base, np.flipud(base), base // 2], axis=-1)
    if changed:
        image[20:48, 20:48] = np.array([250, 250, 250], dtype=np.uint8)
    Image.fromarray(image).save(path)
    return path
