from __future__ import annotations

from app.ingestion.tiler import read_scene, tile_scene
from tests.conftest import write_scene


def test_tiler_preserves_grid_and_transform(tmp_path):
    path = write_scene(tmp_path / "scene_2023-01-15.png")
    scene = read_scene(path)
    tiles = list(tile_scene(scene, tile_size=32))
    assert len(tiles) == 4
    assert tiles[0].image.shape == (3, 32, 32)
    assert tiles[3].grid_row == 1
    assert tiles[3].grid_col == 1
    assert len(tiles[0].transform) == 6
    assert len(tiles[0].bounds) == 4
