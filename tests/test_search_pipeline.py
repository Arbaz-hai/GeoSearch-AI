from __future__ import annotations

from tests.conftest import write_scene


def test_image_search_returns_ranked_tiles(services, settings):
    source = write_scene(settings.raw_dir / "area_2023-01-15.png")
    services.ingest.ingest_scene(source, trigger_change=False)
    results = services.search.search_image(source, limit=3)
    assert results
    assert results[0]["rank"] == 1
    assert results[0]["preview_path"]


def test_text_search_accepts_metadata_filters(services, settings):
    source = write_scene(settings.raw_dir / "area_2023-01-15.png")
    services.ingest.ingest_scene(source, trigger_change=False)
    results = services.search.search_text("open ground", filters={"date_start": "2023-01-01", "date_end": "2023-01-31"})
    assert results
