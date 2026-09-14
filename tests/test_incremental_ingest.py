from __future__ import annotations

from tests.conftest import write_scene


def test_incremental_ingest_preserves_existing_vectors_and_searches_new_scene(services, settings):
    first = write_scene(settings.raw_dir / "area_2023-01-15.png")
    second = write_scene(settings.raw_dir / "area_2023-02-15.png", changed=True)
    services.ingest.ingest_scene(first, trigger_change=False)
    original_ids = services.vectors.point_ids(settings.visual_collection)
    result = services.ingest.ingest_scene(second)
    current_ids = services.vectors.point_ids(settings.visual_collection)
    assert result["status"] == "indexed"
    assert original_ids.issubset(current_ids)
    assert len(current_ids) > len(original_ids)
    found = services.search.search_image(second, limit=1)
    assert found and found[0]["scene_id"] == "area_2023-02-15"
    assert result["change_events"]
