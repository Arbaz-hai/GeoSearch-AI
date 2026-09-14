from __future__ import annotations

from pathlib import Path

from tests.conftest import write_scene


def test_export_contains_provenance(services, settings, tmp_path):
    before = write_scene(settings.raw_dir / "area_2023-01-15.png")
    after = write_scene(settings.raw_dir / "area_2023-02-15.png", changed=True)
    services.ingest.ingest_scene(before, trigger_change=False)
    services.ingest.ingest_scene(after)
    output = services.export.export_change_events(tmp_path / "events")
    assert Path(output["json"]).exists()
    assert Path(output["csv"]).exists()
    assert "provenance" in Path(output["json"]).read_text(encoding="utf-8")
