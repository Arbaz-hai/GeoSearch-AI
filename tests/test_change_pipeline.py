from __future__ import annotations

from tests.conftest import write_scene


def test_change_analysis_creates_confidence_ranked_event(services, settings):
    before = write_scene(settings.raw_dir / "area_2023-01-15.png")
    after = write_scene(settings.raw_dir / "area_2023-02-15.png", changed=True)
    services.ingest.ingest_scene(before, trigger_change=False)
    services.ingest.ingest_scene(after, trigger_change=False)
    events = services.change.analyze("2023-01-01", "2023-03-01")
    assert events
    assert events[0]["confidence"] >= 0
    assert events[0]["earliest_detected_date"] >= "2023-01-15"
