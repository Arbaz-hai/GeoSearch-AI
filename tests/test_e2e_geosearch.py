from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

from app.auth import create_token, verify_token
from tests.conftest import write_scene


def test_complete_geosearch_end_to_end(services, settings):
    """
    End-to-End Workflow Verification:
    1. Create user & hash password
    2. Authenticate & obtain local token
    3. Ingest imagery scenes (T1 & T2)
    4. Text-to-image semantic search (Qwen3 NLP + CLIP)
    5. Image-to-image visual search (CLIP)
    6. Verify scene-level results (not raw tile fragments)
    7. Multi-temporal change detection (alignment + masking)
    8. Analyst review queue & decision (Accept/Reject)
    9. Verify persistent audit trail
    10. 'Find More Like This' similar-scene search
    """

    # 1. Create user & password hash
    username = "analyst_e2e"
    password = "SecurePassword123!"
    user = services.metadata.create_user(username, password)
    assert user["username"] == username
    assert "password_hash" not in user  # Never exposed

    # 2. Authenticate
    auth_user = services.metadata.authenticate_user(username, password)
    assert auth_user is not None
    assert auth_user["id"] == user["id"]
    token = create_token(user["id"], user["username"])
    assert token

    # Test invalid credentials fail
    assert services.metadata.authenticate_user(username, "WrongPass123!") is None

    # 3. Ingest imagery (T1 Baseline & T2 Observation with change)
    t1_path = write_scene(settings.raw_dir / "scene_t1_2024-01-15.png")
    t2_path = write_scene(settings.raw_dir / "scene_t2_2024-06-15.png", changed=True)

    t1_res = services.ingest.ingest_scene(t1_path, trigger_change=False)
    assert t1_res["status"] == "indexed"
    assert t1_res["tiles_indexed"] > 0

    t2_res = services.ingest.ingest_scene(t2_path, trigger_change=False)
    assert t2_res["status"] == "indexed"

    # Verify duplicate detection skips re-ingestion
    t1_dup = services.ingest.ingest_scene(t1_path, trigger_change=False)
    assert t1_dup["status"] == "duplicate"

    # 4. Text-to-image search
    search_res = services.search.search_text_with_status("Find satellite scenes containing terrain")
    assert search_res["matched"] is True
    assert len(search_res["results"]) > 0

    # 5. Verify scene-level results (not raw tile fragments)
    first_result = search_res["results"][0]
    assert "scene_id" in first_result
    assert "preview_path" in first_result
    assert "similarity" in first_result
    assert first_result["similarity"] > 0

    # 6. Image-to-image search
    img_res = services.search.search_image(t1_path, limit=5)
    assert len(img_res) > 0

    # 7. Multi-temporal change detection
    events = services.change.analyze("2024-01-01", "2024-12-31")
    assert len(events) > 0
    top_event = events[0]
    assert "confidence" in top_event
    assert "change_type" in top_event
    assert "earliest_detected_date" in top_event

    # 8. Analyst review queue & decision (Accept / Confirm)
    queue = services.change.review_queue()
    assert len(queue) > 0
    event_to_review = queue[0]

    decision = services.metadata.record_decision(
        event_to_review["id"],
        analyst_label="confirm",
        analyst_id=username,
        note="E2E test verified appearance of new structures",
    )
    assert decision["analyst_label"] == "confirm"
    assert decision["analyst_id"] == username

    # Verify status changed in DB
    updated_event = services.metadata.get_change_event(event_to_review["id"])
    assert updated_event["status"] == "confirmed"

    # 9. Verify audit trail is persistent
    audit_logs = services.metadata.audit_log(entity_id=event_to_review["id"])
    assert len(audit_logs) > 0
    analyst_actions = [l for l in audit_logs if l["action"] == "analyst_decision"]
    assert len(analyst_actions) > 0
    assert analyst_actions[0]["actor"] == username

    # 10. "Find More Like This" similarity search
    similar_scenes = services.search.find_similar_to_scene(t1_res["scene"]["id"])
    # Should exclude the scene itself
    for sim in similar_scenes:
        assert sim["scene_id"] != t1_res["scene"]["id"]
