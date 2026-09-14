from __future__ import annotations

import zipfile

from tests.conftest import write_scene


def test_zip_ingest_saves_descriptions_and_is_searchable(services, settings, tmp_path):
    source = write_scene(tmp_path / "industrial_river_2023-04-12.png")
    archive = tmp_path / "observations.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source, source.name)

    result = services.ingest.ingest_path(archive)

    assert result["status"] == "indexed"
    assert result["files_indexed"] == 1
    tile = services.metadata.list_tiles()[0]
    assert "industrial river" in tile["description"].lower()
    found = services.search.search_text("industrial river")
    assert found and "industrial river" in found[0]["description"].lower()


def test_folder_ingest_keeps_same_named_files_from_different_folders(services, tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    write_scene(left / "observation_2023-04-12.png")
    write_scene(right / "observation_2023-04-12.png")

    result = services.ingest.ingest_path(tmp_path)

    assert result["files_indexed"] == 2
    assert len(services.metadata.list_scenes()) == 2


def test_local_account_uses_a_password_check(services):
    account = services.metadata.create_user("analyst.one", "safe-password")
    assert account["username"] == "analyst.one"
    assert services.metadata.authenticate_user("analyst.one", "safe-password")
    assert services.metadata.authenticate_user("analyst.one", "wrong-password") is None


def test_change_question_uses_detected_events(services):
    answer = services.change.answer_question(
        "When was the earliest change?",
        [{"change_type": "new_construction", "confidence": 0.82, "earliest_detected_date": "2024-03-01"}],
    )
    assert "2024-03-01" in answer
    assert "82%" in answer
