from __future__ import annotations

from pathlib import Path
from typing import Any


def model_registry(settings: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name, details in settings.model_registry.items():
        records.append(
            {
                "name": name,
                "repository": str(details.get("repository", "")),
                "revision": str(details.get("revision", "main")),
                "license": str(details.get("license", "Unknown")),
                "local_path": str(settings.model_path(name)),
            }
        )
    return records


def model_staging_status(settings: Any) -> list[dict[str, Any]]:
    return [{**record, "staged": Path(record["local_path"]).exists() and any(Path(record["local_path"]).iterdir())} for record in model_registry(settings)]
