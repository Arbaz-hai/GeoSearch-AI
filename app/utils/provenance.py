from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def processing_step(name: str, version: str = "1", details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "version": version, "timestamp": utc_now(), "details": details or {}}


def build_provenance(
    source_scene_ids: list[str],
    processing_steps: list[dict[str, Any]],
    model_backends: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_scenes": source_scene_ids,
        "processing_steps": processing_steps,
        "model_backends": model_backends or {},
        "created_at": utc_now(),
        "extra": extra or {},
    }


def append_step(provenance: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    result = dict(provenance)
    result["processing_steps"] = [*provenance.get("processing_steps", []), step]
    return result
