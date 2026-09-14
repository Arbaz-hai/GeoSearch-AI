from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class FeedbackRecord:
    sample_id: str
    task_type: str
    tile_before_id: str | None
    tile_after_id: str | None
    query_text: str | None
    model_prediction: dict[str, Any]
    analyst_label: str
    analyst_id: str | None
    timestamp: str
    provenance: dict[str, Any]
    geo: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feedback_from_event(event: dict[str, Any], analyst_label: str, analyst_id: str | None = None, query_text: str | None = None, task_type: str = "change_detection") -> FeedbackRecord:
    return FeedbackRecord(
        sample_id=str(uuid4()),
        task_type=task_type,
        tile_before_id=event.get("before_tile_id"),
        tile_after_id=event.get("after_tile_id"),
        query_text=query_text,
        model_prediction={"label": event.get("change_type"), "confidence": event.get("confidence")},
        analyst_label=analyst_label,
        analyst_id=analyst_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        provenance=event.get("provenance", {}),
        geo={"aoi_bbox": event.get("location", []), "crs": "EPSG:4326"},
    )
