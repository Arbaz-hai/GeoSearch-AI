from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.index.metadata_store import MetadataStore


class ExportPipeline:
    def __init__(self, metadata: MetadataStore, output_directory: str | Path):
        self.metadata = metadata
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def export_change_events(self, destination: str | Path | None = None) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(destination) if destination else self.output_directory / f"change_events_{timestamp}"
        json_path = base.with_suffix(".json")
        csv_path = base.with_suffix(".csv")
        events = self._all_events()
        json_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
        rows = []
        for event in events:
            rows.append(
                {
                    "event_id": event["id"],
                    "change_type": event["change_type"],
                    "confidence": event["confidence"],
                    "status": event["status"],
                    "earliest_detected_date": event["earliest_detected_date"],
                    "before_tile_id": event["before_tile_id"],
                    "after_tile_id": event["after_tile_id"],
                    "provenance": json.dumps(event["provenance"]),
                }
            )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["event_id", "change_type", "confidence", "status", "earliest_detected_date", "before_tile_id", "after_tile_id", "provenance"])
            writer.writeheader()
            writer.writerows(rows)
        return {"json": str(json_path), "csv": str(csv_path)}

    def _all_events(self) -> list[dict[str, Any]]:
        events = self.metadata.pending_change_events(100000)
        for audit in self.metadata.audit_log():
            if audit["entity_type"] == "change_event":
                event = self.metadata.get_change_event(audit["entity_id"])
                if event and all(event["id"] != existing["id"] for existing in events):
                    events.append(event)
        return sorted(events, key=lambda event: event["created_at"])
