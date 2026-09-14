from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from app.training_data.schema import FeedbackRecord


class FeedbackDataset:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.directory / "feedback.jsonl"
        self.csv_path = self.directory / "feedback.csv"

    def append(self, record: FeedbackRecord) -> None:
        payload = record.to_dict()
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(payload))
            if write_header:
                writer.writeheader()
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in payload.items()})

    def records(self) -> list[dict]:
        if not self.jsonl_path.exists():
            return []
        return [json.loads(line) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def export_split(self, output_directory: str | Path, validation_fraction: float = 0.2) -> dict[str, str]:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        records = self.records()
        split_at = int(len(records) * (1 - validation_fraction))
        paths = {"train": destination / "train.jsonl", "val": destination / "val.jsonl"}
        for name, subset in {"train": records[:split_at], "val": records[split_at:]}.items():
            paths[name].write_text("".join(json.dumps(record) + "\n" for record in subset), encoding="utf-8")
        return {name: str(path) for name, path in paths.items()}
