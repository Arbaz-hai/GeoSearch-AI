from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import load_settings
from app.utils.licenses import model_registry


def append_provenance(path: Path, records: list[dict[str, str]]) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    lines = ["", "## Model staging runs", "", f"Staged at: {stamp}", "", "| Component | Repository | Revision | License | Local path |", "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {record['name']} | {record['repository']} | {record['revision']} | {record['license']} | {record['local_path']} |" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def stage(selected: list[str] | None = None) -> list[dict[str, str]]:
    settings = load_settings()
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("Install requirements.txt before staging model weights") from error
    completed = []
    for record in model_registry(settings):
        if selected and record["name"] not in selected:
            continue
        destination = Path(record["local_path"])
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=record["repository"], revision=record["revision"], local_dir=destination)
        completed.append(record)
    append_provenance(settings.root_dir / "docs" / "MODEL_DATASET_PROVENANCE.md", completed)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage approved public model snapshots for offline inference")
    parser.add_argument("models", nargs="*", help="Optional registry model names")
    arguments = parser.parse_args()
    records = stage(arguments.models or None)
    print(f"Staged {len(records)} model snapshot(s)")


if __name__ == "__main__":
    main()
