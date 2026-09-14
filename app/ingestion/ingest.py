from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services import create_services
from app.settings import load_settings
from app.pipelines.ingest_pipeline import SUPPORTED_IMAGE_EXTENSIONS


def run_ingestion(
    input_path: str | Path,
    config_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = load_settings(config_path)
    services = create_services(settings)
    source = Path(input_path).expanduser().resolve()

    if not source.exists():
        print(f"Error: Path does not exist: {source}", file=sys.stderr)
        return {"status": "error", "message": f"Path not found: {source}"}

    if source.is_file():
        if source.suffix.lower() == ".zip":
            extracted = services.ingest._extract_zip(source, lambda msg: None)
            files = services.ingest._supported_files(extracted)
        elif source.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            files = [source]
        else:
            print(f"Error: Unsupported file format: {source.suffix}", file=sys.stderr)
            return {"status": "error", "message": "Unsupported file format"}
    else:
        files = services.ingest._supported_files(source)

    total_discovered = len(files)
    print("=" * 60)
    print("           GeoSearch AI — Satellite Ingestion Engine          ")
    print("=" * 60)
    print(f"\nDiscovered: {total_discovered:,} scenes")

    existing_scenes = {s["source_path"]: s for s in services.metadata.list_scenes()}
    existing_source_ids = {s["source_id"]: s for s in services.metadata.list_scenes()}

    new_files: list[Path] = []
    already_indexed_count = 0

    for f in files:
        f_resolved = str(f.resolve())
        f_stem = f.stem
        if not force and (f_resolved in existing_scenes or f_stem in existing_source_ids):
            already_indexed_count += 1
        else:
            new_files.append(f)

    new_count = len(new_files)
    print(f"Already indexed: {already_indexed_count:,}")
    print(f"New scenes to process: {new_count:,}\n")

    if new_count == 0:
        print("Archive is up-to-date. No new scenes to ingest.")
        print(f"Total indexed scenes: {len(services.metadata.list_scenes()):,}")
        return {
            "status": "up_to_date",
            "discovered": total_discovered,
            "already_indexed": already_indexed_count,
            "processed": 0,
            "failed": 0,
            "total_indexed": len(services.metadata.list_scenes()),
        }

    clip_count = 0
    blip_count = 0
    indexed_qdrant_count = 0
    failed_count = 0
    start_time = time.time()

    for i, file_path in enumerate(new_files, start=1):
        pct = (i / new_count) * 100
        bar_len = 30
        filled = int(bar_len * i // new_count)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\rProcessing: [{bar}] {pct:5.1f}% ({i}/{new_count}) {file_path.name[:25]:<25}")
        sys.stdout.flush()

        try:
            res = services.ingest.ingest_scene(file_path, trigger_change=True)
            if res.get("status") == "indexed":
                clip_count += 1
                if res.get("scene", {}).get("metadata", {}).get("caption"):
                    blip_count += 1
                indexed_qdrant_count += 1
            elif res.get("status") == "duplicate":
                already_indexed_count += 1
        except Exception as e:
            failed_count += 1

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    bar = "#" * 30
    print(f"Processing: [{bar}] 100.0%\n")
    print(f"CLIP embeddings generated : {clip_count:,}")
    print(f"BLIP captions generated   : {blip_count:,}")
    print(f"Indexed in Qdrant vectors : {indexed_qdrant_count:,}")
    print(f"Failed                    : {failed_count:,}\n")

    total_scenes = len(services.metadata.list_scenes())
    print(f"Total indexed scenes: {total_scenes:,}")
    elapsed = time.time() - start_time
    print(f"Ingestion completed in {elapsed:.2f}s")
    print("=" * 60)

    return {
        "status": "success",
        "discovered": total_discovered,
        "already_indexed": already_indexed_count,
        "processed": clip_count,
        "failed": failed_count,
        "total_indexed": total_scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoSearch AI — Offline Satellite Data Ingestion Pipeline",
        prog="python -m app.ingestion.ingest",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to imagery file, directory, or ZIP archive",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (optional)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-indexing of already processed scenes",
    )
    args = parser.parse_args()
    run_ingestion(args.input, args.config, args.force)


if __name__ == "__main__":
    main()
