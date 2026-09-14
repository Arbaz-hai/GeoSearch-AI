from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.tiler import infer_acquisition_date, infer_sensor
from app.settings import load_settings


def manifest_rows(paths: list[Path], source: str, license_name: str) -> list[dict[str, str]]:
    return [
        {
            "scene_id": path.stem,
            "source": source,
            "sensor": infer_sensor(path),
            "acquisition_date": infer_acquisition_date(path),
            "local_path": str(path),
            "license": license_name,
        }
        for path in paths
    ]


def append_manifest(provenance_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    lines = ["", "## Dataset staging runs", "", f"Staged at: {datetime.now(timezone.utc).isoformat()}", "", "| Scene ID | Source | Sensor | Acquisition date | Local path | License |", "| --- | --- | --- | --- | --- | --- |"]
    lines.extend(f"| {row['scene_id']} | {row['source']} | {row['sensor']} | {row['acquisition_date']} | {row['local_path']} | {row['license']} |" for row in rows)
    with provenance_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def stage_local(source_directory: Path, source: str, license_name: str) -> list[dict[str, str]]:
    settings = load_settings()
    extensions = {".tif", ".tiff", ".cog", ".png", ".jpg", ".jpeg"}
    staged: list[Path] = []
    for path in sorted(source_directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        destination = settings.raw_dir / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)
        staged.append(destination)
    rows = manifest_rows(staged, source, license_name)
    append_manifest(settings.root_dir / "docs" / "MODEL_DATASET_PROVENANCE.md", rows)
    return rows


def import_manifest(manifest: Path) -> list[dict[str, str]]:
    settings = load_settings()
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    staged = []
    for row in rows:
        source = Path(row["local_path"])
        destination = settings.raw_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        row["local_path"] = str(destination)
        staged.append(row)
    append_manifest(settings.root_dir / "docs" / "MODEL_DATASET_PROVENANCE.md", staged)
    return staged


def stage_stac(api_url: str, collection: str, bbox: list[float], start_date: str, end_date: str, max_items: int, source: str, license_name: str) -> list[dict[str, str]]:
    settings = load_settings()
    try:
        from pystac_client import Client
    except ModuleNotFoundError as error:
        raise RuntimeError("Install requirements.txt before STAC staging") from error
    client = Client.open(api_url)
    search = client.search(collections=[collection], bbox=bbox, datetime=f"{start_date}/{end_date}", max_items=max_items)
    staged: list[Path] = []
    for item in search.item_collection():
        assets = list(item.assets.values())
        if not assets:
            continue
        asset = next((candidate for candidate in assets if candidate.media_type and ("geotiff" in candidate.media_type.lower() or "cog" in candidate.media_type.lower())), assets[0])
        extension = Path(asset.href.split("?")[0]).suffix or ".tif"
        destination = settings.raw_dir / f"{item.id}{extension}"
        urllib.request.urlretrieve(asset.href, destination)
        staged.append(destination)
    rows = manifest_rows(staged, source, license_name)
    append_manifest(settings.root_dir / "docs" / "MODEL_DATASET_PROVENANCE.md", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage local public imagery and record a reproducible manifest")
    parser.add_argument("--local-directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", default="Manual local staging")
    parser.add_argument("--license", dest="license_name", default="See source terms")
    parser.add_argument("--stac-api")
    parser.add_argument("--collection")
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-items", type=int, default=10)
    arguments = parser.parse_args()
    local_mode = bool(arguments.local_directory)
    manifest_mode = bool(arguments.manifest)
    stac_mode = bool(arguments.stac_api)
    if sum([local_mode, manifest_mode, stac_mode]) != 1:
        parser.error("Provide exactly one staging mode")
    if stac_mode and not all([arguments.collection, arguments.bbox, arguments.start_date, arguments.end_date]):
        parser.error("STAC staging requires --collection, --bbox, --start-date, and --end-date")
    if local_mode:
        rows = stage_local(arguments.local_directory, arguments.source, arguments.license_name)
    elif manifest_mode:
        rows = import_manifest(arguments.manifest)
    else:
        rows = stage_stac(arguments.stac_api, arguments.collection, arguments.bbox, arguments.start_date, arguments.end_date, arguments.max_items, arguments.source, arguments.license_name)
    print(f"Staged {len(rows)} scene(s)")


if __name__ == "__main__":
    main()
