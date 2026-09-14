from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import create_services
from app.settings import load_settings


def storage_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def generate(build_seconds: float | None = None, search_latency_ms: float | None = None, change_latency_ms: float | None = None) -> Path:
    settings = load_settings()
    services = create_services(settings)
    stats = services.metadata.statistics()
    scenes = services.metadata.list_scenes()
    bounds = [scene["bounds"] for scene in scenes]
    aoi = [min(item[0] for item in bounds), min(item[1] for item in bounds), max(item[2] for item in bounds), max(item[3] for item in bounds)] if bounds else []
    hardware = f"{platform.system()} {platform.release()} | CPU: {platform.processor() or 'unknown'} | Python: {platform.python_version()}"
    report = "\n".join(
        [
            "# Evaluation Report",
            "",
            "## Run summary",
            "",
            f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
            f"- Indexed AOI: {aoi}",
            f"- Scenes: {stats['scene_count']}",
            f"- Tiles: {stats['tile_count']}",
            f"- Index build time: {build_seconds:.3f} seconds" if build_seconds is not None else "- Index build time: not supplied",
            f"- Storage footprint: {storage_size(settings.data_dir) / (1024 * 1024):.2f} MB",
            f"- Average search latency: {search_latency_ms:.3f} ms" if search_latency_ms is not None else "- Average search latency: not supplied",
            f"- Average change-analysis latency: {change_latency_ms:.3f} ms" if change_latency_ms is not None else "- Average change-analysis latency: not supplied",
            f"- Hardware: {hardware}",
            "",
            "## Index backend",
            "",
            f"- Vector backend: {services.vectors.backend}",
            f"- Semantic vectors: {services.vectors.stats()['semantic_vectors']}",
            f"- Visual vectors: {services.vectors.stats()['visual_vectors']}",
            "",
            "## Model and dataset provenance",
            "",
            "See `MODEL_DATASET_PROVENANCE.md` for declared origin, licensing, staged paths, and scene manifests.",
            "",
            "## Reproduction command",
            "",
            "`python scripts/build_index.py --input data/raw`",
        ]
    )
    output = settings.root_dir / "docs" / "EVALUATION_REPORT.md"
    output.write_text(report + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-seconds", type=float)
    parser.add_argument("--search-latency-ms", type=float)
    parser.add_argument("--change-latency-ms", type=float)
    arguments = parser.parse_args()
    print(generate(arguments.build_seconds, arguments.search_latency_ms, arguments.change_latency_ms))


if __name__ == "__main__":
    main()
