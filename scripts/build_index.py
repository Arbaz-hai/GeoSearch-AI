from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import create_services
from app.settings import load_settings
from scripts.generate_eval_report import generate_comprehensive_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local vector index using the incremental ingestion path")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()
    settings = load_settings()
    services = create_services(settings)
    started = time.perf_counter()
    results = services.ingest.ingest_directory(settings.resolve_path(arguments.input), progress=print)
    build_seconds = time.perf_counter() - started
    query_started = time.perf_counter()
    services.search.search_text("satellite imagery", limit=5)
    search_latency = (time.perf_counter() - query_started) * 1000
    change_started = time.perf_counter()
    services.change.analyze(settings.default_date_start, settings.default_date_end)
    change_latency = (time.perf_counter() - change_started) * 1000
    report = generate_comprehensive_report()
    print(f"Processed {len(results)} scene(s) in {build_seconds:.3f}s")
    print(f"Report generated: {report}")


if __name__ == "__main__":
    main()
